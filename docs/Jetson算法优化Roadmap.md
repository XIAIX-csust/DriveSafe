# DriveSafe「不牺牲效果」算法优化 Roadmap

> 目标：在 **Jetson（Orin Nano 8G/16G 档，预算 ¥5000）** 上把整体帧率提上去，
> 且尽量**不牺牲效果**（深度/跟踪/测距的质量与新鲜度保持）。
> 与已落地的 `runtime_governor.py`（运行时降频）互补：Governor 解决“吃不消时保流畅”，
> 本文档解决“怎么让同样算力跑出更高上限/更好质量”。

原则：**优先做“效果零损失”的工程优化 → 再做“近无损”的算法级优化 → 最后才碰“有损/需重训”**。

---

## 0. 现状基线（已完成的改动）

- `runtime_governor.py`：按实测 FPS 压力比，自动对 DepthAnything（隔帧/降采样/复用深度图）、
  路面双模型、DeepSort REID、BEV 渲染做降频。**代价**：深度图在间隔帧是旧图、BEV 抽帧。
- `depth_model.py`：FP16（CUDA）、输入降采样、`last_depth_map` 缓存。
- `deep_sort/.../deep_sort.py`：`extract_reid` 参数 + IoU>0.35 特征复用。
- `risk_alerts/.../alerter.py`：Linux aplay/paplay 播放。

本 Roadmap 假设这些已经在；下面每项优化都写明**和 Governor 如何协作**。

---

## A. 效果零损失（工程级，先做）

### A1. 双 CUDA Stream 流水线：YOLO(帧 N) ∥ Depth(帧 N−1)
- **原理**：现在每帧串行 `YOLO → Depth → …`，耗时相加。改成两个 CUDA stream /
  两个 worker 队列：Depth 算的是上一帧、YOLO 算当前帧，两段 GPU kernel 真正并发。
- **落地**：`detect_3d_with_surface.py` / `detect_3d.py` 主循环改造：
  - 新增 `DepthWorker(threading.Thread)`，内部 `queue.Queue(maxsize=2)` 收帧、出深度图；
  - 主循环：`yolo(img)` 的同时 `depth_worker.submit(cur_frame)`；
  - 使用深度图的时刻取 `depth_worker.latest()`（滞后 ≤1 帧）。
- **工作量**：M（1–2 天）。**收益**：帧率从 `1/(t_yolo+t_depth)` → `1/max(t_yolo,t_depth)`，约 1.6–2x。
- **风险**：低（保持单进程，无模型复制）。显存峰值 +1 帧图像缓冲，可忽略。
- **与 Governor**：`should_run_depth()==False` 时 worker 不提交新帧，直接喂最新缓存。
- **验收**：`jtop` 看 GPU 利用率提升；同视频 FPS 对比（`--max-frames 300` 两次运行计时）。

### A2. 预处理线程化（读帧/resize 与推理重叠）
- `LoadImages/LoadStreams` 的读帧 + letterbox 是纯 CPU；独立 producer 线程 + 双缓冲队列，
  主线程只消费。**工作量**：S–M。**收益**：省 5–15ms/帧（1080p）。
- **验收**：CPU 忙碌时 FPS 不塌。

### A3. cudnn.benchmark 在文件模式也开启
- 现在只有 `webcam` 分支 `cudnn.benchmark = True`。改为主循环前统一开启（相同输入尺寸下自动搜最优 kernel）。
- **工作量**：XS。**收益**：YOLO 单帧隐含 5–15% 提升（首帧有自动调优开销，视频/长流程没问题）。

### A4. 相机标定（精度，非算力）
- `config/camera_params.yaml`：缺失时现在用"估的 K"。棋盘格标定一次内参（fx,fy,cx,cy,畸变）
  写入 yaml，3D 测距、风险场、BEV 全部受益。
- **工作量**：S（标定 20 分钟 + 写文件）。**风险**：无。**验收**：已知间距标尺测距误差 < 5%。

### A5. 检测超参合理化
- 现在 `conf_thres=0.01, iou_thres=0.01` → 几乎每条残留框都进 DeepSort REID（又贵又容易串 ID）。
- 建议默认 `conf 0.15~0.25, iou 0.45`，并用 `--classes 0 1 2 3 5 7` 过滤无关类。
- **工作量**：XS。**收益**：REID 输入量减半以上 + 误检/ID 切换减少，**效果通常变好而非变差**。

### A6. 显示/编码与推理分离
- `cv_show`（imshow）、`cv2.VideoWriter`、Streamlit 回调都别占主循环；
  放入交付线程/低优先级线程，主循环只推最新帧。**工作量**：M。**收益**：显示卡顿不再拖推理。

---

## B. 近无损算法级（重点投入区）

### B1. 深度帧间运动补偿（temporal depth warp）
- **问题**：Governor 隔帧复用旧深度图时，目标一动距离就错。
- **做法**：取"上一帧深度图 + 本帧稀疏运动估计"做轻量对齐：
  1. 对上一帧每个检测框中心区域提取少量特征点（ORB / 稀疏 Farneback，≤ 200 点，~2ms）；
  2. 用 LK 光流跟踪到本帧，得到每目标框的位移 (dx,dy)；
  3. 把上一帧该目标区域的深度 patch 按 (dx,dy) 搬移（或整图 `cv2.warpAffine` 仿射近似）；
  4. 深度值再过一个小的时域中值/EMA 平滑（目标级），抑制抖动。
- **工作量**：M。**收益**：Governor 的深度间隔可从 2 提到 4–8 帧，**深度新鲜度几乎不变**；
  距离曲线更平滑（顺带修复"同一辆车距离每两帧跳一下"的观感问题）。
- **风险**：低（只在"该目标深度 patch"上做，不影响其他模块）；高速横切目标光流可能失效 → 回退到复用旧值。
- **与 Governor**：`should_run_depth()==True` 时刷新全图深度并清空 warp 基线；False 时走 warp。
- **验收**：同一段视频，深度间隔 1 vs 8 帧的测距曲线误差 < 5%（离线对比）。

### B2. 几何约束混合测距（每帧免费）
- 单目深度可直接用几何先验兜底，公式：
  ```python
  depth_geometric = fx * REAL_WIDTH[class] / bbox_pixel_width   # fx 来自相机内参
  depth_fused     = α · depth_cnn(box) + (1−α) · depth_geometric
  ```
- 车辆真实宽度已存在 `width[]`（代码里有 `width = [0,0.2,1.85,0.5,0,2.3,0,2.5]`），
  `bbox3d_utils.estimate_3d_box` 也接收宽高先验。
- **关键点**：CNN 深度做**低频校准**——每 K 帧（K=5~10）把 `depth_cnn − depth_geometric` 的偏差
  滑窗估计出来，其余帧只用几何法 + 已校准偏差。这样深度模型可以更省着跑，
  而测距精度反而更高（几何法对遮挡/侧面目标比 CNN 更稳）。
- **工作量**：M。**收益**：深度依赖度大降 + 距离精度提升（两全）。
- **验收**：真实标尺或已知车长数据对比误差；几何法坏例（遮挡/侧向）统计比例 < 10%。

### B3. 速度直接读卡尔曼状态
- DeepSort track 已有 Kalman（`track.mean` 位置/速度）与 `pose_3d`；
  把 `Estimated_speed()` 的 5 帧窗口差分替换为 `track.mean` 的速度分量 + 平滑系数。
- **收益**：速度更稳、省掉 O(N·5) 窗口逻辑；无效果损失。
- **工作量**：S。

### B4. 风险场"判断/渲染分离" + BEV 低分辨率重绘
- 风险判断只依赖矢量公式（SPE/SCF，O(目标数)）；只有给用户看 BEV 才需要 400×400 网格。
- 改法：
  1. 判断路径：不构建网格，直接对每个目标算 `scf`（现有 `risk_engine.calculate_scf`）；
  2. 渲染路径（`should_render_bev()` 为 True 的帧）：网格降到 200×200，绘制后 `cv2.resize` 回显示尺寸。
- **收益**：判断帧省掉网格构建（约 5–15ms）；渲染帧视觉几乎无差。
- **工作量**：M。**验收**：判断结果（HIGH/MEDIUM 计数）逐帧对比无差异。

---

## C. 换算法（对效果通常中性偏正，需回归测试）

### C1. ByteTrack / OC-SORT 替换 DeepSort 的 ReID
- BYTE：高置信框 + 低置信框两级关联，只用 IoU + 卡尔曼，**无外观特征网络**；
  MOT17/MOT20 上 MOTA 与 DeepSort 相当、ID 切换略少（BYTE 论文数据），速度显著更快。
- 本项目 deep_sort 已有 `bbox_3d` 参与匹配，可在 ByteTrack 关联通道里保留 3D 位置距离，
  保住 3D 能力。
- **收益**：砍掉整条 ReID 前向（每目标 5–30ms 按目标数增长）；
  配合 `reid_strategy=off` 或直接删 `extract_reid` 调用。
- **工作量**：L（换引擎 + 3D pose 打通 + 回归对比）。**风险**：中（ID 切换行为变化，需视频回归）。
- **验收**：同视频 MOTA/IDSW 对比旧 DeepSort，允许 IDSW ±10% 内、FPS 提升确认。

---

## D. 有损 / 需重训（谨慎，放在最后）

- **TensorRT INT8**：YOLO/Depth/ReID 导出 engine + 标定集（INT8 精度大概率可接受、不保证）。
- **模型蒸馏**：YOLOv8n-Distill / DepthAnything Small 版替代，需自备训练/蒸馏数据。
- **检测分辨率自适应**：空旷场景 480、城市 640 动态切（`img_size` 切换会触发 letterbox 变化，中风险）。
- **时域深度滤波**（CRF/递归滤波）可追回一部分降频损失，但引入 1–2 帧延迟。

---

## 推荐实施顺序与里程碑

| 阶段 | 内容 | 工作量 | 预期 |
|------|------|--------|------|
| **M1（快赢）** | A3+A5+A4+A6、B3 | 2–3 天 | +15~30% 帧率，测距/速度更准 |
| **M2（核心）** | A1 双 stream 流水线 + B1 深度运动补偿 + B2 混合测距 | 4–6 天 | 帧率趋近 max(YOLO,Depth)，深度间隔 4–8 帧仍准 |
| **M3（可选）** | C1 ByteTrack + A2 + B4 | 5–8 天 | 再砍 ReID 开销，CPU 瓶颈消除 |
| **M4（进阶）** | TensorRT FP16/INT8 导出（YOLO+Depth+ReID） | 3–5 天 | 整体再快 1.5–2.5x |

**每条都配“改前/改后”同视频对比验收**：FPS（`--max-frames 300` 计时）、
测距误差（标尺）、跟踪质量（MOTA/IDSW 抽样标注）、深度新鲜度（间隔帧曲线）。

---

## 与 RuntimeGovernor 的最终关系

```
Governor（已实现）  = 资源不足时的“保底护栏”（降频到不崩溃）
Roadmap（本文档）   = 把“上限”抬高，让护栏很少被触发
两者叠加效果：
    A1 流水线 → 帧率上限≈max(YOLO,Depth)
    B1 运动补偿 → 深度即使 4–8 帧刷新依然准
    B2 混合测距 → 深度依赖更小，可再降频而不损精度
    C1 ByteTrack → REID 整条消失，reid_strategy 不再需要
```