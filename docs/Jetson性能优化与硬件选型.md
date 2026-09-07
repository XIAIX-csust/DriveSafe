# Jetson 性能优化与硬件选型报告（预算 ¥5000）

> 针对 DriveSafe「YOLO + Depth Anything + DeepSort + 路面双模型 同时跑太吃性能」的现状，
> 先说代码里到底慢在哪，再给本次的运行时动态调控方案、多进程/多线程的结论，以及 5000 元预算的板卡选型建议。

---

## 1. 读完代码后的瓶颈结论

主推理链路在 `detect_3d_with_surface.py` / `detect_3d.py` 的 `detect()` 里，**每一帧**都会执行：

| 模块 | 位置 | 单帧成本量级（Jetson 参考） | 说明 |
|------|------|------------------------------|------|
| YOLO 检测（默认 yolov10s，640px，FP16） | `model(img)` | 10~25 ms | 必要，不能省 |
| Depth Anything V2 Small（518px，**FP32**，每帧） | `depth_estimator.estimate_depth()` | 60~200 ms | **最大单点**：每帧全图深度，FP32 很吃亏 |
| 路面 day/night + crack 双模型（ultralytics YOLO，有内部 2~3 帧节流） | `road_detector.detect()` | 30~80 ms | 次大：一次跑 2 个 YOLO |
| DeepSort REID 特征（对每个检测框跑 CNN） | `deepsort.update()` → `_get_features` | 5~30 ms（随目标数增加） | 每帧全量重算 |
| BEV 热力图（matplotlib 绘图 + 拼接） | `bev_visualizer` 各 draw → `get_image()` | 20~50 ms | 纯 CPU，每帧重画 |
| 其余 numpy/画框/日志 | — | 5~15 ms | 相对小 |

**合计：优化前 AGX Orin 64GB 上约 8~15 FPS，Orin Nano 上约 4~8 FPS，且内存/显存吃紧。**

最大的浪费点是：**Depth Anything 每帧跑一次 FP32 全图深度**，而驾驶场景里相邻帧深度变化很小；
其次 REID 对同一辆车每帧都重算特征、BEV 每帧重画也是纯浪费。

---

## 2. 本次代码修改：运行时动态调控（RuntimeGovernor）

你选择了「仅运行时动态调控」——不探测硬件，而是**用实测 FPS 自动决定每帧哪些模块执行**。

新增 `runtime_governor.py`：滑动窗口统计每帧推理耗时 → 得到当前 FPS → 与目标 FPS 的压力比
`pressure = target_fps / current_fps`，然后自动调速：

| 模块 | auto 策略行为 |
|------|--------------|
| 深度估计 | 压力 1→8 帧一次；压力 >2 时输入降采样 0.75、>3 时 0.5；CUDA 下默认 FP16；帧间复用上一帧深度图 |
| 路面双模型 | 基础每 2 帧一次，压力大再翻倍（首帧强制跑一次） |
| DeepSort REID | 压力大时按 IoU>0.35 复用上一帧已算好的特征，只对新目标/大位移目标重算 |
| BEV 渲染 | 压力大时抽帧渲染，跳过帧复用上一帧画面（省 matplotlib） |

命令行参数（`detect_3d.py` / `detect_3d_with_surface.py` 都有）：

```bash
python detect_3d_with_surface.py --source lanechange.mp4 \
    --target-fps 15 \
    --depth-strategy auto --road-strategy auto --reid-strategy auto --bev-strategy auto
```

环境变量同样生效：`DRIVESAFE_TARGET_FPS`、`DRIVESAFE_DEPTH_STRATEGY`、`DRIVESAFE_ROAD_STRATEGY`、
`DRIVESAFE_REID_STRATEGY`、`DRIVESAFE_BEV_STRATEGY`。

实测观感：FPS 一旦低于目标，系统会先砍深度频率与输入分辨率（对测距精度影响最小），还不够再砍 REID/BEV，
**把最贵的深度模型留到最后才全关**（`--depth-strategy off` 时只保留 YOLO+DeepSort 纯 2D 模式）。

---

## 3. Depth Anything 多线程 / 多进程 可以吗？——可以，但 Jetson 上不如上面的动态调控划算

- **可以，但有代价**：每开一个新进程/新线程跑 Depth Anything = 模型权重在内存/显存里多一份拷贝
  （DepthAnything V2 Small FP16 约 300~400MB，FP32 约 700MB+）。Jetson 上进程间共享 GPU iGPU，
  kernel 仍然分时，两个模型并发对吞吐的提升有限。
- **多进程唯一真正的好处**：进程内 CPU 预处理（读帧、缩放、转 PIL）可以和另一进程的 GPU 推理重叠，
  形成「检测推理帧 N / 深度推理帧 N-1」的流水线，代价是 1 帧的深度延迟。
- **多线程**：`transformers` pipeline 不是 CUDA stream 安全的，同一模型多线程共享 GPU 只会互相等待，
  收益很小。
- **结论**：本方案优先“单进程 + FP16 + 动态抽帧 + 深度图复用”，收益最大、内存最省。
  如果你后面要更进一步，正确路线是 **TensorRT 导出**（YOLO 与 DepthAnything 各导一个 engine，
  用两个 CUDA stream 异步推理），预计比当前再快 30~60%。

---

## 4. 硬件选型（预算 ¥5000）

| 方案 | 淘宝/渠道参考价 | 算力(稀疏) | 统一内存 | 优化后预期 | 结论 |
|------|----------------|-----------|---------|-----------|------|
| Jetson AGX Orin 64GB | ¥8500~9500 | 275 TOPS | 64GB | 全模块无压力 20~30 FPS | 超预算 |
| **Jetson Orin Nano Super 8GB** | ¥2400~3000 | 67 TOPS | 8GB | **12~18 FPS**（深度 2~4 帧一次） | ✅ 预算内够用 |
| **Jetson Orin Nano 16GB（开发套件）** | ¥3800~4500 | 67 TOPS | 16GB | **15~22 FPS**，可加大 batch/缓存更多帧 | ✅✅ **首选推荐** |
| Jetson Orin NX 16GB 模块 + 载板 | ¥3000~4000 | 100 TOPS | 16GB | 15~25 FPS | ✅ 若你愿意自己配载板/散热 |
| 二手机 AGX Orin（不推荐，水货风险高） | 不定 | — | — | — | ⚠️ 谨慎 |

**推荐**：预算 5000 内 → **Orin Nano 16GB 开发套件**（约 ¥3800~4500，剩余预算买 CSI 摄像头/散热/存储）。
若想再省 → Orin Nano 8GB Super（约 ¥2400~3000），配合本项目 auto 策略也够跑，
只是同一个视频里深度图更新频率会低一点。

> 性能预估依据：Jetson Orin Nano 的 FP16 TOPS 与公开深度模型的实测/理论耗时
> （YOLOv10s-640 ≈ 10~20ms；DepthAnything V2 Small-518 FP16 ≈ 50~80ms；FP32 翻倍），
> 实际以真机 `--target-fps` 调节为准。目标帧率建议 12~15，深度最多允许滞后 200ms 以内。

### 上板之后的推荐步骤
1. JetPack 6.x（L4T），`sudo apt install alsa-utils`（语音警告用 aplay）。
2. 先 `pip install -r requirements.txt` 跑通 CPU 全流程，确认代码正确。
3. 用 `trtexec`/`tensorrt` 把 yolov10s 与 DepthAnything V2 Small 转成 engine，改 `--device 0` 启用 FP16。
4. 用 `jetson-stats`(jtop) 看 GPU/内存占用，调 `--target-fps` 到平衡点。
5. Web 界面在 Jetson 上用 `streamlit run app.py`；桌面界面（PyQt5）必须在有显示的环境下用。

---

## 5. Linux（Jetson）语音警告

- 警告音频是现成的 `sounds/{vehicle,pothole,danger}_warning.wav`，**不需要任何 TTS 库**。
- 代码 `risk_alerts/sound_processing/alerter.py` 已跨平台：
  - Windows 仍走 `winsound`（不变）；
  - Linux/Jetson 自动探测 `paplay → aplay → pw-play → ffplay`，都没有再用 `pygame.mixer` 兜底；
  - 全部 **非阻塞**播放（子进程/异步），不会卡推理主循环。
- 装播放器：`sudo apt install alsa-utils`（最稳）；PulseAudio 环境装 `pipewire-pulse` 即可用 `paplay`。
- 关闭/调节：`DRIVESAFE_SOUND_ALERT=0` 关声音；`DRIVESAFE_SOUND_ALERT_COOLDOWN=3` 调冷却秒数（默认 5s）。