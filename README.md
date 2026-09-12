# 驭安DriveSafe 主动安全预警系统

## 项目概述

驭安DriveSafe是一个基于计算机视觉和深度学习的主动安全预警系统，专注于实时车辆检测、距离测量、风险评估和可视化。该系统使用先进的3D目标检测和深度估计技术，为驾驶员提供周围环境的实时风险分析，帮助提高驾驶安全性。

## 核心功能

- **实时目标检测**：使用YOLOv10模型检测车辆、行人和其他障碍物
- **3D边界框估计**：基于深度信息和相机参数计算目标的3D位置和尺寸
- **深度估计**：使用Depth Anything v2模型估计场景深度
- **目标跟踪**：使用DeepSort算法跟踪目标并分配唯一ID
- **轨迹预测**：基于卡尔曼滤波的线性/多项式轨迹预测
- **风险场计算**：基于论文公式计算车辆周围的风险场
- **路面平整度检测**：检测路面坑洼和裂缝，评估路面风险
- **鸟瞰图可视化**：提供场景的鸟瞰视图，显示目标位置和风险分布
- **速度估计**：基于目标运动计算速度
- **用户友好界面**：使用Streamlit构建的直观界面

## 技术栈

- **深度学习框架**：PyTorch
- **目标检测**：YOLOv10
- **深度估计**：Depth Anything v2
- **目标跟踪**：DeepSort
- **图像处理**：OpenCV
- **用户界面**：Streamlit
- **其他库**：NumPy, SciPy, Transformers

## 项目结构

```
├── app.py                    # Streamlit应用程序
├── detect_3d_with_surface.py # 主检测和处理逻辑
├── depth_model.py            # 深度估计模型
├── bbox3d_utils.py           # 3D边界框估计和可视化
├── risk_field.py             # 风险场计算
├── depth_worker.py           # 深度异步 worker（独立 CUDA stream）
├── frame_pacer.py            # 固定帧率节拍器
├── data_store.py             # Streamlit 侧内存数据仓（风险分级/趋势/统计）
├── risk_alerts/              # 语音告警 + 中文横幅提示
├── scripts/                  # ONNX/TensorRT 导出、Jetson 部署、FPS 基准
├── docs/                     # 测试结果与 Jetson 部署实测文档
├── road_surface_fusion/      # 路面平整度检测和风险融合
├── deep_sort/                # DeepSort目标跟踪
├── yolov10/                  # YOLOv10模型
├── models/                   # 模型权重
├── utils/                    # 工具函数
├── data/                     # 测试数据
├── code/                     # 路面检测模型
├── requirements.txt          # 依赖项
├── requirements_common.txt   # 通用依赖项
├── requirements_gpu.txt      # GPU依赖项
├── requirements_jetson.txt   # Jetson 补充依赖（onnxruntime-gpu 等）
├── README.md                 # 项目说明（本文档）
└── README_DETAILED.md        # 早期详细教程（部分路径已过期，以本文档为准）
```

> ⚠️ `README_DETAILED.md` 写于早期版本，其中 `main_ui.py`、`trajectory_prediction/`、`utils/motion_engine.py`、`deep_sort/webserver/` 等路径**已删除**，命令照抄会失败。

## 安装说明

### 1. 克隆仓库

```bash
git clone <repository-url>
cd Distance-measurement
```

### 2. 安装依赖

根据您的环境选择合适的依赖文件：

#### CPU环境

```bash
pip install -r requirements_common.txt
```

#### GPU环境

```bash
# 首先安装与CUDA版本匹配的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 然后安装其他依赖
pip install -r requirements_gpu.txt
```

### 3. 下载模型权重

系统默认使用以下模型：
- YOLOv10s：用于目标检测
- Depth Anything v2 Small：用于深度估计
- 路面检测模型：用于检测坑洼和裂缝

模型会在首次运行时自动下载。

## 使用方法

### 1. 启动Streamlit应用

```bash
streamlit run app.py
```

### 2. 运行带路面检测的检测

```bash
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --device cpu
```

> 默认**不保存**标注视频（逐帧 1080p 编码是纯 CPU 开销，20~50ms/帧，明显拖低帧率）；需要保存时加 `--save`。
> 默认会弹 OpenCV 窗口显示（`--no-view-img` 关闭）；基准测试建议关掉窗口与保存，只留结构化 JSONL。

可选参数：`--fps 25` 固定帧率节拍（默认）、`--no-pacing` 关闭节拍、`--no-depth-async` 关闭深度异步、`--depth-onnx` 使用 ONNX 深度后端、`--road-roi-top 0.5` 路面模型 ROI 起点（0 表示整幅，越小保留越多画面上部）、`--no-road-parallel` 关闭主/辅路面模型并行。

### 3. FPS 基准测试（上板调参用）

```bash
python scripts/benchmark_jetson.py --source lanechange.mp4 --frames 60 --depth-onnx
```

### 4. 本机实测数据

本机（Windows / 纯 CPU）的 300 帧实测、阈值标定前后对比、方向修正前后对照见 [docs/测试结果.md](docs/测试结果.md)（原始逐帧数据在 `docs/test-results/`）。

## 深度教程

如需更详细的项目信息、安装指南、代码结构分析和高级使用方法，请参考 [README_DETAILED.md](README_DETAILED.md) 文件。

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议。请遵循以下步骤：

1. Fork仓库
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 致谢

- YOLOv10团队：提供高效的目标检测模型
- Depth Anything团队：提供先进的深度估计模型
- DeepSort团队：提供可靠的目标跟踪算法
- 所有开源贡献者

---

**驭安DriveSafe** - 为安全驾驶保驾护航

---

## 本次改动（2026-09，已提交到 `main`）

### 新增能力

| 改动 | 说明 | 文件 / 参数 |
|------|------|------------|
| 固定 25 FPS 节拍 + 丢帧背压 | 帧率写死、不动态调整；离线按源帧率确定性抽帧，实时按墙钟节拍，积压时丢帧而非降分辨率/精度 | `frame_pacer.py`；`--fps` / `--no-pacing` |
| 深度异步 worker | 独立线程 + 独立 `torch.cuda.Stream`，只保留最新帧，滞后 ≤ 1 个深度处理周期 | `depth_worker.py`；`--no-depth-async` 退回同步 |
| CUDA FP16 | 仅 CUDA 路径启用 FP16，启动时小图自检，不兼容自动回退 FP32 | `depth_model.py` |
| Linux/Jetson 语音告警 | Linux 自动探测 `paplay → aplay → pw-play → ffplay → pygame`，非阻塞播放现有 wav；Windows 仍走 winsound | `risk_alerts/sound_processing/alerter.py` |
| 路面模型 ROI 裁剪 + 主/辅并行 | 只把画面下部（默认 50%）送路面模型、day/night 与 crack 并行推理。实测 150 帧：**0.635 → 0.576 s/帧（约 9%）** | `road_surface_fusion/detector.py`；`--road-roi-top` / `--no-road-parallel` |
| 深度走 float 预测 | 直接用 `predicted_depth`（float32）替代 pipeline 的 uint8 PIL，去掉 PIL 往返与 8bit 量化：转换 15.3 → 7.9 ms/帧，可区分深度级数 **256 → 154 万** | `depth_model.py` |
| 风险趋势图窗口化 | Web 端折线图默认只显示**最近 150 帧**（点不再累积糊成一片），数据不裁剪，图上拖动/滚轮可回看更早的帧 | `app.py` `render_risk_trend()` |

### 修正

| 改动 | 说明 |
|------|------|
| 深度→距离方向 | DepthAnything 相对深度**数值越大越近**，原公式方向相反；车辆测距与路面隐患距离两处均改为 `distance = 1.0 + (1.0 - depth_value) * 9.0`（范围仍 1–10 m） |
| 风险阈值 | 原 `0.8/0.55/0.25` 与 SCF 量级不匹配（300 帧实测 P50=91 / P95=107），曾出现 298/300 帧恒为 HIGH 的误报。修正检测阈值 `conf 0.01→0.25 / iou 0.01→0.45` 后重新标定（见 [docs/测试结果.md](docs/测试结果.md) §2.1），现行阈值为 **`50/40/30`**，实测 HIGH 0.3% / MEDIUM 1.7% / LOW 1.0% / CLEAR 97.0% |
| 风险等级口径统一 | 同一个 SCF 此前在 5 处各写了一套阈值（decision `50/40/30`、`data_store` `510/500/480`、画框 `0.7/0.3`、BEV 提示 `0.4/0.8`、仪表盘 `400~610`），导致"画面判 HIGH、侧边栏显示安全"、3D 框恒红、BEV 中文提示恒不触发。现全部收敛到 `risk_field.RISK_THRESHOLDS` 单一真源，动态风险与路面风险**各自按本尺度分级后取更严重者** |
| FP16 判定 | `transformers.pipeline` 会就地修改传入的 `model_kwargs`（`{}` → `{'dtype':'auto'}`），导致 CPU 上误判为 FP16；改为调用前记录 + 传副本 + 以模型实际 dtype 判定 |

### 运行提示

- 推荐入口：Web 端 `streamlit run app.py`；CLI 端 `detect_3d_with_surface.py`。旧入口 `detect_3d.py` / `main_ui.py` **已删除**（见下方"代码清理"）。
- 首次运行需下载 Depth Anything v2 权重，国内建议 `$env:HF_ENDPOINT = "https://hf-mirror.com"`。
- `detect_3d_with_surface.py` 启动时会执行 `check_requirements()`；它在 Windows 下拼的 `pip install 'pkg<版本'` 会因 `<` 被当作重定向而失败，建议先 `pip install -r requirements_common.txt` 装齐依赖。
- CPU 冒烟测试：
  ```bash
  python detect_3d_with_surface.py --source lanechange.mp4 --device cpu --no-view-img --max-frames 30 --no-pacing
  ```

### 已知遗留

- **深度仍是相对映射**：`distance = 1.0 + (1.0 - depth_value) * 9.0` 只保证顺序正确，绝对值恒落在 1–10 m（实测车辆集中在 7–10 m），**不是真实米数**；要拿真实距离需做标定或换 metric 深度模型（Roadmap B2）。
- `config/camera_params.yaml` 缺失时用估算内参（fx ≈ 图像宽度），几何测距会系统性偏大，建议标定一次写入。
- `data_store._determine_risk_type` 的距离判据（`distance < 3` / `< 5`）在当前深度尺度下不会触发。
- 结构化输出里的 `speed_mps` 实际单位是 **km/h**（风险场内部也按 km/h 处理，`/3.6` 换算），仅字段名不符。
- 深度刷新率受算力限制（CPU 300 帧实测约 73%）；A1 的"深度过期改用几何测距"兜底尚未实现（P1–P8 计划见会话记录）。
- Linux/Jetson 语音告警后端只做过单元级验证，**未在真机运行过**；TensorRT/ONNX 路径（`--depth-onnx`、`.engine` 优先加载）同理，需上板验证。
- `utils/general.py` 的 `check_requirements()` 用 `open(path, 'r')` 读 requirements 文件（跟随系统 locale），**该文件含非 ASCII 字符会在 Windows 上崩溃**；pip 输出 `.decode()` 也有同类问题。
- **路面 ROI 50% 有确定的召回风险**：只保留画面下半部，位于 `y < 540` 的远处缺陷（如参考 run 中 bbox y1=285 的 Crack）**完全看不到**；在意远处缺陷时把 `--road-roi-top` 降到 0.25–0.3（省时相应减少）。详见 [docs/测试结果.md](docs/测试结果.md) §7。
- **端到端延迟未做优化**：当前板子只负责采集与传输，若采用"服务器推理 + 回传渲染画面"会因两次编解码 + 抖动缓冲导致高延迟；建议改为只回传结构化数据、显示端本地渲染（方案②），仓库内尚无网络传输层。

### 代码清理（2026-09）

已移除两类不再使用的代码，**文件仍保留在 git 历史中**，需要时可直接取回：

| 类别 | 内容 | 取回方式 |
|------|------|----------|
| 未接线/重复代码 | 旧入口 `detect_3d.py`+`main_ui.py`、`_backup/`、`UI/`、`counter/`、`plan/`、`picture/`、`trajectory_prediction/`、`deep_sort/{detector,webserver,DeepSORT_Monet_traffic}`（仅保留中文字体）、死函数 `generate_bev_map()` 等 | `git log --diff-filter=D --name-only --oneline` 找到删除提交，再 `git checkout <提交>~1 -- <路径>` |
| 训练/实验脚本 | `yolov10/train.py`、`yolov10/models/*`、`yolov10/utils/{datasets,activations}.py`、`yolov10/yolov10/*`、`utils/{loss,activations}.py`、`utils/{aws,wandb_logging}/*`、`models/{export.py,yolov10/*}` | 同上 |

> ⚠️ 注意：`models/yolo.py` 与 `utils/autoanchor.py` **不能删** —— `yolov10s.pt` / `weights/best.pt` 反序列化时会动态 import `models.yolo`，而它又 import `utils.autoanchor`，静态分析会把它们误判成"未使用"。

