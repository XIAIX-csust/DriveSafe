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
├── README.md                 # 项目说明（本文档）
└── README_DETAILED.md        # 深度教程
```

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
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --nosave --device cpu
```

可选参数：`--fps 25` 固定帧率节拍（默认）、`--no-pacing` 关闭节拍、`--no-depth-async` 关闭深度异步、`--depth-onnx` 使用 ONNX 深度后端。

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

### 修正

| 改动 | 说明 |
|------|------|
| 深度→距离方向 | DepthAnything 相对深度**数值越大越近**，原公式方向相反；车辆测距与路面隐患距离两处均改为 `distance = 1.0 + (1.0 - depth_value) * 9.0`（范围仍 1–10 m） |
| 风险阈值 | 原 `0.8/0.55/0.25` 与 SCF 量级不匹配（300 帧实测 P50=91 / P95=107），导致 298/300 帧恒为 HIGH；改为 `110/100/90`，实测 HIGH 2.0% / MEDIUM 13.0% / LOW 46.0% / CLEAR 39.0% |
| FP16 判定 | `transformers.pipeline` 会就地修改传入的 `model_kwargs`（`{}` → `{'dtype':'auto'}`），导致 CPU 上误判为 FP16；改为调用前记录 + 传副本 + 以模型实际 dtype 判定 |

### 运行提示

- 推荐入口：Web 端 `streamlit run app.py`；CLI 端 `detect_3d_with_surface.py`。`detect_3d.py` / `main_ui.py` 为旧入口，已不再维护。
- 首次运行需下载 Depth Anything v2 权重，国内建议 `$env:HF_ENDPOINT = "https://hf-mirror.com"`。
- `detect_3d_with_surface.py` 启动时会执行 `check_requirements()`；它在 Windows 下拼的 `pip install 'pkg<版本'` 会因 `<` 被当作重定向而失败，建议先 `pip install -r requirements_common.txt` 装齐依赖。
- CPU 冒烟测试：
  ```bash
  python detect_3d_with_surface.py --source lanechange.mp4 --device cpu --no-view-img --max-frames 30 --no-pacing
  ```

### 已知遗留

- `data_store.py`（480/500/510）与 `app.py`（≥500）仍沿用旧风险尺度，与新阈值不完全一致。
- `combined_risk = max(dynamic_risk, surface_risk)` 中 `surface_risk` 为 0–1，与 SCF（几十~几百）量级不同，路面风险难以单独触发告警。
- 深度刷新率受算力限制（300 帧实测约 73%）；后续可做「深度过期时改用几何测距」的兜底（见 A1 改进计划）。
