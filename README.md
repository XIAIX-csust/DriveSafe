# 驭安DriveSafe — 主动安全预警系统

![驭安DriveSafe](picture/logo.png)

> 基于 **YOLOv10 + 单目深度估计** 的实时车辆检测、3D 测距与主动安全预警系统。
> 面向驾驶辅助（ADAS）场景，提供目标检测、多目标跟踪、深度估计、3D 边界框测距、轨迹预测、风险场评估、路面平整度检测与多端可视化的一站式解决方案。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [详细安装](#详细安装)
- [使用方法](#使用方法)
- [模块详解与二次开发](#模块详解与二次开发)
- [参数配置](#参数配置)
- [输出与产物](#输出与产物)
- [常见问题](#常见问题)
- [应用场景](#应用场景)
- [未来方向](#未来方向)
- [贡献与许可](#贡献与许可)

---

## 项目简介

驭安DriveSafe 是一个基于计算机视觉与深度学习的**主动安全预警系统**，专注于 **实时车辆检测、距离测量、风险评估与可视化**。系统通过单目摄像头输入，使用先进的目标检测与深度估计技术，将二维图像还原为带距离信息的三维驾驶场景，并对周围交通参与者进行轨迹预测与碰撞风险评估，最后通过视觉、语音等多种方式向驾驶员发出预警。

系统沉淀了完整的工程链路：**2D 检测 → 多目标跟踪 → 单目深度估计 → 3D 边界框还原 → 运动预测 → 风险场计算 → 路面风险融合 → 预警输出**，并提供 Streamlit Web、PyQt 桌面、命令行三种使用形态，既可直接运行演示，也可作为二次开发的模块库。

---

## 核心特性

| 功能模块 | 说明 | 关键代码 |
|----------|------|----------|
| 实时目标检测 | 基于 YOLOv10 检测车辆、行人等障碍物，默认过滤 `bicycle / car / motorcycle / bus / truck` 等有效车辆类型 | `detect_3d.py`、`yolov10/` |
| 多目标跟踪 | 基于 DeepSort 对目标进行前后帧关联，分配稳定唯一 ID | `deep_sort/deep_sort/` |
| 单目深度估计 | 基于 Depth Anything v2 估计单帧场景逐像素深度图 | `depth_model.py:DepthEstimator` |
| 3D 边界框与测距 | 由 2D 框 + 深度值反投影出车辆三维位置与尺寸，内置 Kalman 平滑 | `bbox3d_utils.py:BBox3DEstimator` |
| 速度 / 运动分析 | 由跟踪轨迹解算目标运动信息（速度、方向） | `utils/motion_engine.py:MotionEngine` |
| 轨迹预测 | 基于卡尔曼滤波的线性 / 多项式轨迹外推，支持时间衰减加权 | `trajectory_prediction/trajectory_predictor.py` |
| 风险场计算 | 基于速度、位置、路面附着生成高斯风险场，支持安全场重叠判断（SCF） | `risk_field.py:RiskFieldEngine` |
| 路面平整度检测 | 独立模型检测坑洼 / 裂缝，白天与夜间双模型，深度评估路面风险并与整体风险融合 | `road_surface_fusion/` |
| 语音/提示预警 | 危险场景触发中文语音播报与画面提示 | `risk_alerts/sound_processing/`、`risk_alerts/warning_prompt/` |
| 鸟瞰图与热力图 | 俯视视角叠加目标 3D 框、风险热力图、HUD 信息 | `bbox3d_utils.py:BirdEyeView` |
| 多端界面 | Streamlit Web 应用、PyQt5 桌面客户端、命令行推理三种形态 | `app.py`、`main_ui.py`、`detect_3d*.py` |

---

## 系统架构

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   视频输入     │──▶│  YOLOv10 检测 │──▶│  DeepSort 跟踪 │──▶│ Depth Anything│
│  (摄像头/文件)  │   │   2D 目标框   │   │   稳定 ID     │   │   深度图      │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                 │
                                                                 ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  风险场叠加     │◀──│  路面风险融合   │◀──│  3D 边界框还原 │◀──│  2D+深度反投影 │
│  轨迹风险      │   │  坑洼/裂缝     │   │  Kalman 平滑  │   │  (测距)       │
└──────┬───────┘   └──────────────┘   └──────┬───────┘   └──────────────┘
       │                                      │
       ▼                                      ▼
┌──────────────┐                    ┌──────────────────┐
│  语音/画面预警  │                    │  BirdEyeView 可视化 │
│  危险告警      │                    │  鸟瞰图+热力图+HUD  │
└──────────────┘                    └──────────────────┘
```

代码主链路位于 `detect_3d_with_surface.py` 的主流程（对应函数 `detect()`），各模块通过注入方式组合，便于替换实现。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch ≥ 2.2.0 |
| 目标检测 | YOLOv10（`THU-MIG/yolov10`，实测另备 yolov5s/yolov8 模型） |
| 深度估计 | Depth Anything v2（通过 Hugging Face `transformers` pipeline 加载） |
| 目标跟踪 | DeepSort（含 YOLOv3 检测备份实现） |
| 数学 / 滤波 | NumPy, SciPy, filterpy（Kalman） |
| 图像处理 | OpenCV, Pillow |
| 风险场 / 轨迹 | 自研 `RiskFieldEngine`、`MotionEngine`、`TrajectoryPredictor` |
| 界面 | Streamlit（Web）、PyQt5（桌面） |
| 数据 / 可视化 | pandas, matplotlib, seaborn |

---

## 目录结构

```
.
├── app.py                        # Streamlit Web 应用（登录/视频上传/参数调节/实时预览）
├── main_ui.py                    # PyQt5 桌面客户端（登录对话框 + 检测线程 + 实时展示）
├── detect_3d.py                  # 主检测推理（YOLOv10 + DeepSort + 3D 测距）
├── detect_3d_with_surface.py     # 主检测推理（扩展：路面平整度检测 + 风险融合 + JSONL 导出）
├── depth_model.py                # 单目深度估计 DepthEstimator
├── bbox3d_utils.py               # 3D 边界框还原 BBox3DEstimator + 鸟瞰图 BirdEyeView
├── risk_field.py                 # 风险场引擎 RiskFieldEngine
├── data_store.py                 # 线程安全的数据共享存储 DetectionDataStore
├── requirements*.txt             # 环境依赖（CPU 默认 / GPU / 公共）
├── packages.txt                  # 系统级依赖（libgl1 等）
├── README.md / README_DETAILED.md / 作品说明书.md
├── yolov10m.pt / yolov10s.pt / yolov5s.pt   # 检测模型权重（仓库内置）
├── weights/best.pt               # 默认检测权重
├── lanechange.mp4                # 演示视频
│
├── code/                         # 路面检测独立模型
│   └── models/                   #   best.pt(白天) / best_night.pt(夜间) / crack_best.pt(裂缝)
├── road_surface_fusion/          # 路面平整度检测与风险融合
│   ├── detector.py               #   RoadSurfaceDetector 坑洼/裂缝检测
│   ├── depth_runtime.py          #   RobustDepthEstimator 鲁棒深度估计
│   ├── surface_analysis.py       #   RoadSurfaceAnalyzer 路面风险分析
│   ├── risk_fusion.py            #   RoadSurfaceRiskFuser 风险融合
│   ├── structured_output.py      #   StructuredOutputWriter JSONL 结构化导出
│   └── visualization.py          #   RoadSurfaceVisualizer 可视化
├── trajectory_prediction/        # 轨迹预测（predictor / 风险集成 / 可视化）
├── deep_sort/                    # DeepSort 跟踪（含 configs 配置与权重）
├── counter/                      # 断面车流计数 draw_counter
├── risk_alerts/                  # 预警子系统
│   ├── sound_processing/         #   文字转语音播报 alerter
│   └── warning_prompt/           #   中文提示语生成 chinese_prompt
├── utils/                        # YOLOv5 风格工具集 + motion_engine.py（运动引擎）
├── models/                       # YOLOv5 风格模型定义（含 yaml 配置）
├── yolov10/                      # YOLOv10 模型实现（THU-MIG 仓库）
├── data/                         # 数据集配置（coco / argoverse 等 yaml）
├── sounds/                       # 预警音效（危险/坑洼/车辆 warning.wav）
├── picture/                      # 界面与文档素材
├── UI/                           # 前端原型（React + Tailwind，可选）
├── plan/                         # 设计方案与技术文档（含 mermaid 图）
├── docs/                         # 各模块说明
├── runs/                         # 默认推理输出目录
└── user/                         # 本地用户数据（登录测试用）
```

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/XIAIX-csust/DriveSafe.git
cd DriveSafe

# 2. 安装依赖（CPU 环境）
pip install -r requirements.txt

# 3. 运行 CLI 推理（默认使用仓库内置 yolov10s.pt；Depth Anything 权重首次运行自动下载）
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --device cpu

# 4.（可选）启动 Web 界面
streamlit run app.py
```

> 需要 GPU 加速时，先安装 CUDA 版 PyTorch，再 `pip install -r requirements_gpu.txt`。

---

## 详细安装

### 环境要求

- Python **3.9 ~ 3.11**（建议 3.9 / 3.10）
- PyTorch **≥ 2.2**（GPU 版需 CUDA ≥ 11.7，与本地 CUDA 运行时匹配）
- 内存 ≥ 8GB（推荐 16GB）；GPU 显存 ≥ 2GB（使用 GPU 时）
- 安装 `libgl1` 等系统库（Linux）：内置于 `packages.txt`

### 创建虚拟环境

```bash
# conda
conda create -n drivesafe python=3.9
conda activate drivesafe

# 或 venv
python -m venv drivesafe
# Windows
drivesafe\Scripts\activate
# Linux / macOS
source drivesafe/bin/activate
```

### 安装依赖

**方式一：CPU / 无 NVIDIA 独显**（默认）

```bash
pip install -r requirements.txt
```

**方式二：GPU 加速**

```bash
# 1) 先安装与 CUDA 版本匹配的 PyTorch（以 CUDA 12.1 为例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2) 再安装项目其余依赖
pip install -r requirements_gpu.txt
```

> `requirements_common.txt` 同时被两个文件引用，保证检测 / 跟踪 / 深度 / UI 依赖基线一致。
> 深度估计依赖 Hugging Face（`transformers + huggingface_hub`），`transformers` 在首次运行时自动下载模型；无网络环境需提前下载并放到本机 HF 缓存。

### 模型权重

| 权重 | 路径 | 用途 |
|------|------|------|
| `yolov10s.pt` | 仓库根目录 | 目标检测（轻量，实时） |
| `yolov10m.pt` | 仓库根目录 | 目标检测（中量，精度更高） |
| `yolov5s.pt` / `weights/best.pt` | 仓库根目录 | 备用检测模型 |
| `code/models/best.pt` | 仓库内置 | 路面坑洼检测（白天） |
| `code/models/best_night.pt` | 仓库内置 | 路面坑洼检测（夜间） |
| `code/models/crack_best.pt` | 仓库内置 | 路面裂缝检测 |
| Depth Anything v2 | 自动下载 | 单目深度估计（v2_small 默认） |

检测与路面模型均已内置，无需额外下载；**仅 Depth Anything 需要在首次运行时联网下载**。

---

## 使用方法

### 1. CLI 推理：`detect_3d.py`

三维测距主流程（检测 + DeepSort 跟踪 + 深度估计 + 3D 框测距 + 风险场 + 鸟瞰图）。

```bash
python detect_3d.py --source lanechange.mp4 --nosave --device cpu
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `yolov10s.pt` | 检测模型权重路径 |
| `--source` | `lanechange.mp4` | 输入源：视频文件 / 摄像头（如 `0`） |
| `--img-size` | `640` | 推理分辨率（像素） |
| `--conf-thres` | `0.01` | 目标置信度阈值 |
| `--iou-thres` | `0.01` | NMS 的 IOU 阈值 |
| `--device` | `''` | 推理设备：`cpu` 或 `0,1,2` 等 CUDA 编号 |
| `--view-img` | `True` | 是否弹出图像窗口（默认开启；无显示器环境请用 `detect_3d_with_surface.py` 的 `--no-view-img`） |
| `--save-txt` / `--save-conf` | `False` | 保存检测结果标签（含置信度） |
| `--nosave` | `False` | 不保存输出图像 / 视频 |
| `--classes` | - | 按类别过滤，如 `--classes 0 2` |
| `--agnostic-nms` | `False` | 类别无关 NMS |
| `--augment` | `False` | 增强推理（TTA） |
| `--project` / `--name` | `runs/detect` / `exp` | 输出保存目录 |
| `--config_deepsort` | `deep_sort/configs/deep_sort.yaml` | DeepSort 跟踪配置 |

### 2. CLI 推理（带路面检测）：`detect_3d_with_surface.py`

在上面链路基础上，**额外集成路面平整度检测与风险融合**，并支持结构化结果导出。无界面环境下推荐使用本脚本。

```bash
# CPU、无窗口、不保存视频
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --nosave --device cpu

# GPU、启画面可视化
python detect_3d_with_surface.py --source lanechange.mp4 --view-img --device 0

# 导出逐帧 JSONL 结构化结果
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --save-jsonl --structured-dir structured --device cpu
```

在 `detect_3d.py` 参数基础上**追加**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-view-img` | `False` | 禁用画面显示（无头环境必用） |
| `--road-model-dir` | `code/models` | 路面检测模型目录 |
| `--road-conf-thres` | `0.25` | 路面检测置信度阈值 |
| `--depth-backend` | `depth-anything` | 深度估计后端 |
| `--max-frames` | `None` | 最多处理帧数（截取片段调试） |
| `--save-jsonl` / `--no-save-jsonl` | `False` | 启用 / 禁用逐帧 JSONL 导出 |
| `--structured-dir` | `structured` | 结构化输出子目录名 |

### 3. Web 界面（Streamlit）

```bash
streamlit run app.py
```

`app.py` 复用 `detect_3d_with_surface.py` 的检测管线，提供图形化操作：

- 登录 / 注册（本地用户数据，见 `user/`）
- 上传视频（`mp4 / avi / mov`）或使用默认演示视频
- 置信度 / IOU 阈值滑杆实时调节
- 深色 / 浅色主题一键切换
- 开始 / 停止检测，实时预览检测画面与结果

### 4. 桌面客户端（PyQt5）

```bash
python main_ui.py
```

基于 PyQt5 的原生桌面端：

- 登录对话框（`LoginDialog`）
- 独立检测线程（`DetectThread`），界面不阻塞
- 通过 `data_store.py:DetectionDataStore` 线程安全地共享检测帧与目标数据，实时刷新画面

### 5. 轨迹预测示例

```bash
python trajectory_prediction/example_usage.py
```

演示 `Tracker → MotionEngine → TrajectoryPredictor → RiskFieldIntegrator` 的轨迹外推与风险集成调用链。

---

## 模块详解与二次开发

以下 API 均为仓库真实导出，可直接在自定义脚本中 import 使用。

### 目标检测与跟踪

```python
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
tracker = DeepSort(cfg.DEEPSORT.REID_CKPT,
                   max_dist=cfg.DEEPSORT.MAX_DIST,
                   min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                   nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                   max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                   max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                   nn_budget=cfg.DEEPSORT.NN_BUDGET,
                   use_cuda=torch.cuda.is_available())
```

### 深度估计 `depth_model.py`

```python
from depth_model import DepthEstimator

estimator = DepthEstimator(model_size="small", device=None)   # small / base / large
depth_map  = estimator.estimate_depth(image)                  # 逐像素深度图
colorized  = estimator.colorize_depth(depth_map)              # 伪彩色可视化
d_center   = estimator.get_depth_at_point(depth_map, x, y)    # 单点深度
d_region   = estimator.get_depth_in_region(depth_map, bbox, method="median", scale=0.5)  # 框内深度
```

### 3D 边界框与测距 `bbox3d_utils.py`

```python
from bbox3d_utils import BBox3DEstimator, BirdEyeView

bbox3d = BBox3DEstimator(camera_matrix=None, projection_matrix=None, class_dims=None)
box3d  = bbox3d.estimate_3d_box(bbox_2d, depth_value, class_name, object_id=None)  # 反投影还原 3D 框
box3d  = bbox3d.refine_box_3d(box3d, object_id)               # Kalman 时序平滑
pts2d  = bbox3d.project_box_3d_to_2d(box3d)                    # 3D 框投影回图像
image  = bbox3d.draw_box_3d(image, box3d, risk_score=0.0)     # 图像上绘制 3D 框

bird = BirdEyeView(size=(400, 600), scale=25, camera_height=1.2)
bird.draw_box(box3d, risk_score=0.0)
bird.draw_risk_heatmap(risk_map)
bird.draw_hud(max_risk_val, max_risk_id, ego_speed)
view = bird.get_image()
```

### 运动 / 轨迹预测 `utils/motion_engine.py` + `trajectory_prediction/`

```python
from utils.motion_engine import MotionEngine
engine = MotionEngine(fps=30)
tracks = engine.predict_trajectories(tracks, steps=5, method="linear")   # linear / polynomial
decay  = engine.calculate_time_decay(steps=5)                            # 时间衰减因子
info   = engine.get_motion_info(track)                                   # 速度等运动信息

from trajectory_prediction.trajectory_predictor import TrajectoryPredictor, RiskFieldIntegrator
predictor = TrajectoryPredictor(camera_matrix=None)
trajs = predictor.predict_trajectories(tracks, steps=10)
risks = predictor.calculate_risk(trajs, ego_position=(0, 0, 0), time_horizon=5)
integ = RiskFieldIntegrator(predictor)
integ.integrate(tracks, risk_field, ego_position=(0, 0, 0))
```

### 风险场 `risk_field.py`

```python
from risk_field import RiskFieldEngine
engine = RiskFieldEngine(width_meter=16, depth_meter=25, backward_meter=10, grid_res=0.1)
field  = engine.get_gaussian_field(x_c, z_c, v_x=0, v_z=0,
                                   sigma_x=0.8, sigma_z=2.0, v_stretch_factor=0.2, weather_factor=1.0)
scf, ego_f, target_f, overlap = engine.calculate_scf(ego_pos, target_pos, v_ego, v_target, weather_factor=1.0)
vis    = engine.get_visualization_field(x_c, z_c, v_x, v_z)
traj_risk, traj_field = engine.calculate_trajectory_risk(ego_pos, ego_vel, trajectories, velocities_list)
```

### 路面平整度检测与风险融合 `road_surface_fusion/`

```python
from road_surface_fusion import (RoadSurfaceDetector, RoadSurfaceAnalyzer,
                                 RoadSurfaceRiskFuser, RobustDepthEstimator,
                                 StructuredOutputWriter, build_frame_record)

detector = RoadSurfaceDetector(model_dir="code/models", conf_thres=0.25)
detect   = detector.detect(image)               # → 坑洼 / 裂缝检测结果
analysis = RoadSurfaceAnalyzer().analyze(detect, depth_map)   # → 路面风险分析
fused    = RoadSurfaceRiskFuser().fuse(analysis, risk_field)  # → 融合进风险场
writer   = StructuredOutputWriter.build()        # → 结构化 JSONL 导出
```

### 预警输出 `risk_alerts/`

- `risk_alerts/sound_processing/alerter.py`：危险 / 坑洼 / 车辆等场景触发中文 **TTS 语音播报**。
- `risk_alerts/warning_prompt/chinese_prompt.py`：生成对应的中文提示语。

### 线程安全数据共享 `data_store.py`

```python
from data_store import DetectionDataStore
store = DetectionDataStore(maxlen=60)
store.add_frame(frame_data)          # FrameData：帧 + 检测目标列表
latest = store.get_latest()          # 供 UI 线程读取展示
```

---

## 参数配置

### 相机标定（关键）

3D 测距依赖相机内参。默认内参 / 投影矩阵定义在 **`bbox3d_utils.py` 文件头部**的默认常量中，可直接修改；或在构建 `BBox3DEstimator` 时传入 `camera_matrix`、`projection_matrix` 覆盖（见上方 API 示例）。

常见物体的先验三维尺寸（高 / 宽 / 长，单位为米）在 `bbox3d_utils.py` 顶部的 `class_dims` 常量中，可按实际车型调整。

> 本项目**没有** `config/` 目录，相机参数在代码常量中维护，不是外部 yaml 文件。

### 风险场参数

`RiskFieldEngine(width_meter, depth_meter, backward_meter, grid_res)`：

| 参数 | 默认 | 说明 |
|------|------|------|
| `width_meter` | 16 | 风险场横向宽度（米） |
| `depth_meter` | 25 | 风险场纵深（米） |
| `backward_meter` | 10 | 车后回看范围（米） |
| `grid_res` | 0.1 | 网格分辨率（米） |

高斯核 `sigma_x / sigma_z` 及速度拉伸因子 `v_stretch_factor`、天气因子 `weather_factor` 控制风险分布形态。

### DeepSort 跟踪

配置文件：`deep_sort/configs/deep_sort.yaml`（最大距离、最小置信度、最大年龄、`n_init`、`nn_budget` 等）。

### 模型选择

- 检测：`--weights yolov10s.pt`（实时）→ `yolov10m.pt`（更高精度）
- 深度：`DepthEstimator(model_size="small" | "base" | "large")`，越大越精确、越慢
- 路面：`best.pt`（白天）/ `best_night.pt`（夜间）自动按场景切换，`crack_best.pt` 检测裂缝

---

## 输出与产物

| 产物 | 触发方式 | 位置 |
|------|----------|------|
| 标注视频 / 图像 | 默认（不传 `--nosave`） | `runs/detect/<name>/` |
| 检测标签 `*.txt` | `--save-txt`（可加 `--save-conf`） | 同上 |
| 逐帧结构化 JSONL | `--save-jsonl` | `structured/`（`--structured-dir` 可改） |
| 集中画面预览 | `--view-img` / Web / 桌面端 | 屏幕 |

`structured/` 下的 JSONL 每条对应一帧，包含目标 ID、类别、2D/3D 框、距离、速度、风险评分等信息，便于后续离线上报、回放或训练。

---

## 常见问题

**Q1：首次运行报模型下载失败？**
Depth Anything 权重从 Hugging Face 自动下载，需保持网络通畅；若网络受限，请提前下载对应权重并配置 HF 缓存目录（`HF_HOME`）。

**Q2：GPU 显存不足（OOM）？**
改用轻量模型：`--weights yolov10s.pt` + `DepthEstimator(model_size="small")`；或降低 `--img-size`（如 480）；或直接用 CPU：`--device cpu`。

**Q3：检测精度不够？**
换大模型（`yolov10m.pt` / `base`-`large` 深度模型）、调高 `--conf-thres`、核对相机标定参数 `camera_matrix`。

**Q4：无显示器环境运行报 `cv2.imshow` 错误？**
使用 `detect_3d_with_surface.py --no-view-img`；`detect_3d.py` 默认 `--view-img` 会弹窗，勿在纯服务器上误用。

**Q5：运行速度慢？**
优先 GPU（`--device 0`）；其次减小 `--img-size`、降帧采样、升级轻量模型；深度估计是最大耗时项，可降 `model_size`。

**Q6：摄像头输入？**
`--source 0`（或 1、2…）即可接入本机摄像头；Web 端同样支持。

---

## 应用场景

- **高级驾驶辅助（ADAS）**：实时监测前方车辆 / 行人，计算碰撞风险，视觉 + 语音双通道预警。要求 ≥ 30 FPS 时建议 GPU + 轻量模型。
- **交通监控**：对监控画面进行车流检测、断面计数（`counter/`）、异常行为与事故识别。
- **园区 / 停车场与低速作业**：障碍物检测、车位占用判断、低速碰撞预警。

---

## 未来方向

1. 多传感器融合：接入雷达 / LiDAR，提升恶劣与盲区场景可靠性
2. 极端天气适应：雨、雪、雾场景专项增强
3. 场景语义理解：交叉口、环岛等复杂场景
4. 轻量化与端侧部署：TensorRT / ONNX 导出，移动端
5. 行为预测：预测其他交通参与者意图，进一步提前预警
6. 个性化预警策略：按驾驶风格调节风险等级与提示强度

---

## 贡献与许可

欢迎提交 Issue、PR 或以任意形式参与改进：

1. Fork 本仓库
2. 创建功能分支 `git checkout -b feature/xxx`
3. 提交更改并推送
4. 发起 Pull Request

**许可证**：README 声明使用 **MIT License**（对应文件暂未随仓库附上，如需正式开源请补充 `LICENSE` 文件）。

---

**驭安DriveSafe** — 为安全驾驶保驾护航 🚗