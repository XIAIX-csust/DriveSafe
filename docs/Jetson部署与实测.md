# Jetson Orin Nano Super 8GB 部署与实测

## 一、目标

在 Jetson Orin Nano Super 8GB（67 TOPS）上完成「摄像头采集 → 本地推理 → 风险判断 → 展示 + 语音」的
现场演示，目标帧率 ≥ 15 FPS。

## 二、硬件清单

- 计算板：Jetson Orin Nano Super 8GB 开发套件（约 ¥2200~3400）
- 摄像头：USB 或 CSI 1080p 广角
- 扬声器：USB 或 3.5mm
- 屏幕：HDMI 触摸屏（现场演示）
- 存储：NVMe SSD 或高速 SD 卡（≥ 64GB）

## 三、环境准备

1. 用 NVIDIA SDK Manager 或 balenaEtcher 烧录 JetPack 6.x。
2. 首次开机完成初始化，确认 `nvcc --version`、`python3 -c "import torch; print(torch.cuda.is_available())"` 正常。
3. 把仓库同步到 Jetson（或从 GitHub clone）。

## 四、一键部署

```bash
bash scripts/deploy_jetson.sh
```

脚本会依次：装系统依赖 → 装 Python 依赖 → 导出 TensorRT 引擎 → 跑基准测试。

手动方式：

```bash
pip install -r requirements_gpu.txt
pip install -r requirements_jetson.txt
python scripts/export_jetson_engine.py
python scripts/benchmark_jetson.py --source lanechange.mp4 --frames 60 --depth-onnx
```

## 五、模型导出说明

| 模型 | 输出 | 用途 |
| --- | --- | --- |
| `code/models/best.pt` 等三模型 | `.engine` | 路面检测（`detector.py` 自动优先加载） |
| `yolov10s.pt` | `.engine` | 主检测（当前默认仍走 PyTorch，可后续切） |
| `depth_anything_v2_small.onnx` | `.engine` | 深度估计（ONNX 直接可用，engine 更快） |

> TensorRT engine 只能在生成它的设备上运行，必须在 Jetson 本机构建。

## 六、实测与调参

```bash
python scripts/benchmark_jetson.py --source lanechange.mp4 --frames 60 --depth-onnx
```

如果帧率不够，按顺序调：

1. `--img-size` 降到 416；
2. 深度用 `--depth-onnx`（已默认推荐）；
3. 调 `--target-fps` 和 `--depth-strategy / --road-strategy / --reid-strategy / --bev-strategy`，让运行时调度器自动降频；
4. 主检测换 TensorRT engine（把 `yolov10s.pt` 换成导出的 `.engine` 加载路径）。

## 七、现场启动

```bash
streamlit run app.py
```

接入摄像头后把视频源切到摄像头编号（默认 `0`）。
