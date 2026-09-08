#!/usr/bin/env bash
# Jetson Orin Nano Super 8GB 一键部署脚本（在 Jetson 本机运行）
set -euo pipefail

echo "[1/4] 安装系统依赖"
sudo apt-get update
sudo apt-get install -y python3-pip ffmpeg libgl1 libglib2.0-0

echo "[2/4] 安装 Python 依赖（PyTorch 用 JetPack 自带版本）"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_gpu.txt
python3 -m pip install -r requirements_jetson.txt

echo "[3/4] 导出 TensorRT 引擎（路面三模型 + yolov10s + 深度）"
python3 scripts/export_jetson_engine.py

echo "[4/4] 冒烟测试 + 基准测试"
python3 scripts/benchmark_jetson.py --source lanechange.mp4 --frames 60 --depth-onnx

echo "部署完成。现场演示启动：streamlit run app.py"
