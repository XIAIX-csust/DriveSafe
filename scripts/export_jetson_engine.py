"""Build TensorRT engines for Jetson Orin Nano/NX deployment.

Run this ON the Jetson device (requires JetPack with TensorRT, PyTorch,
ultralytics and onnx installed). TensorRT engines are device-specific and
cannot be built on this Windows machine.
"""

import subprocess
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def export_ultralytics_engine(weights: Path, imgsz: int = 512, half: bool = True) -> None:
    model = YOLO(str(weights))
    model.export(format="engine", imgsz=imgsz, half=half, simplify=True)
    print(f"Exported {weights.name} -> engine")


def export_depth_onnx_to_engine(onnx_path: Path, engine_path: Path, fp16: bool = True) -> None:
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if fp16:
        cmd.append("--fp16")
    subprocess.run(cmd, check=True)
    print(f"Exported {onnx_path.name} -> {engine_path.name}")


def main() -> None:
    # Road-surface models (day / night / crack). The detector automatically
    # prefers the .engine files once they exist.
    for name in ["best", "best_night", "crack_best"]:
        export_ultralytics_engine(ROOT / "code" / "models" / f"{name}.pt")

    # Main detector (optional: the current loader uses PyTorch; switching to
    # this engine requires the loader swap described in the docs).
    export_ultralytics_engine(ROOT / "yolov10s.pt", imgsz=512)

    # Depth model: ONNX Runtime with CUDA/TensorRT EP is usually sufficient on
    # the Jetson. Build a dedicated TensorRT engine only if you want max FPS.
    export_depth_onnx_to_engine(
        ROOT / "depth_anything_v2_small.onnx",
        ROOT / "depth_anything_v2_small.engine",
    )


if __name__ == "__main__":
    main()
