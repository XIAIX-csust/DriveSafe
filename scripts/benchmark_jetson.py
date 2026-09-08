"""Run the detection pipeline via the real CLI and report FPS for Jetson tuning.

Usage (on the Jetson device):
    python scripts/benchmark_jetson.py --source lanechange.mp4 --frames 60 --depth-onnx
"""

import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="lanechange.mp4")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--device", default="0")
    parser.add_argument("--depth-onnx", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "detect_3d_with_surface.py",
        "--source",
        args.source,
        "--max-frames",
        str(args.frames),
        "--device",
        args.device,
        "--nosave",
        "--no-save-jsonl",
        "--no-view-img",
    ]
    if args.depth_onnx:
        cmd.append("--depth-onnx")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    print(f"Exit code: {proc.returncode}")
    print(f"Wall time: {elapsed:.2f}s")
    if elapsed > 0 and args.frames:
        print(f"Estimated FPS: {args.frames / elapsed:.2f}")

    if proc.returncode != 0:
        print("\n--- stderr tail ---")
        print("\n".join(proc.stderr.splitlines()[-20:]))


if __name__ == "__main__":
    main()
