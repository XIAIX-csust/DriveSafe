"""
depth_worker.py — Depth Anything 异步 worker（真 CUDA Stream 版）

对应需求 6（A1 重写）：
    - DepthAnything 推理放到独立后台线程，该线程**独占** transformers pipeline；
    - CUDA 环境下创建独立 `torch.cuda.Stream`，与主线程 YOLO 的默认 stream 真正并行
      （PyTorch 的 Stream 为 non-blocking，不与 legacy default stream 隐式同步）；
    - 只保留最新帧：新帧到来时未处理的旧帧被直接覆盖丢弃（背压丢帧，不降精度）；
    - 主线程通过 `latest()` 取最近一次结果，滞后 ≤ 1 个深度处理周期。

并发安全：
    - `estimate_depth()` 只允许在 worker 线程内调用，主线程用 `seed()` 预置首帧结果；
    - `estimate_depth` 末尾会做 `.cpu().numpy()` / PIL→numpy 的主机同步，
      因此 `latest()` 返回的结果一定已就绪，无需额外 event 等待。

无 torch / 无 CUDA 时自动退化为普通后台线程（不做 stream 切换），CPU 环境也能用。
"""

from __future__ import annotations

import threading
from typing import Optional


class DepthAsyncWorker:
    """独占式深度估计异步 worker：独立 CUDA stream + 最新帧优先。"""

    def __init__(self, estimator, use_stream: bool = True) -> None:
        self.est = estimator
        self._torch = None
        self._stream = None
        self._cv = threading.Condition()
        self._pending = None
        self._latest_depth = getattr(estimator, "last_depth_map", None)
        self._running = True
        self._frames_done = 0
        self._frames_dropped = 0

        self._init_stream(use_stream)

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="depth-async-worker"
        )
        self._thread.start()

    # ── 初始化独立 CUDA stream ───────────────────────────────────────
    def _init_stream(self, use_stream: bool) -> None:
        if not use_stream:
            return
        device = str(getattr(self.est, "pipe_device", getattr(self.est, "device", "cpu")))
        if not device.startswith("cuda"):
            return
        try:
            import torch

            if torch.cuda.is_available():
                self._torch = torch
                self._stream = torch.cuda.Stream()
                print("[depth-async] 独立 CUDA stream 已启用（与主线程并行）")
        except Exception as exc:
            print(f"[depth-async] CUDA stream 不可用，回退默认 stream：{exc}")

    @property
    def using_cuda_stream(self) -> bool:
        return self._stream is not None

    # ── 对外接口 ─────────────────────────────────────────────────────
    def seed(self, depth_map) -> None:
        """预置首帧结果（首帧由主线程同步算出后调用，避免下游拿到 None）。"""
        with self._cv:
            self._latest_depth = depth_map

    def submit(self, frame) -> None:
        """提交最新帧；上一帧未处理完则覆盖（丢旧帧，保最新）。"""
        with self._cv:
            if not self._running:
                return
            if self._pending is not None:
                self._frames_dropped += 1
            self._pending = frame
            self._cv.notify()

    def latest(self):
        """最近一次算完的深度图（尚未出结果时返回 seed 值或 None）。"""
        with self._cv:
            return self._latest_depth

    def frames_done(self) -> int:
        with self._cv:
            return self._frames_done

    def frames_dropped(self) -> int:
        with self._cv:
            return self._frames_dropped

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()
        self._thread.join(timeout=timeout)

    def status_line(self) -> str:
        return (
            f"[depth-async] stream={'cuda' if self._stream is not None else 'default'} "
            f"done={self.frames_done()} dropped={self.frames_dropped()}"
        )

    # ── worker 线程 ──────────────────────────────────────────────────
    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and self._running:
                    self._cv.wait()
                if not self._running:
                    return
                frame = self._pending
                self._pending = None

            try:
                if self._stream is not None:
                    with self._torch.cuda.stream(self._stream):
                        depth = self.est.estimate_depth(frame)
                else:
                    depth = self.est.estimate_depth(frame)
            except Exception as exc:  # 单帧失败不致命，保持上次结果
                print(f"[depth-async] estimate failed, keeping last depth: {exc}")
                with self._cv:
                    self._frames_done += 1
                continue

            with self._cv:
                self._latest_depth = depth
                self._frames_done += 1


if __name__ == "__main__":
    # 极简自测：0.05s/帧的假估计器，验证“最新帧优先 + 丢旧帧 + 可关闭”
    import time

    class _FakeEstimator:
        pipe_device = "cpu"
        last_depth_map = None

        def estimate_depth(self, frame):
            time.sleep(0.05)
            self.last_depth_map = frame
            return frame

    est = _FakeEstimator()
    w = DepthAsyncWorker(est, use_stream=False)
    w.submit("frame-1")
    time.sleep(0.12)
    print("latest after f1:", w.latest())          # frame-1
    for i in range(2, 6):
        w.submit(f"frame-{i}")                     # 连发，只应处理最新的 frame-5
    time.sleep(0.15)
    print("latest after burst:", w.latest())       # frame-5
    print(w.status_line())                          # dropped > 0
    w.shutdown()
    print("thread alive after shutdown:", w._thread.is_alive())  # False
