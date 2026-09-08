"""
depth_worker.py — Depth Anything 异步流水线 worker（Roadmap A1：双 CUDA Stream 流水线）

原理：
    把 DepthAnything 推理放到独立后台线程，该线程**独占** transformers pipeline；
    主线程每帧只跑 YOLO，同时把当前帧交给 worker 异步计算。
    worker 始终只处理"最新一帧"——新帧到来时未处理的旧帧被直接覆盖丢弃（深度是慢变量，
    丢中间帧无感知损失）。
    端到端帧率从 1/(t_yolo + t_depth) 提升到 ≈ 1/max(t_yolo, t_depth)。

并发安全（重要）：
    - DepthEstimator.estimate_depth() 只允许在 worker 线程内调用；
      主线程永远不要直接调用它，否则两个线程同时进入 transformers pipeline 会互相干扰。
    - 首帧深度由调用方用 seed() 同步预置（此时 worker 尚未处理任何帧，天然无并发），
      之后全部走 submit()/latest() 异步路径。
    - 本模块只依赖标准库，可在任意环境（含 Jetson）直接使用。
"""

from __future__ import annotations

import threading
from typing import Optional


class DepthAsyncWorker:
    """独占式深度估计异步 worker：始终处理最新帧，滞后 ≤ 1 个处理周期。"""

    def __init__(self, estimator) -> None:
        self.est = estimator
        self._cv = threading.Condition()
        self._pending: Optional[object] = None
        self._latest_depth = getattr(estimator, "last_depth_map", None)
        self._running = True
        self._frames_done = 0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="depth-async-worker"
        )
        self._thread.start()

    def seed(self, depth_map) -> None:
        """预置结果（首帧同步算出后调用，避免首帧 None 导致下游崩溃）。"""
        with self._cv:
            self._latest_depth = depth_map

    def submit(self, frame) -> None:
        """提交最新帧；若上一帧尚未处理完，直接覆盖（只保最新，丢弃过期帧）。"""
        with self._cv:
            if not self._running:
                return
            self._pending = frame
            self._cv.notify()

    def latest(self):
        """取最近一次算完的深度图（尚未完成时为 None / 上一次结果）。"""
        with self._cv:
            return self._latest_depth

    def frames_done(self) -> int:
        with self._cv:
            return self._frames_done

    def shutdown(self) -> None:
        """停掉 worker（detect() 结束时调用，释放线程）。"""
        with self._cv:
            self._running = False
            self._cv.notify_all()
        self._thread.join(timeout=2.0)

    # ── worker 线程内部 ──────────────────────────────────────────────
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
                depth = self.est.estimate_depth(frame)
            except Exception as exc:  # 单帧失败不致命，保持上次深度
                print(f"[depth-async-worker] estimate failed, keeping last depth: {exc}")
                with self._cv:
                    self._frames_done += 1
                continue
            with self._cv:
                self._latest_depth = depth
                self._frames_done += 1


if __name__ == "__main__":
    # 极简自测：模拟 0.05s/帧 的深度估计，验证"最新帧优先、丢帧"语义
    import time

    class _FakeEstimator:
        input_scale = 1.0
        last_depth_map = None

        def estimate_depth(self, frame):
            time.sleep(0.05)
            self.last_depth_map = frame
            return frame

    est = _FakeEstimator()
    w = DepthAsyncWorker(est)
    w.submit("frame-1")
    time.sleep(0.12)
    print("latest after f1:", w.latest())  # frame-1
    for i in range(2, 6):
        w.submit(f"frame-{i}")  # 快速连发，应只处理最新的 frame-5
    time.sleep(0.15)
    print("latest after burst:", w.latest())  # frame-5
    print("frames_done:", w.frames_done())
    w.shutdown()
    print("thread alive after shutdown:", w._thread.is_alive())  # False