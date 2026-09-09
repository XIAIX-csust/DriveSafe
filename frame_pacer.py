"""
frame_pacer.py — 固定帧率节拍器（帧率写死，不做任何动态调整）

对应需求 5：
    - 目标帧率**写死**（默认 25 FPS），不随负载自适应；
    - **离线模式**（realtime=False，文件/批处理）：按源帧率确定性地抽帧到目标帧率。
      源 30fps → 目标 25fps 时均匀丢 1/6 帧；源 ≤ 目标时逐帧全处理，绝不乱丢。
    - **实时模式**（realtime=True，摄像头/网络流）：按墙钟节拍输出目标帧率；
      距上次处理已超过一个目标周期即视为"准时"（源比目标慢时不误丢帧）；
      只有在一个周期内又收到新帧（真实积压）时才丢帧追平。
    - 全流程只降"帧"，不降分辨率、不降模型精度；只依赖标准库。

用法：
    pacer = FramePacer(target_fps=25.0, realtime=webcam, source_fps=src_fps)
    for ... in dataset:
        if not pacer.begin_frame():
            continue            # 丢弃本帧（抽帧/追平），不处理也不输出
        ... 正常处理 ...
"""

from __future__ import annotations

import time
from typing import Dict, Optional


class FramePacer:
    """固定 FPS 节拍器：离线确定性抽帧；实时墙钟节拍 + 积压丢帧。"""

    def __init__(
        self,
        target_fps: float = 25.0,
        enabled: bool = True,
        realtime: bool = True,
        source_fps: Optional[float] = None,
        max_lag_frames: float = 2.0,
    ) -> None:
        if target_fps is None or float(target_fps) <= 0:
            raise ValueError("target_fps 必须 > 0")
        self.target_fps = float(target_fps)
        self.interval = 1.0 / self.target_fps
        self.enabled = bool(enabled)
        self.realtime = bool(realtime)
        self.source_fps = float(source_fps) if source_fps and float(source_fps) > 0 else None
        self.max_lag_frames = max(1.0, float(max_lag_frames))

        self._next_t: Optional[float] = None
        self._last_processed_t: Optional[float] = None
        self._acc = 1.0  # 离线抽帧累加器（初值 1.0 保证首帧被处理）
        self._processed = 0
        self._dropped = 0

    # ── 主循环每帧调用一次 ───────────────────────────────────────────
    def begin_frame(self, now: Optional[float] = None) -> bool:
        """返回 True = 处理本帧；False = 丢弃本帧。"""
        if not self.enabled:
            self._processed += 1
            return True
        if not self.realtime:
            return self._begin_offline()
        return self._begin_realtime(now)

    def _begin_offline(self) -> bool:
        """离线：按 源帧率/目标帧率 的比例均匀抽帧。"""
        if self.source_fps and self.source_fps > self.target_fps + 1e-6:
            self._acc += self.target_fps / self.source_fps
            if self._acc >= 1.0:
                self._acc -= 1.0
                self._processed += 1
                return True
            self._dropped += 1
            return False
        self._processed += 1
        return True

    def _begin_realtime(self, now: Optional[float]) -> bool:
        now = time.monotonic() if now is None else float(now)

        if self._next_t is None or self._last_processed_t is None:
            self._next_t = now + self.interval
            self._last_processed_t = now
            self._processed += 1
            return True

        # 距上次处理已 ≥ 一个目标周期 → 没有积压（源比目标慢/刚好），视为准时
        if (now - self._last_processed_t) >= self.interval:
            self._next_t = now + self.interval
            self._last_processed_t = now
            self._processed += 1
            return True

        lag = now - self._next_t

        # 一个周期内又来新帧且落后超过 max_lag_frames → 真实积压：丢帧追平
        if lag > self.interval * self.max_lag_frames:
            self._dropped += 1
            missed = int(lag // self.interval)
            self._next_t += self.interval * max(missed, 1)
            return False

        # 还没到下一个时隙：睡到点，把输出速率压在目标帧率上
        if lag < 0.0:
            time.sleep(-lag)
            now = time.monotonic()
        self._last_processed_t = now
        self._next_t += self.interval
        self._processed += 1
        return True

    # ── 统计 ─────────────────────────────────────────────────────────
    @property
    def processed(self) -> int:
        return self._processed

    @property
    def dropped(self) -> int:
        return self._dropped

    def stats(self) -> Dict[str, float]:
        total = self._processed + self._dropped
        return {
            "target_fps": self.target_fps,
            "mode": "realtime" if self.realtime else "offline",
            "source_fps": self.source_fps or 0.0,
            "processed": self._processed,
            "dropped": self._dropped,
            "drop_ratio": (self._dropped / total) if total else 0.0,
        }

    def status_line(self) -> str:
        s = self.stats()
        return (
            f"[frame-pacer] target={s['target_fps']:.0f}fps mode={s['mode']} "
            f"src={s['source_fps']:.0f}fps processed={int(s['processed'])} "
            f"dropped={int(s['dropped'])} drop_ratio={s['drop_ratio'] * 100:.1f}%"
        )


if __name__ == "__main__":
    # 极简自测（用假时钟，不真等待）
    print("== 离线 30fps 源 → 25fps 目标：应均匀抽掉 1/6 ==")
    p = FramePacer(target_fps=25.0, realtime=False, source_fps=30.0)
    for _ in range(60):
        p.begin_frame()
    print(p.status_line(), "（期望 processed≈50, dropped≈10）")

    print("== 离线 12.5fps 源 → 25fps 目标：源慢，应逐帧全处理 ==")
    p2 = FramePacer(target_fps=25.0, realtime=False, source_fps=12.5)
    for _ in range(40):
        p2.begin_frame()
    print(p2.status_line(), "（期望 dropped=0）")

    print("== 实时 12.5fps 到达 → 25fps 目标：不应误丢帧 ==")
    p3 = FramePacer(target_fps=25.0, realtime=True)
    fake = 0.0
    for _ in range(40):
        p3.begin_frame(now=fake)
        fake += 0.08
    print(p3.status_line(), "（期望 dropped=0）")

    print("== 实时 25fps 到达 → 25fps 目标：逐帧全处理 ==")
    p4 = FramePacer(target_fps=25.0, realtime=True)
    fake = 0.0
    for _ in range(40):
        p4.begin_frame(now=fake)
        fake += 0.04
    print(p4.status_line(), "（期望 processed=40, dropped=0）")