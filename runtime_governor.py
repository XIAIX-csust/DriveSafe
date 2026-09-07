"""
runtime_governor.py — DriveSafe 运行时负载调度器（Runtime Load Governor）

核心思路（仅运行时动态调控，不做启动时硬件探测）：
  用指数滑动平均统计每一帧的推理耗时，换算当前 FPS 与目标 FPS 的压力比，
  然后按压力自动升降频：
    - 深度估计 Depth Anything   （最贵，优先降频 + 输入降采样）
    - 路面平整度双模型          （次贵，降低调用频率）
    - DeepSort REID 特征提取    （按帧复用已算特征，只在必要时重算）
    - BEV 热力图渲染（matplotlib）（抽帧渲染，跳过帧复用上一帧画面）

用法：
    governor = RuntimeGovernor(target_fps=15)
    ...
    loop_t0 = time.time()
    ... 推理 ...
    governor.tick(time.time() - loop_t0)

    if governor.should_run_depth(): depth_map = est.estimate_depth(frame)
    else: depth_map = est.last_depth_map
    if governor.should_run_road(): ...  # 路面模型
    if governor.should_render_bev(): ... # matplotlib BEV

每个模块三种策略（auto / always / off），可用命令行参数或环境变量覆盖：
    DRIVESAFE_TARGET_FPS / DRIVESAFE_DEPTH_STRATEGY / DRIVESAFE_ROAD_STRATEGY
    DRIVESAFE_REID_STRATEGY / DRIVESAFE_BEV_STRATEGY

该模块只依赖 Python 标准库，可在任何环境（含 Jetson）直接使用。
"""

from __future__ import annotations

import os
import time
from typing import Optional

_SKIP_DISABLED = {"0", "false", "no", "off"}
_MAX_INTERVAL = {"depth": 8, "road": 6, "reid": 3, "bev": 4}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class RuntimeGovernor:
    """按当前实测 FPS 动态决定各昂贵模块是否执行本帧。"""

    def __init__(
        self,
        target_fps: Optional[float] = None,
        depth_strategy: Optional[str] = None,
        road_strategy: Optional[str] = None,
        reid_strategy: Optional[str] = None,
        bev_strategy: Optional[str] = None,
        window: int = 15,
        warmup: int = 8,
    ) -> None:
        self.target_fps = target_fps if target_fps and target_fps > 0 else _env_float(
            "DRIVESAFE_TARGET_FPS", 15.0
        )
        self.depth_strategy = (depth_strategy or os.environ.get("DRIVESAFE_DEPTH_STRATEGY", "auto")).lower()
        self.road_strategy = (road_strategy or os.environ.get("DRIVESAFE_ROAD_STRATEGY", "auto")).lower()
        self.reid_strategy = (reid_strategy or os.environ.get("DRIVESAFE_REID_STRATEGY", "auto")).lower()
        self.bev_strategy = (bev_strategy or os.environ.get("DRIVESAFE_BEV_STRATEGY", "auto")).lower()

        self.window = max(2, int(window))
        self.warmup = max(1, int(warmup))
        self._frame_idx = 0
        self._dts: list[float] = []
        self._ewma_dt: Optional[float] = None
        self._last_status_printed = -1

    # ── 采样 ──────────────────────────────────────────────────────────
    def tick(self, seconds: float) -> None:
        """每帧结束后调用一次，记录本帧推理耗时（不含显示/保存）。"""
        if seconds < 0:
            seconds = 0.0
        self._dts.append(seconds)
        if len(self._dts) > self.window:
            self._dts.pop(0)

        if self._ewma_dt is None:
            self._ewma_dt = seconds
        else:
            alpha = 0.3  # 新样本权重，兼顾平滑与响应速度
            self._ewma_dt = alpha * seconds + (1.0 - alpha) * self._ewma_dt

        self._frame_idx += 1
        if self._frame_idx % 120 == 0:
            self._print_status()

    @property
    def current_fps(self) -> float:
        if not self._dts:
            return float("inf")
        avg = sum(self._dts) / len(self._dts)
        return 1.0 / avg if avg > 0 else float("inf")

    def load_pressure(self) -> float:
        """压力比 = 目标FPS / 实测FPS。1.0 = 达标；越大越吃力；<1 = 有余量。"""
        fps = self.current_fps
        if fps == float("inf"):
            return 1.0
        return self.target_fps / max(fps, 1e-6)

    # ── 模块级决策 ────────────────────────────────────────────────────
    def _interval_for(self, kind: str, base_interval: int = 1) -> int:
        if self._frame_idx < self.warmup:
            return 1  # 预热期全速跑，让采样有意义
        if self._frame_idx < self.window * 2:
            return 1  # 采样窗口还没填满，先用满速
        pressure = self.load_pressure()
        interval = base_interval
        if pressure >= 1.0:
            # 每多一倍压力，间隔乘 2（最多封顶）
            steps = int(pressure)
            interval = base_interval * (2 ** min(steps, 4))
        return max(1, min(interval, _MAX_INTERVAL[kind]))

    def _strategy_enabled(self, strategy: str) -> bool:
        if strategy == "off":
            return False
        if strategy == "always":
            return True
        return True  # auto —— 由 _interval_for 决定是否执行

    def should_run_depth(self) -> bool:
        if not self._strategy_enabled(self.depth_strategy):
            return False
        if self.depth_strategy == "always":
            return True
        return self._frame_idx % self._interval_for("depth") == 0

    def depth_input_scale(self) -> float:
        """深度输入降采样比例（0.5~1.0）。压力越大，采样越小。"""
        if self._frame_idx < self.window * 2:
            return 1.0
        pressure = self.load_pressure()
        if pressure >= 3.0:
            return 0.5
        if pressure >= 2.0:
            return 0.75
        return 1.0

    def should_run_road(self) -> bool:
        if not self._strategy_enabled(self.road_strategy):
            return False
        if self.road_strategy == "always":
            return True
        # 路面模型自带内部缓存节流，把基础间隔定为 2
        return self._frame_idx % self._interval_for("road", base_interval=2) == 0

    def should_run_reid(self) -> bool:
        if not self._strategy_enabled(self.reid_strategy):
            return False
        if self.reid_strategy == "always":
            return True
        return self._frame_idx % self._interval_for("reid") == 0

    def should_render_bev(self) -> bool:
        if not self._strategy_enabled(self.bev_strategy):
            return False
        if self.bev_strategy == "always":
            return True
        return self._frame_idx % self._interval_for("bev") == 0

    # ── 诊断 ──────────────────────────────────────────────────────────
    def _print_status(self) -> None:
        print(
            "[runtime-governor] "
            f"fps={self.current_fps:.1f} (target={self.target_fps:.0f}) "
            f"pressure={self.load_pressure():.2f} "
            f"depth_interval={self._interval_for('depth')} "
            f"road_interval={self._interval_for('road', 2)} "
            f"reid_interval={self._interval_for('reid')} "
            f"bev_interval={self._interval_for('bev')}"
        )


if __name__ == "__main__":
    # 极简自测：模拟固定 0.1s/帧（≈10fps，目标15fps）
    g = RuntimeGovernor(target_fps=15.0, window=10, warmup=5)
    for i in range(60):
        time.sleep(0)  # 不真跑，直接喂模拟耗时
        g.tick(0.1)
        if i % 5 == 0:
            print(
                f"frame={i:3d} fps={g.current_fps:5.1f} pressure={g.load_pressure():4.2f} "
                f"depth={g.should_run_depth()} road={g.should_run_road()} "
                f"reid={g.should_run_reid()} bev={g.should_render_bev()} "
                f"depth_scale={g.depth_input_scale()}"
            )