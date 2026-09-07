from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional


_DISABLED_VALUES = {"0", "false", "no", "off"}
_ALERT_PRIORITY = {
    "vehicle": 1,
    "pothole": 1,
    "danger": 2,
}
_ALERT_SOUND_FILES = {
    "vehicle": "vehicle_warning.wav",
    "pothole": "pothole_warning.wav",
    "danger": "danger_warning.wav",
}


class RiskSoundAlerter:
    """Play voice warnings for vehicle, pothole, and high-danger decisions."""

    def __init__(
        self,
        sounds_dir: Optional[str | Path] = None,
        cooldown_seconds: Optional[float] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env_sounds_dir = os.environ.get("DRIVESAFE_SOUND_DIR")
        self.sounds_dir = Path(sounds_dir or env_sounds_dir or repo_root / "sounds")
        self.cooldown_seconds = (
            float(cooldown_seconds)
            if cooldown_seconds is not None
            else float(os.environ.get("DRIVESAFE_SOUND_ALERT_COOLDOWN", "5.0"))
        )
        self.enabled = os.environ.get("DRIVESAFE_SOUND_ALERT", "1").strip().lower() not in _DISABLED_VALUES
        self._last_play_time = 0.0
        self._last_alert: Optional[str] = None
        self._warned_keys: set[str] = set()
        self._winsound = None
        self._player_cmd: Optional[tuple] = None  # (backend名, 可执行文件路径)
        self._pygame_mixer = None

        if self.enabled and os.name == "nt":
            try:
                import winsound

                self._winsound = winsound
            except Exception as exc:
                self._warn_once("winsound", f"[risk-sound-alert] winsound unavailable: {exc}")
        elif self.enabled:
            self._init_linux_player()

    def handle_frame_record(self, frame_record: dict) -> None:
        if not self.enabled:
            return

        alert = self._select_alert(frame_record)
        if alert is None:
            return

        self._handle_alert(alert)

    def handle_status(self, status: object) -> None:
        if not self.enabled:
            return

        level = str(status or "").upper()
        if level == "HIGH":
            alert = "danger"
        elif level == "MEDIUM":
            alert = "vehicle"
        else:
            return

        self._handle_alert(alert)

    def _handle_alert(self, alert: str) -> None:
        now = time.monotonic()
        if not self._should_play(alert, now):
            return

        if self._play(alert):
            self._last_play_time = now
            self._last_alert = alert

    def _select_alert(self, frame_record: dict) -> Optional[str]:
        level = str(frame_record.get("decision_status") or "").upper()
        if level == "HIGH":
            return "danger"
        if level != "MEDIUM":
            return None

        dynamic_risk = self._to_float(frame_record.get("dynamic_risk"))
        surface_risk = self._to_float(frame_record.get("surface_risk"))
        surface_hazards = frame_record.get("surface_hazards") or []
        targets = frame_record.get("targets") or []
        max_risk_source = str(frame_record.get("max_risk_source") or "").upper()
        warning_text = str(frame_record.get("warning_text") or "").lower()

        surface_is_dominant = surface_risk >= dynamic_risk or max_risk_source == "ROAD"
        text_mentions_surface = any(
            token in warning_text
            for token in ("road", "surface", "pothole", "crack", "坑", "洼", "路面")
        )

        if surface_hazards and (surface_is_dominant or text_mentions_surface):
            return "pothole"
        if targets or dynamic_risk > 0:
            return "vehicle"
        if surface_hazards:
            return "pothole"
        if text_mentions_surface:
            return "pothole"
        return "vehicle"

    def _should_play(self, alert: str, now: float) -> bool:
        elapsed = now - self._last_play_time
        if elapsed >= self.cooldown_seconds:
            return True

        last_priority = _ALERT_PRIORITY.get(self._last_alert or "", 0)
        current_priority = _ALERT_PRIORITY[alert]
        return current_priority > last_priority

    def _init_linux_player(self) -> None:
        """Linux/Jetson：按优先级探测可用的命令行音频播放器。"""
        for kind in ("paplay", "aplay", "pw-play", "ffplay"):
            exe = shutil.which(kind)
            if exe:
                self._player_cmd = (kind, exe)
                return

        # 兜底：pygame.mixer（pip install pygame）
        try:
            import pygame

            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._pygame_mixer = pygame.mixer
        except Exception as exc:
            self._warn_once(
                "linux-player",
                "[risk-sound-alert] no audio backend on Linux: install alsa-utils "
                "(aplay) or pipewire/pulseaudio (paplay), or pip install pygame. "
                f"details: {exc}",
            )

    def _play(self, alert: str) -> bool:
        sound_file = _ALERT_SOUND_FILES[alert]
        sound_path = self.sounds_dir / sound_file
        if not sound_path.is_file():
            self._warn_once(str(sound_path), f"[risk-sound-alert] missing sound file: {sound_path}")
            return False

        # Windows：winsound（异步，不阻塞主循环）
        if self._winsound is not None:
            try:
                self._winsound.PlaySound(
                    str(sound_path),
                    self._winsound.SND_FILENAME | self._winsound.SND_ASYNC,
                )
                return True
            except Exception as exc:
                self._warn_once(f"play:{alert}", f"[risk-sound-alert] failed to play {sound_path}: {exc}")
                return False

        # Linux/Jetson：命令行播放器（非阻塞子进程，不阻塞推理主循环）
        if self._player_cmd is not None:
            kind, exe = self._player_cmd
            args = [exe]
            if kind == "aplay":
                args += ["-q", str(sound_path)]
            elif kind == "paplay":
                args += [str(sound_path)]
            elif kind == "pw-play":
                args += [str(sound_path)]
            elif kind == "ffplay":
                args += ["-nodisp", "-autoexit", "-loglevel", "quiet", str(sound_path)]
            try:
                subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as exc:
                self._warn_once(f"play:{alert}", f"[risk-sound-alert] failed to spawn {kind}: {exc}")
                return False

        # Linux 兜底：pygame.mixer（非阻塞）
        if self._pygame_mixer is not None:
            try:
                sound_obj = self._pygame_mixer.Sound(str(sound_path))
                sound_obj.play()
                return True
            except Exception as exc:
                self._warn_once(f"play:{alert}", f"[risk-sound-alert] pygame play failed: {exc}")
                return False

        self._warn_once("no-backend", "[risk-sound-alert] no working audio backend")
        return False

    def _to_float(self, value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        print(message)
