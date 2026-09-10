import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# 风险等级阈值统一来源（SCF 尺度），不要在此另写数字
from risk_field import RISK_THRESHOLDS, risk_level


@dataclass
class DetectionObject:
    track_id: int
    class_name: str
    x: float
    z: float
    speed: float
    distance: float
    risk_score: float
    risk_level: str
    risk_type: str
    bbox: List[float]
    timestamp: float = 0.0


@dataclass
class FrameData:
    frame_idx: int
    objects: List[DetectionObject]
    max_risk_score: float
    avg_risk_score: float
    total_tracked: int
    alert_triggered: bool
    timestamp: float


class DetectionDataStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._current_objects: List[DetectionObject] = []
        self._frame_history: deque = deque(maxlen=300)
        self._risk_trend: List[Dict[str, Any]] = []
        self._current_frame_idx: int = 0
        self._total_frames: int = 0
        self._total_alerts: int = 0
        self._lock = threading.Lock()

    def update_frame(
        self,
        frame_idx: int,
        risk_sources: List[Dict[str, Any]],
        alert_triggered: bool = False,
        frame_risk_score: Optional[float] = None
    ):
        with self._lock:
            self._current_frame_idx = frame_idx
            self._total_frames = frame_idx + 1
            if alert_triggered:
                self._total_alerts += 1

            detection_objects = []
            max_risk = 0.0
            total_risk = 0.0

            for src in risk_sources:
                risk_score = src.get('scf', 0.0)

                # 优先使用 class_name，如果没有则使用 type
                class_name = src.get('class_name', src.get('type', 'vehicle'))

                risk_level = self._calculate_risk_level(risk_score)
                risk_type = self._determine_risk_type(src)

                box_3d_draw = src.get('box_3d_draw', {})
                depth_value = box_3d_draw.get('depth_value', 0.0)

                det_obj = DetectionObject(
                    track_id=src.get('id', 0),
                    class_name=class_name,
                    x=src.get('x', 0.0),
                    z=src.get('z', 0.0),
                    speed=src.get('speed', 0.0),
                    distance=src.get('distance', src.get('z', 0.0)),
                    risk_score=risk_score,
                    risk_level=risk_level,
                    risk_type=risk_type,
                    bbox=src.get('xyxy', []),
                    timestamp=time.time()
                )
                detection_objects.append(det_obj)

                if risk_score > max_risk:
                    max_risk = risk_score
                total_risk += risk_score

            avg_risk = total_risk / len(detection_objects) if detection_objects else 0.0

            frame_data = FrameData(
                frame_idx=frame_idx,
                objects=detection_objects,
                max_risk_score=max_risk,
                avg_risk_score=avg_risk,
                total_tracked=len(detection_objects),
                alert_triggered=alert_triggered,
                timestamp=time.time()
            )

            self._current_objects = detection_objects
            self._frame_history.append(frame_data)
            self._update_risk_trend(frame_idx, frame_risk_score)

    def _update_risk_trend(self, frame_idx: int, risk_score: Optional[float]):
        if risk_score is None:
            return

        try:
            trend_frame_idx = int(frame_idx)
            trend_risk_score = float(risk_score)
        except (TypeError, ValueError):
            return

        if trend_frame_idx < 0:
            return

        point = {
            'frame_idx': trend_frame_idx,
            'risk_score': trend_risk_score,
            'timestamp': time.time()
        }

        for existing in reversed(self._risk_trend):
            if existing['frame_idx'] == trend_frame_idx:
                existing.update(point)
                return

        self._risk_trend.append(point)

    def _calculate_risk_level(self, risk_score: float) -> str:
        # 与 detect_3d_with_surface.py 的 decision_status 同源同阈值（risk_field.RISK_THRESHOLDS）
        level = risk_level(risk_score)
        return {"HIGH": "高风险", "MEDIUM": "中风险", "LOW": "低风险"}.get(level, "安全")

    def _determine_risk_type(self, obj: Dict[str, Any]) -> str:
        speed = obj.get('speed', 0)
        distance = obj.get('z', 999)
        risk_score = obj.get('scf', 0)

        # 根据风险值(scf)判断（阈值同 risk_field.RISK_THRESHOLDS）
        level = risk_level(risk_score)
        if level == "HIGH":
            return "高风险区域"
        elif level == "MEDIUM":
            return "中风险区域"
        elif level == "LOW":
            return "低风险区域"
        # 根据距离判断
        elif distance < 3:
            return "距离过近"
        elif distance < 5:
            return "距离较近"
        # 根据速度判断
        elif speed > 30:
            return "超速行驶"
        elif speed > 20 and distance < 15:
            return "高速接近"
        else:
            return "正常"

    def get_current_objects(self) -> List[DetectionObject]:
        with self._lock:
            return self._current_objects.copy()

    def get_current_frame(self) -> Optional[FrameData]:
        with self._lock:
            if self._frame_history:
                return self._frame_history[-1]
            return None

    def get_risk_trend(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [point.copy() for point in self._risk_trend]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            current_objs = self._current_objects
            max_risk = 0.0
            high_risk_count = 0

            for obj in current_objs:
                if obj.risk_score > max_risk:
                    max_risk = obj.risk_score
                if obj.risk_score >= RISK_THRESHOLDS["HIGH"]:
                    high_risk_count += 1

            return {
                'track_count': len(current_objs),
                'total_frames': self._total_frames,
                'total_alerts': self._total_alerts,
                'max_risk_score': max_risk,
                'high_risk_count': high_risk_count
            }

    def reset(self):
        with self._lock:
            self._current_objects = []
            self._frame_history.clear()
            self._risk_trend.clear()
            self._current_frame_idx = 0
            self._total_frames = 0
            self._total_alerts = 0


data_store = DetectionDataStore()
