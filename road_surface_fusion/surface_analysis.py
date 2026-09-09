from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SurfaceHazard:
    hazard_id: str
    hazard_type: str
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    centroid_px: Tuple[int, int]
    pixel_area: float
    area_ratio: float
    bottom_ratio: float
    distance_m: float
    x_m: float
    z_m: float
    severity: float
    near_zone: bool
    mask: Optional[np.ndarray] = None


@dataclass
class SurfaceAnalysisResult:
    hazards: List[SurfaceHazard] = field(default_factory=list)
    pothole_count: int = 0
    crack_count: int = 0
    crack_severity: str = "none"
    near_hazard_count: int = 0
    total_surface_area_ratio: float = 0.0
    road_risk_score: float = 0.0
    road_danger_level: int = 0
    warning_text: str = "Road surface looks clear"
    dominant_hazard_type: Optional[str] = None
    model_label: str = "day"
    metrics: Dict[str, float] = field(default_factory=dict)


class RoadSurfaceAnalyzer:
    def __init__(self, danger_zone_ratio: float = 0.4):
        self.danger_zone_ratio = danger_zone_ratio

    def analyze(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        camera_matrix: np.ndarray,
        main_results: List,
        aux_results: List,
        model_label: str,
    ) -> SurfaceAnalysisResult:
        result = SurfaceAnalysisResult(model_label=model_label)
        if image is None or image.size == 0:
            return result

        image_h, image_w = image.shape[:2]
        image_area = float(max(image_h * image_w, 1))

        pothole_result = main_results[0] if main_results else None
        crack_result = aux_results[0] if aux_results else None

        pothole_hazards = self._extract_hazards(
            image_shape=(image_h, image_w),
            depth_map=depth_map,
            camera_matrix=camera_matrix,
            yolo_result=pothole_result,
            hazard_type="pothole",
            label="Pothole",
        )
        crack_hazards = self._extract_hazards(
            image_shape=(image_h, image_w),
            depth_map=depth_map,
            camera_matrix=camera_matrix,
            yolo_result=crack_result,
            hazard_type="crack",
            label="Crack",
        )

        result.hazards = sorted(pothole_hazards + crack_hazards, key=lambda item: item.severity, reverse=True)
        result.pothole_count = len(pothole_hazards)
        result.crack_count = len(crack_hazards)
        result.near_hazard_count = sum(1 for hazard in result.hazards if hazard.near_zone)
        result.total_surface_area_ratio = float(sum(hazard.pixel_area for hazard in result.hazards) / image_area)
        result.crack_severity = self._classify_crack_severity(crack_hazards)
        result.road_risk_score = self._compute_road_risk_score(result.hazards, result.near_hazard_count)
        result.road_danger_level, result.warning_text = self._decide_warning(result)
        result.dominant_hazard_type = result.hazards[0].hazard_type if result.hazards else None
        result.metrics = {
            "pothole_count": result.pothole_count,
            "crack_count": result.crack_count,
            "near_hazard_count": result.near_hazard_count,
            "total_surface_area_ratio": result.total_surface_area_ratio,
            "road_risk_score": result.road_risk_score,
        }
        return result

    def _extract_hazards(
        self,
        image_shape: Tuple[int, int],
        depth_map: np.ndarray,
        camera_matrix: np.ndarray,
        yolo_result,
        hazard_type: str,
        label: str,
    ) -> List[SurfaceHazard]:
        if yolo_result is None or not hasattr(yolo_result, "boxes") or len(yolo_result.boxes) == 0:
            return []

        image_h, image_w = image_shape
        image_area = float(max(image_h * image_w, 1))

        masks = None
        if hasattr(yolo_result, "masks") and yolo_result.masks is not None:
            masks = yolo_result.masks.data.detach().cpu().numpy()

        hazards: List[SurfaceHazard] = []
        for index, box in enumerate(yolo_result.boxes):
            bbox = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            confidence = float(box.conf[0].detach().cpu().item()) if hasattr(box, "conf") else 0.0

            mask = None
            if masks is not None and index < len(masks):
                mask = self._resize_mask(masks[index], image_shape)

            bbox_tuple, centroid, pixel_area, bottom_ratio = self._geometry_from_mask_or_box(
                bbox,
                mask,
                image_shape,
            )
            if pixel_area <= 0:
                continue

            area_ratio = pixel_area / image_area
            distance_m = self._estimate_distance(depth_map, bbox_tuple, mask)
            x_m = self._estimate_lateral_position(centroid[0], distance_m, camera_matrix)
            z_m = distance_m
            near_zone = bottom_ratio >= (1.0 - self.danger_zone_ratio)
            severity = self._compute_severity(
                hazard_type=hazard_type,
                confidence=confidence,
                area_ratio=area_ratio,
                bottom_ratio=bottom_ratio,
                distance_m=distance_m,
            )

            hazards.append(
                SurfaceHazard(
                    hazard_id=f"{hazard_type}_{index}",
                    hazard_type=hazard_type,
                    label=label,
                    confidence=confidence,
                    bbox=bbox_tuple,
                    centroid_px=centroid,
                    pixel_area=pixel_area,
                    area_ratio=area_ratio,
                    bottom_ratio=bottom_ratio,
                    distance_m=distance_m,
                    x_m=x_m,
                    z_m=z_m,
                    severity=severity,
                    near_zone=near_zone,
                    mask=mask,
                )
            )

        return hazards

    def _resize_mask(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        image_h, image_w = image_shape
        if mask.shape != (image_h, image_w):
            mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        return (mask > 0.5).astype(np.uint8)

    def _geometry_from_mask_or_box(
        self,
        bbox: List[int],
        mask: Optional[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int], float, float]:
        image_h, image_w = image_shape
        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(x1, 0, image_w - 1))
        y1 = int(np.clip(y1, 0, image_h - 1))
        x2 = int(np.clip(x2, x1 + 1, image_w))
        y2 = int(np.clip(y2, y1 + 1, image_h))

        if mask is not None and np.any(mask):
            ys, xs = np.where(mask > 0)
            bbox_tuple = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            centroid = (int(np.mean(xs)), int(np.mean(ys)))
            pixel_area = float(len(xs))
            bottom_ratio = float((ys.max() + 1) / max(image_h, 1))
            return bbox_tuple, centroid, pixel_area, bottom_ratio

        bbox_tuple = (x1, y1, x2, y2)
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        pixel_area = float(max((x2 - x1) * (y2 - y1), 0))
        bottom_ratio = float(y2 / max(image_h, 1))
        return bbox_tuple, centroid, pixel_area, bottom_ratio

    def _estimate_distance(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray],
    ) -> float:
        if depth_map is None or depth_map.size == 0:
            return 10.0

        if mask is not None and np.any(mask):
            mask_values = depth_map[mask > 0]
            if mask_values.size > 0:
                depth_value = float(np.median(mask_values))
            else:
                depth_value = self._depth_from_bbox(depth_map, bbox)
        else:
            depth_value = self._depth_from_bbox(depth_map, bbox)

        depth_value = float(np.clip(depth_value, 0.0, 1.0))
        # 方向修正：相对深度数值越大 = 越近（与 bbox3d_utils 保持一致，范围 1-10m 不变）
        return 1.0 + (1.0 - depth_value) * 9.0

    def _depth_from_bbox(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        crop = depth_map[y1:y2, x1:x2]
        if crop.size == 0:
            return 1.0
        return float(np.median(crop))

    def _estimate_lateral_position(self, pixel_x: int, distance_m: float, camera_matrix: np.ndarray) -> float:
        fx = float(camera_matrix[0, 0]) if camera_matrix is not None else 1.0
        cx = float(camera_matrix[0, 2]) if camera_matrix is not None else 0.0
        if fx == 0:
            return 0.0
        return ((pixel_x - cx) / fx) * distance_m

    def _compute_severity(
        self,
        hazard_type: str,
        confidence: float,
        area_ratio: float,
        bottom_ratio: float,
        distance_m: float,
    ) -> float:
        near_score = float(np.clip((bottom_ratio - 0.45) / 0.55, 0.0, 1.0))
        size_multiplier = 16.0 if hazard_type == "pothole" else 10.0
        size_score = float(np.clip(area_ratio * size_multiplier, 0.0, 1.0))
        distance_score = float(np.clip((12.0 - distance_m) / 12.0, 0.0, 1.0))
        type_bias = 0.15 if hazard_type == "pothole" else 0.05

        severity = (
            type_bias
            + confidence * 0.15
            + size_score * 0.4
            + near_score * 0.3
            + distance_score * 0.2
        )
        return float(np.clip(severity, 0.0, 1.0))

    def _classify_crack_severity(self, crack_hazards: List[SurfaceHazard]) -> str:
        if not crack_hazards:
            return "none"

        max_severity = max(hazard.severity for hazard in crack_hazards)
        crack_count = len(crack_hazards)
        if max_severity >= 0.75 or crack_count >= 5:
            return "high"
        if max_severity >= 0.45 or crack_count >= 3:
            return "medium"
        return "low"

    def _compute_road_risk_score(self, hazards: List[SurfaceHazard], near_hazard_count: int) -> float:
        if not hazards:
            return 0.0

        peak = max(hazard.severity for hazard in hazards)
        density_boost = min(0.3, 0.07 * len(hazards))
        near_boost = min(0.25, 0.08 * near_hazard_count)
        return float(np.clip(peak + density_boost + near_boost, 0.0, 1.0))

    def _decide_warning(self, result: SurfaceAnalysisResult) -> Tuple[int, str]:
        if result.road_risk_score >= 0.8:
            return 3, "Road surface risk is high"
        if result.road_risk_score >= 0.55:
            return 2, "Multiple potholes or cracks ahead"
        if result.road_risk_score >= 0.25:
            return 1, "Minor road damage detected"
        return 0, "Road surface looks clear"
