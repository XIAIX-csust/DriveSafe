from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from .surface_analysis import SurfaceAnalysisResult


class RoadSurfaceVisualizer:
    def __init__(self):
        self.colors: Dict[str, Tuple[int, int, int]] = {
            "pothole": (0, 165, 255),
            "crack": (255, 0, 255),
        }

    def draw_on_frame(self, image: np.ndarray, analysis: SurfaceAnalysisResult) -> np.ndarray:
        if image is None:
            return image

        annotated = image.copy()
        overlay = annotated.copy()

        for hazard in analysis.hazards:
            color = self.colors.get(hazard.hazard_type, (255, 255, 255))
            if hazard.mask is not None and np.any(hazard.mask):
                overlay[hazard.mask > 0] = color

        if analysis.hazards:
            annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

        for hazard in analysis.hazards:
            color = self.colors.get(hazard.hazard_type, (255, 255, 255))
            x1, y1, x2, y2 = hazard.bbox
            thickness = 2 if hazard.hazard_type == "pothole" else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label = f"{hazard.label} {hazard.confidence:.2f} {hazard.distance_m:.1f}m"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_top = max(y1 - text_h - 6, 0)
            cv2.rectangle(
                annotated,
                (x1, text_top),
                (x1 + text_w + 6, text_top + text_h + 6),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, text_top + text_h + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        self._draw_summary_panel(annotated, analysis)
        return annotated

    def draw_on_bev(self, bev_visualizer, analysis: SurfaceAnalysisResult) -> None:
        canvas = bev_visualizer.bev_image
        for hazard in analysis.hazards:
            px = bev_visualizer.origin_x + int(hazard.x_m * bev_visualizer.scale)
            py = bev_visualizer.origin_y - int(hazard.z_m * bev_visualizer.scale)

            if px < 0 or py < 0 or px >= bev_visualizer.width or py >= bev_visualizer.height:
                continue

            color = self.colors.get(hazard.hazard_type, (255, 255, 255))
            radius = max(4, int(4 + hazard.severity * 8))
            cv2.circle(canvas, (px, py), radius, color, -1)
            cv2.circle(canvas, (px, py), radius + 2, (255, 255, 255), 1)
            # 短标签取实际类别首字母（Pothole→P、Crack→C、Manhole→M、Net→N…）
            short_label = hazard.label.strip()[:1].upper() or "?"
            cv2.putText(
                canvas,
                short_label,
                (px - 4, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        status = self._status_name(analysis.road_danger_level)
        cv2.putText(
            canvas,
            f"ROAD: {status}",
            (10, bev_visualizer.height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"P:{analysis.pothole_count} C:{analysis.crack_count} SCORE:{analysis.road_risk_score:.2f}",
            (10, bev_visualizer.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_summary_panel(self, image: np.ndarray, analysis: SurfaceAnalysisResult) -> None:
        panel_lines = [
            f"Road model: {analysis.model_label}",
            f"Pothole: {analysis.pothole_count}  Crack: {analysis.crack_count}  Score: {analysis.road_risk_score:.2f}",
            f"Surface: {self._status_name(analysis.road_danger_level)}",
        ]

        panel_x = 15
        panel_y = 20
        panel_w = 460
        panel_h = 82

        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

        for index, line in enumerate(panel_lines):
            cv2.putText(
                image,
                line,
                (panel_x + 10, panel_y + 22 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _status_name(self, danger_level: int) -> str:
        if danger_level >= 3:
            return "HIGH"
        if danger_level == 2:
            return "MEDIUM"
        if danger_level == 1:
            return "LOW"
        return "CLEAR"

    def draw_trajectory_envelope(self, image: np.ndarray, trajectory: list, risk_score: float = 0.0) -> np.ndarray:
        """
        Draw trajectory envelope (prediction envelope) on the image.
        :param image: Input image
        :param trajectory: List of predicted positions [(x1, y1), (x2, y2), ...]
        :param risk_score: Risk score (0-1) for coloring the envelope
        :return: Image with trajectory envelope drawn
        """
        if not trajectory or len(trajectory) < 2:
            return image
        
        annotated = image.copy()
        
        # Calculate envelope width based on distance
        envelope_points = []
        for i, (x, y) in enumerate(trajectory):
            # Base width on distance (farther points have wider envelope)
            width = 5 + i * 0.5
            # Add left and right points for envelope
            if i > 0:
                # Calculate direction vector
                dx = x - trajectory[i-1][0]
                dy = y - trajectory[i-1][1]
                # Normalize direction vector
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx /= length
                    dy /= length
                    # Perpendicular vector for envelope width
                    perp_x = -dy
                    perp_y = dx
                    # Add left and right points
                    envelope_points.append((int(x - perp_x * width), int(y - perp_y * width)))
                    envelope_points.append((int(x + perp_x * width), int(y + perp_y * width)))
        
        # Close the envelope
        if len(envelope_points) >= 4:
            # Create polygon points
            polygon = np.array(envelope_points, dtype=np.int32)
            
            # Determine color based on risk score
            if risk_score > 0.7:
                color = (0, 0, 255)  # Red for high risk
            elif risk_score > 0.3:
                color = (0, 165, 255)  # Orange for medium risk
            else:
                color = (0, 255, 0)  # Green for low risk
            
            # Draw filled polygon with transparency
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], color)
            annotated = cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0)
            
            # Draw trajectory line
            trajectory_points = np.array(trajectory, dtype=np.int32)
            cv2.polylines(annotated, [trajectory_points], False, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw trajectory points
            for i, (x, y) in enumerate(trajectory):
                # Point size decreases with distance
                radius = max(2, 4 - i // 2)
                cv2.circle(annotated, (int(x), int(y)), radius, (255, 255, 255), -1)
        
        return annotated

    def draw_trajectories(self, image: np.ndarray, trajectories: dict, risk_scores: dict = None) -> np.ndarray:
        """
        Draw trajectories for multiple targets.
        :param image: Input image
        :param trajectories: Dict of trajectories {track_id: [(x1, y1), ...]}
        :param risk_scores: Dict of risk scores {track_id: score}
        :return: Image with trajectories drawn
        """
        annotated = image.copy()
        
        for track_id, trajectory in trajectories.items():
            risk_score = risk_scores.get(track_id, 0.0) if risk_scores else 0.0
            annotated = self.draw_trajectory_envelope(annotated, trajectory, risk_score)
        
        return annotated
