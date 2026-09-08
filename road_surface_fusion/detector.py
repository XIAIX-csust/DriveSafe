from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 修复 PyTorch 2.6+ 的 weights_only 问题
try:
    # 猴子补丁修复 torch.load 函数
    import torch
    original_load = torch.load
    
    def patched_load(f, map_location=None, pickle_module=None, **kwargs):
        """Patched version that forces weights_only=False"""
        # 强制设置 weights_only=False
        kwargs['weights_only'] = False
        return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
    
    # 应用猴子补丁
    torch.load = patched_load
    print("✅ Patched torch.load to use weights_only=False")
except Exception as e:
    print(f"Error patching torch.load: {e}")
    pass


class RoadSurfaceDetector:
    """
    Re-implements the road-surface model selection logic from the `code` project
    inside Distance-measurement so the original code can remain untouched.
    """

    def __init__(self, model_dir: Optional[str] = None, preferred_device: Optional[str] = None):
        integration_root = Path(__file__).resolve().parents[1]

        self.model_dir = Path(model_dir) if model_dir else integration_root / "code" / "models"
        self.day_model_path = self._pick_model("best")
        self.night_model_path = self._pick_model("best_night")
        self.crack_model_path = self._pick_model("crack_best")

        missing = [
            str(path)
            for path in [self.day_model_path, self.night_model_path, self.crack_model_path]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Road-surface model weights were not found. Missing files:\n" + "\n".join(missing)
            )

        self.preferred_device = preferred_device

        self.day_model = YOLO(str(self.day_model_path))
        self.night_model = YOLO(str(self.night_model_path))
        self.auxiliary_model = YOLO(str(self.crack_model_path))

        self.last_results: List = []
        self.last_aux_results: List = []
        self.last_model_label = "day"
        self.frame_skip = 2
        self.current_frame_count = 0

        self._optimize_models()

    def _pick_model(self, name: str) -> Path:
        """Prefer a TensorRT engine if it was exported, otherwise the .pt weights."""
        engine = self.model_dir / f"{name}.engine"
        if engine.exists():
            return engine
        return self.model_dir / f"{name}.pt"

    def _optimize_models(self) -> None:
        if self.preferred_device == "cpu":
            self.device = torch.device("cpu")
        elif self.preferred_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inference_device = "cuda:0" if self.device.type == "cuda" else self.device.type

        for model in [self.day_model, self.night_model, self.auxiliary_model]:
            model.conf = 0.25
            model.iou = 0.45
            model.agnostic = False
            model.multi_label = False
            model.max_det = 100

    def is_night(self, image: np.ndarray) -> bool:
        if image is None or image.size == 0:
            return False

        height, width = image.shape[:2]
        scale = min(1.0, 640 / max(height, width))
        if scale < 1.0:
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_brightness = float(np.mean(gray))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dark_threshold = 50
        dark_ratio = float(np.sum(hist[:dark_threshold]) / np.sum(hist))
        return average_brightness < 80 or dark_ratio > 0.5

    def detect(
        self,
        image: np.ndarray,
        conf_thres: Optional[float] = None,
        skip_frame_check: bool = False,
    ) -> Tuple[List, List, str]:
        if image is None or image.size == 0:
            return [], [], self.last_model_label

        if not skip_frame_check and self.last_results:
            self.current_frame_count += 1
            if self.current_frame_count % self.frame_skip != 0:
                return self.last_results, self.last_aux_results, self.last_model_label

        self.current_frame_count = 0

        if conf_thres is not None:
            for model in [self.day_model, self.night_model, self.auxiliary_model]:
                model.conf = conf_thres

        night_mode = self.is_night(image)
        self.last_model_label = "night" if night_mode else "day"
        main_model = self.night_model if night_mode else self.day_model

        inference_args = {
            "verbose": False,
            "device": self.inference_device,
        }

        try:
            main_results = main_model(image, **inference_args)
            aux_results = self.auxiliary_model(image, **inference_args)
        except TypeError:
            inference_args.pop("device", None)
            main_results = main_model(image, **inference_args)
            aux_results = self.auxiliary_model(image, **inference_args)

        if main_results is None:
            main_results = []
        if aux_results is None:
            aux_results = []

        self.last_results = main_results
        self.last_aux_results = aux_results
        return main_results, aux_results, self.last_model_label
