from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Masks

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

    def __init__(
        self,
        model_dir: Optional[str] = None,
        preferred_device: Optional[str] = None,
        roi_top_ratio: float = 0.5,
        parallel: bool = True,
    ):
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
        # 本帧是否真的跑了推理（False = 走的缓存复用）。
        # 供上层把 analyze() 等后处理按同样节奏降频，避免每帧白跑一遍掩码/几何。
        self.last_run_fresh = False
        self.frame_skip = 2
        self.current_frame_count = 0
        # 裂缝模型变化慢，按更长间隔运行（每 aux_frame_skip 次主模型运行跑一次）
        self.aux_frame_skip = 2
        self.aux_frame_count = 0

        # ROI：只把画面下半部喂给路面模型（0 = 关闭，用整幅）。
        # 路面隐患只出现在下半部，裁掉上半部能显著减少 letterbox 预处理面积，
        # 而 ROI 内的有效分辨率不变 —— 比整幅降 imgsz 更安全。
        self.roi_top_ratio = float(roi_top_ratio or 0.0)
        # 主模型与 aux 模型互不依赖，串行时是两个推理时间相加，故默认并行
        self.parallel = bool(parallel)
        self._executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="roadsurf")
            if self.parallel
            else None
        )

        self._optimize_models()

    def close(self) -> None:
        """释放并行的线程池（进程退出或显式销毁时调用）。"""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
            self.last_run_fresh = False
            return [], [], self.last_model_label

        if not skip_frame_check and self.last_results:
            self.current_frame_count += 1
            if self.current_frame_count % self.frame_skip != 0:
                # 未到推理周期：复用上一帧结果
                self.last_run_fresh = False
                return self.last_results, self.last_aux_results, self.last_model_label

        self.current_frame_count = 0
        self.last_run_fresh = True

        if conf_thres is not None:
            for model in [self.day_model, self.night_model, self.auxiliary_model]:
                model.conf = conf_thres

        height, width = image.shape[:2]
        # ROI 裁剪：路面隐患只出现在画面下半部（roi_top_ratio=0 表示不裁）
        if 0.0 < self.roi_top_ratio < 1.0:
            y_off = max(0, min(int(height * self.roi_top_ratio), height - 1))
        else:
            y_off = 0
        roi = image[y_off:, :] if y_off > 0 else image

        # 昼夜判断仍用整幅（下半幅亮度不具代表性）
        night_mode = self.is_night(image)
        self.last_model_label = "night" if night_mode else "day"
        main_model = self.night_model if night_mode else self.day_model

        run_aux = (not self.last_aux_results) or (self.aux_frame_count % self.aux_frame_skip == 0)
        self.aux_frame_count += 1

        if run_aux and self._executor is not None:
            # 主/辅模型并行：两者互不依赖，串行时是两个推理耗时相加
            future_main = self._executor.submit(self._infer, main_model, roi)
            future_aux = self._executor.submit(self._infer, self.auxiliary_model, roi)
            main_results = future_main.result()
            aux_results = future_aux.result()
        else:
            main_results = self._infer(main_model, roi)
            # 未到 aux 周期时复用上次结果（已是整幅坐标，不要再平移一次）
            aux_results = self._infer(self.auxiliary_model, roi) if run_aux else self.last_aux_results

        if main_results is None:
            main_results = []
        if aux_results is None:
            aux_results = []

        # 把 ROI 局部坐标还原为整幅坐标（下游客度采样/比例计算依赖它）
        main_results = self._restore_coords(main_results, y_off, (height, width))
        if run_aux:
            aux_results = self._restore_coords(aux_results, y_off, (height, width))

        self.last_results = main_results
        self.last_aux_results = aux_results
        return main_results, aux_results, self.last_model_label

    def _infer(self, model, image):
        """统一推理入口：TensorRT engine 不接受 device= 参数时，去掉后重试。

        每次调用使用独立的参数 dict（不再共享、不再就地修改），因此可被多线程安全并发调用。
        """
        args = {"verbose": False, "device": self.inference_device}
        try:
            return model(image, **args)
        except TypeError:
            args.pop("device", None)
            return model(image, **args)

    def _restore_coords(self, results, y_off: int, full_shape: Tuple[int, int]):
        """把在 ROI 裁剪图上得到的检测框/掩码还原到整幅图像坐标。

        模型输入是 `image[y_off:, :]`，返回的框坐标属于 ROI 局部像素坐标系
        （boxes.orig_shape 也是 ROI 尺寸），而下游 (surface_analysis) 按整幅坐标做
        深度采样、bottom_ratio、area_ratio 计算，所以框要平移 y。

        ⚠️ 掩码要特别小心：`masks.data` 的形状是 **letterbox 预处理分辨率**
        （实测 ROI 1906x540 -> masks.data (1,192,640)），并不是 ROI 像素尺寸。
        因此必须先把它缩放到 ROI 像素尺寸，再整块贴回原图对应位置；
        直接按 ROI 坐标贴会得到完全错误的几何（bbox/质心/面积/距离全错）。
        """
        if not results or y_off <= 0:
            return results

        height, width = full_shape
        roi_h = height - y_off
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes):
                data = boxes.data.clone()
                data[:, 1] += y_off  # y1
                data[:, 3] += y_off  # y2
                res.boxes = Boxes(data, (height, width))

            masks = getattr(res, "masks", None)
            if masks is not None and masks.data is not None and len(masks.data):
                m = masks.data.float()  # (N, h_lb, w_lb) —— letterbox 预处理空间
                # letterbox 按长边缩放：宽度正好映射到 w_lb，故 r = w_lb / width；
                # 高度方向多出来的若干行是 stride 对齐补的 padding，必须先裁掉再缩放，
                # 否则掩码会整体上移（实测约 6% 偏差）。
                r = (m.shape[2] / float(width)) if width else 0.0
                if r > 0 and roi_h > 0:
                    content_h = max(1, min(int(m.shape[1]), int(round(roi_h * r))))
                    m = m[:, :content_h, :]
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(1), size=(roi_h, width), mode="bilinear", align_corners=False
                    ).squeeze(1)
                full = m.new_zeros((m.shape[0], height, width))
                full[:, y_off:, :] = m
                res.masks = Masks(full, (height, width))

        return results
