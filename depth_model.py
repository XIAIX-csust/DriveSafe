import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

class DepthEstimator:
    """
    Depth estimation using Depth Anything v2
    """
    def __init__(self, model_size='small', device=None, half=None):
        """
        Initialize the depth estimator
        
        Args:
            model_size (str): Model size ('small', 'base', 'large')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
            half (bool, optional): Use FP16 inference on CUDA (默认: CUDA 时开启)
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Set MPS fallback for operations not supported on Apple Silicon
        if self.device == 'mps':
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # For Depth Anything v2, we'll use CPU directly due to MPS compatibility issues
            self.pipe_device = 'cpu'
            print("Forcing CPU for depth estimation pipeline due to MPS compatibility issues")
        else:
            self.pipe_device = self.device
        
        print(f"Using device: {self.device} for depth estimation (pipeline on {self.pipe_device})")
        
        # Map model size to model name
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        
        model_name = model_map.get(model_size.lower(), model_map['small'])

        # FP16（Jetson 上收益明显；不支持时自动回退 FP32）
        self.half = False
        if half is None:
            half = (self.pipe_device == 'cuda')
        if half and self.pipe_device != 'cpu':
            model_kwargs = {"torch_dtype": torch.float16}
        else:
            model_kwargs = {}
        # 注意：transformers 会就地修改传入的 model_kwargs（实测 {} -> {'dtype': 'auto'}），
        # 所以在调用前记录"是否请求 FP16"，并把副本传进去。
        use_fp16 = bool(model_kwargs)

        # 最近一帧深度图缓存（供外部/异步 worker 复用，避免重复推理）
        self.last_depth_map = None

        # Create pipeline
        # Note: transformers pipeline handles device placement
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name,
                                 device=self.pipe_device, model_kwargs=dict(model_kwargs))
            # 以模型实际 dtype 为准，避免 CPU 上被就地修改的 dict 误判成 FP16
            self.half = bool(use_fp16)
            model_dtype = getattr(getattr(self.pipe, "model", None), "dtype", None)
            if self.half and model_dtype is not None and model_dtype != torch.float16:
                print(f"Warning: requested FP16 but loaded dtype is {model_dtype}; using FP32 path")
                self.half = False
            print(f"Loaded Depth Anything v2 {model_size} model on {self.pipe_device}"
                  f"{' (FP16)' if self.half else ''}")
        except Exception as e:
            # Fallback to CPU if there are issues
            print(f"Error loading model on {self.pipe_device}: {e}")
            print("Falling back to CPU for depth estimation")
            self.pipe_device = 'cpu'
            self.half = False
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 {model_size} model on CPU (fallback)")

        # Jetson 兼容自检：个别 CUDA 驱动对 FP16 ViT 推理不兼容。
        # 启动时用小图试跑一次，失败自动回退 FP32，避免运行到一半才崩溃。
        if self.half and self.pipe_device != 'cpu':
            try:
                probe = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                self.pipe(probe)
                print("Depth Anything FP16 self-check passed")
            except Exception as exc:
                print(f"Depth Anything FP16 self-check failed ({exc}); reloading in FP32")
                self.half = False
                try:
                    self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
                    print("Depth Anything reloaded in FP32")
                except Exception as exc2:
                    print(f"Depth Anything FP32 reload also failed ({exc2})")
    
    def estimate_depth(self, image):
        """
        Estimate depth from an image

        Args:
            image (numpy.ndarray): Input image (BGR format)

        Returns:
            numpy.ndarray: Depth map, float32, normalized to 0-1, 尺寸与原图一致
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        target_hw = image.shape[:2]

        # Get depth map
        try:
            depth_result = self.pipe(pil_image)
        except RuntimeError as e:
            # Handle potential MPS errors during inference
            if self.device == 'mps':
                print(f"MPS error during depth estimation: {e}")
                print("Temporarily falling back to CPU for this frame")
                cpu_pipe = pipeline(task="depth-estimation", model=self.pipe.model.config._name_or_path, device='cpu')
                depth_result = cpu_pipe(pil_image)
            else:
                # Re-raise the error if not MPS
                raise

        depth_map = self._to_normalized_depth(depth_result, target_hw)

        # 缓存最近一次结果（供异步 worker / 外部复用）
        self.last_depth_map = depth_map

        return depth_map

    @staticmethod
    def _to_normalized_depth(depth_result, target_hw=None):
        """把 pipeline 输出转成 0-1 的 float32 深度图。

        优先使用模型的原生 float 预测 ``predicted_depth``（约 518×518）：
          - 避免 pipeline 把结果量化成 8bit PIL（只有 256 级，而实测有效区间只占
            0.03~0.18，等价于仅约 38 级用于区分所有距离）；
          - 避免 pipeline 用双三次插值放大到原分辨率（1906×1080 ≈ 206 万像素）后
            我们只取 bbox 中心小区域、99% 的插值像素被丢弃。
        归一化在低分辨率上完成（更省），再自行缩放到原图尺寸（cv2 比 PIL 快），
        从而保持下游（bbox 采样 / 路面分析）坐标语义不变。
        """
        raw = None
        if isinstance(depth_result, dict):
            raw = depth_result.get("predicted_depth")

        if isinstance(raw, torch.Tensor):
            arr = raw.detach().to(torch.float32).cpu().numpy()
        elif raw is not None:
            arr = np.asarray(raw, dtype=np.float32)
        else:
            # 回退：老路径（uint8 PIL 图）
            legacy = depth_result["depth"] if isinstance(depth_result, dict) else depth_result
            arr = np.array(legacy) if isinstance(legacy, Image.Image) else np.asarray(legacy)
            arr = arr.astype(np.float32)

        if arr.ndim == 3:          # (1, H, W) -> (H, W)
            arr = arr[0]

        depth_min = float(arr.min())
        depth_max = float(arr.max())
        if depth_max > depth_min:
            arr = (arr - depth_min) / (depth_max - depth_min)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        if target_hw is not None and tuple(arr.shape[:2]) != tuple(target_hw):
            arr = cv2.resize(arr, (int(target_hw[1]), int(target_hw[0])),
                             interpolation=cv2.INTER_LINEAR)

        return np.ascontiguousarray(arr, dtype=np.float32)
    
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization
        
        Args:
            depth_map (numpy.ndarray): Depth map (normalized to 0-1)
            cmap (int): OpenCV colormap
            
        Returns:
            numpy.ndarray: Colorized depth map (BGR format)
        """
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth
    
    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at a specific point
        
        Args:
            depth_map (numpy.ndarray): Depth map
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Depth value at (x, y)
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0
    
    def get_depth_in_region(self, depth_map, bbox, method='median', scale=0.5):
        """
        Get depth value in a region defined by a bounding box
        
        Args:
            depth_map (numpy.ndarray): Depth map
            bbox (list): Bounding box [x1, y1, x2, y2]
            method (str): Method to compute depth ('median', 'mean', 'min')
            scale (float): Scale factor for the region (default 0.5 to use center 50%)
            
        Returns:
            float: Depth value in the region
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Apply scaling to focus on the center region
        # This helps avoid background noise (e.g., road, other objects)
        new_width = width * scale
        new_height = height * scale
        
        x1 = int(center_x - new_width / 2)
        y1 = int(center_y - new_height / 2)
        x2 = int(center_x + new_width / 2)
        y2 = int(center_y + new_height / 2)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)
        
        # Extract region
        region = depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        # Compute depth based on method
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region))


class OnnxDepthEstimator:
    """
    Depth estimation using an exported ONNX model.

    Preprocessing follows the official Depth Anything recipe (resize to 518x518,
    ImageNet normalize); only the heavy backbone/head runs through ONNX Runtime.
    This is faster than the transformers pipeline and directly reusable with the
    CUDA/TensorRT execution providers on a Jetson device.
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _INPUT_SIZE = 518

    def __init__(self, onnx_path=None, device=None):
        import onnxruntime as ort

        if onnx_path is None:
            onnx_path = Path(__file__).resolve().parent / 'depth_anything_v2_small.onnx'
        self.onnx_path = str(onnx_path)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Loaded ONNX depth model on {device}")

    def estimate_depth(self, image):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (self._INPUT_SIZE, self._INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = (rgb - self._MEAN) / self._STD
        pixel_values = rgb.transpose(2, 0, 1)[None].astype(np.float32)

        depth = self.session.run([self.output_name], {self.input_name: pixel_values})[0][0]

        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        return depth

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth

    def get_depth_at_point(self, depth_map, x, y):
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0

    def get_depth_in_region(self, depth_map, bbox, method='median', scale=0.5):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        new_width = width * scale
        new_height = height * scale

        x1 = int(center_x - new_width / 2)
        y1 = int(center_y - new_height / 2)
        x2 = int(center_x + new_width / 2)
        y2 = int(center_y + new_height / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region))
