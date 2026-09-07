import os
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
    def __init__(self, model_size='small', device=None, half=None, input_scale=1.0):
        """
        Initialize the depth estimator
        
        Args:
            model_size (str): Model size ('small', 'base', 'large')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
            half (bool, optional): Use FP16 inference on CUDA (默认: CUDA 时开启)
            input_scale (float, optional): 输入图降采样比例 0.25~1.0，
                由 RuntimeGovernor 在负载高时自动调低（CPU 侧预处理节省 + GPU 端省带宽）
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

        # 输入降采样（运行时负载高时由 Governor 调低）
        self.input_scale = max(0.25, min(float(input_scale if input_scale and input_scale > 0 else 1.0), 1.0))

        # 最近一帧深度图缓存（Governor 跳过本帧估算时直接复用）
        self.last_depth_map = None

        # Create pipeline
        # Note: transformers pipeline handles device placement
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name,
                                 device=self.pipe_device, model_kwargs=model_kwargs)
            self.half = bool(model_kwargs)
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
            numpy.ndarray: Depth map (normalized to 0-1)
        """
        original_h, original_w = image.shape[:2]

        # 负载高时对输入降采样（由 RuntimeGovernor.depth_input_scale() 给出比例）
        if self.input_scale < 1.0:
            small_w = max(64, int(original_w * self.input_scale))
            small_h = max(64, int(original_h * self.input_scale))
            image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Get depth map
        try:
            depth_result = self.pipe(pil_image)
            depth_map = depth_result["depth"]
            
            # Convert PIL Image to numpy array if needed
            if isinstance(depth_map, Image.Image):
                depth_map = np.array(depth_map)
            elif isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu().numpy()
        except RuntimeError as e:
            # Handle potential MPS errors during inference
            if self.device == 'mps':
                print(f"MPS error during depth estimation: {e}")
                print("Temporarily falling back to CPU for this frame")
                # Create a CPU pipeline for this frame
                cpu_pipe = pipeline(task="depth-estimation", model=self.pipe.model.config._name_or_path, device='cpu')
                depth_result = cpu_pipe(pil_image)
                depth_map = depth_result["depth"]
                
                # Convert PIL Image to numpy array if needed
                if isinstance(depth_map, Image.Image):
                    depth_map = np.array(depth_map)
                elif isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
            else:
                # Re-raise the error if not MPS
                raise
        
        # Normalize depth map to 0-1
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map
    
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