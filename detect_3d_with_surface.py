import time
import math
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_requirements, check_imshow, set_logging, increment_path
from utils.plots import plot_one_box
# 说明：check_img_size / non_max_suppression / apply_classifier / scale_coords / xyxy2xywh /
# strip_optimizer 与 select_device / load_classifier / time_synchronized 统一取自 yolov10.utils，
# 原先同时从 utils.* 导入同名函数会被此处覆盖（重复导入，已清理）。
from yolov10.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolov10.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
import platform
import shutil
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import os
from PIL import Image
from collections import defaultdict

# 新增引用：YOLO-3D 核心模块
from depth_model import DepthEstimator, OnnxDepthEstimator
from frame_pacer import FramePacer
from depth_worker import DepthAsyncWorker
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from risk_field import (
    RISK_LEVEL_ORDER,
    RISK_THRESHOLDS,
    SURFACE_TO_SCF,
    RiskFieldEngine,
    level_from_road_danger,
    risk_level,
    worse_level,
)
from road_surface_fusion import (
    RoadSurfaceAnalyzer,
    RoadSurfaceDetector,
    RoadSurfaceRiskFuser,
    RoadSurfaceVisualizer,
    StructuredOutputWriter,
    build_frame_record,
)
from risk_alerts.sound_processing.alerter import RiskSoundAlerter
from risk_alerts.warning_prompt import draw_chinese_risk_prompt

import sys
sys.path.insert(0, './yolov10')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


# --------------------------
# 风险场计算辅助函数（论文附录A公式依赖）
# --------------------------
def calc_g_v(v):
    """论文2.1节：车速对虚拟质量的影响函数g(v)（v单位：km/h）"""
    return 0.002368 * (v ** 4) - 0.3224 * (v ** 3) + 12.57 * (v ** 2) - 1492


def calc_phi_mu(mu):
    """论文2.2节：路面附着系数影响函数φ(μ)"""
    return -87290 * (mu ** 3) + 151900 * (mu ** 2) - 84630 * mu + 21780


def calc_M_b(vehicle_type, m, v):
    """论文公式(13)：计算风险源（车辆）的虚拟质量M_b"""
    psi_T = 1.000 if vehicle_type == 'car' else 1.443  # 论文表5：汽车=1.000，载货车=1.443
    g_v = calc_g_v(v)
    return psi_T * m * g_v


def calc_R_b(psi_delta, psi_rho_tau, mu):
    """论文公式(14)：计算道路条件影响因子R_b"""
    phi_mu = calc_phi_mu(mu)
    return psi_delta * psi_rho_tau * phi_mu


def SPE_v_bi(x_phi, y_phi, x_obs, y_obs, v_b, phi_v, K, k1, k3, R_b, M_b, R_i, M_i, DR_i):
    """论文附录A公式A10：运动物体安全势能（风险值）计算"""
    r_bi = np.sqrt((x_phi - x_obs) ** 2 + (y_phi - y_obs) ** 2)
    r_bi = np.maximum(r_bi, 1e-6)

    vec_r = np.stack([x_phi - x_obs, y_phi - y_obs], axis=-1)
    vec_v = np.array([np.cos(phi_v), np.sin(phi_v)])
    cos_theta_b = np.dot(vec_r, vec_v) / r_bi
    cos_theta_b = np.clip(cos_theta_b, -1.0, 1.0)

    numerator = K * R_b * M_b * R_i * M_i * (1 + DR_i) * k3
    denominator = (k1 - 1) * (r_bi ** (k1 - 1))
    term1 = np.maximum(k3 - np.abs(v_b) * cos_theta_b, 1e-6)
    term2 = np.maximum(k3 - np.abs(v_b), 1e-6)
    power_term = (term1 ** (1 - k1) / term2) ** (1 / k1)

    return (numerator / denominator) * power_term


def SPE_r_ai(x_phi, y_phi, x_obs, y_obs, K, k1, R_b, M_b, R_i, M_i, DR_i):
    """论文附录A公式A8：静止物体安全势能（风险值）计算"""
    r_bi = np.sqrt((x_phi - x_obs) ** 2 + (y_phi - y_obs) ** 2)
    r_bi = np.maximum(r_bi, 1e-6)
    return (K * R_b * M_b * R_i * M_i * (1 + DR_i)) / ((k1 - 1) * (r_bi ** (k1 - 1)))


# --------------------------
# 辅助函数：边界框相对坐标计算
# --------------------------
def bbox_rel(image_width, image_height, *xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


# --------------------------
# 速度计算函数
# --------------------------
def Estimated_speed(locations, fps, width):
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index = []
    work_IDs_prev_index = []
    work_locations = []
    work_prev_locations = []
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])
    for m, n in enumerate(present_IDs):
        if n in prev_IDs:
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:
        work_prev_locations.append(locations[0][x])
    for i in range(len(work_IDs)):
        speed.append(
            math.sqrt((work_locations[i][0] - work_prev_locations[i][0]) ** 2 +
                      (work_locations[i][1] - work_prev_locations[i][1]) ** 2) *
            width[work_locations[i][3]] / (work_locations[i][4]) * fps / 5 * 3.6 * 2)
    for i in range(len(speed)):
        speed[i] = [round(speed[i], 1), work_locations[i][2]]
    return speed


# --------------------------
# 主检测函数
# --------------------------
def detect(save_img=False, callback=None):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')

    # === 修复：初始化 cam_params ===
    try:
        with open('config/camera_params.yaml', 'r') as f:
            cam_params = yaml.safe_load(f)
    except:
        print("Warning: config/camera_params.yaml not found or invalid. Will use auto-estimated intrinsics.")
        cam_params = None
    # ============================

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 结果保存目录
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    structured_writer = None
    sound_alerter = RiskSoundAlerter()
    if opt.save_jsonl:
        structured_writer = StructuredOutputWriter(save_dir / opt.structured_dir)
        print(f"Structured output will be saved to {structured_writer.output_path}")

    # 获取视频帧率
    fps = 30
    if not webcam and source.endswith(('.mp4', '.avi', '.mov')):
        capture = cv2.VideoCapture(source)
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()

    # DeepSort初始化
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # 车型参数
    width = [0, 0.2, 1.85, 0.5, 0, 2.3, 0, 2.5]  # 各类别真实宽度
    time_person = 3  # 刹车反应时间
    locations = []
    speed = []
    current_frame_speeds = {}  # 当前帧车辆速度（ID→speed）
    prev_centers = {}  # 【关键】存储上一帧车辆位置：{ID: (x, y)}
    
    # === 新增：轨迹历史 ===
    track_history = defaultdict(list) # ID -> list of (x, z)
    # ====================

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # YOLO-3D 初始化：Depth Anything v2 和 3D BBox Estimator
    print("Initializing Depth Anything v2...")
    # 使用与 YOLO 相同的 device，但如果显存不足，DepthEstimator 会自动 fallback 到 CPU
    depth_device = device.type if device.type != 'cpu' else 'cpu'
    _onnx_path = Path(__file__).resolve().parent / 'depth_anything_v2_small.onnx'
    if getattr(opt, 'depth_onnx', False) and _onnx_path.exists():
        depth_estimator = OnnxDepthEstimator(device=depth_device)
        print("Depth backend: ONNX Runtime")
    else:
        depth_estimator = DepthEstimator(model_size='small', device=depth_device)
        print("Depth backend: transformers pipeline")

    # 需求 5：固定帧率节拍器（帧率写死，不动态调整；处理慢时丢帧，不降精度）
    pacer = FramePacer(
        target_fps=getattr(opt, 'fps', 25.0),
        enabled=not getattr(opt, 'no_pacing', False),
        realtime=webcam,
        source_fps=fps,
    )
    print(f"[frame-pacer] 目标帧率写死为 {pacer.target_fps:.0f} FPS（enabled={pacer.enabled}）")

    # 需求 6：DepthAnything 异步 worker（独立 CUDA stream；--no-depth-async 可退回同步）
    depth_async = not getattr(opt, 'no_depth_async', False)
    depth_worker = DepthAsyncWorker(depth_estimator) if depth_async else None
    depth_seeded = False  # 首帧同步算出后 seed，避免与 worker 并发进 pipeline
    
    # 3D BBox Estimator (使用已加载的相机参数)
    if cam_params:
        K = np.array([
            [cam_params['camera_matrix']['fx'], 0, cam_params['camera_matrix']['cx']],
            [0, cam_params['camera_matrix']['fy'], cam_params['camera_matrix']['cy']],
            [0, 0, 1]
        ])
    else:
        # Fallback: 自动估算内参
        vid_w, vid_h = 1920, 1080 # Default
        if not webcam and source.endswith(('.mp4', '.avi', '.mov')):
             try:
                 cap_tmp = cv2.VideoCapture(source)
                 if cap_tmp.isOpened():
                     vid_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
                     vid_h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                     cap_tmp.release()
             except:
                 pass
        
        # 估算焦距 (假设标准镜头)
        est_f = vid_w * 1.0 
        est_cx = vid_w / 2
        est_cy = vid_h / 2
        
        K = np.array([[est_f, 0, est_cx], [0, est_f, est_cy], [0, 0, 1]])
        print(f"Warning: No camera params found. Using estimated intrinsics for {vid_w}x{vid_h}")
        
    bbox3d_estimator = BBox3DEstimator(camera_matrix=K)
    road_detector = RoadSurfaceDetector(
        model_dir=opt.road_model_dir,
        preferred_device=device.type,
        roi_top_ratio=getattr(opt, 'road_roi_top', 0.5),
        parallel=not getattr(opt, 'no_road_parallel', False),
    )
    road_analyzer = RoadSurfaceAnalyzer()
    road_visualizer = RoadSurfaceVisualizer()
    road_fuser = RoadSurfaceRiskFuser()

    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

        # 二级分类器（默认关闭）
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device, weights_only=False)['model']).to(device).eval()

    # 数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        if view_img:
            view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 类别名称
    names = model.module.names if hasattr(model, 'module') else model.names

    # 图像显示函数（拼接原画面与风险场）
    def cv_show(p, im0, risk_img=None):
        height, width = im0.shape[:2]
        a = 800 / width  # 缩放比例
        size = (800, int(height * a))
        img_resize = cv2.resize(im0, size, interpolation=cv2.INTER_AREA)

        if risk_img is not None:
            # 增加 risk_img 的显示比例
            # 原始逻辑：保持 risk_img 纵横比，缩放高度匹配 img_resize
            # 新逻辑：强制设置 risk_img 宽度为 img_resize 宽度的 1/2 (或者自定义像素值，比如 500)
            
            target_h = img_resize.shape[0]
            target_w = 600 # 增加宽度，使风险图更大更清晰
            
            risk_resize = cv2.resize(risk_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            combined_img = np.hstack((img_resize, risk_resize))  # 横向拼接
            cv2.imshow(p, combined_img)
        else:
            cv2.imshow(p, img_resize)
        cv2.waitKey(1)

    # 模型预热
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        
    # Initialize Risk Field Engine & BEV Visualizer
    # 物理范围：宽16m，深25m，分辨率10cm (0.1m)
    risk_engine = RiskFieldEngine(width_meter=16, depth_meter=25, grid_res=0.2)
    
    # BEV 视图：宽400px，高600px (对应16m x 24m => 25 px/m)
    # 稍微调整高度以匹配比例
    bev_visualizer = BirdEyeView(size=(400, 625), scale=25, camera_height=1.5)
    
    t0 = time.time()

    # 遍历视频帧
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if opt.max_frames is not None and frame_idx >= opt.max_frames:
            break

        # 需求 5：固定 25 FPS 节拍 —— 落后太多则丢帧追平（不降分辨率/不降模型精度）
        if not pacer.begin_frame():
            continue

        # 图像预处理
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

            # 模型推理
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # 非极大值抑制
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 二级分类（可选）
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # === 深度估计 (需求 6：异步 worker + 独立 CUDA stream；首帧同步 seed) ===
        # 注意：im0s 可能是列表（多摄像头）或单张图
        # 这里假设是单张图处理，如果是多路视频可能需要遍历
        curr_im0_depth = im0s.copy() if not webcam else im0s[0].copy()

        if depth_worker is None:
            # 同步路径（--no-depth-async）
            depth_map = depth_estimator.estimate_depth(curr_im0_depth)
        elif not depth_seeded:
            depth_map = depth_estimator.estimate_depth(curr_im0_depth)
            depth_worker.seed(depth_map)
            depth_seeded = True
        else:
            depth_worker.submit(curr_im0_depth)   # 只保最新帧，积压时丢旧帧
            depth_map = depth_worker.latest()     # 滞后 ≤ 1 个深度处理周期
            if depth_map is None:                 # 极端兜底：worker 还没出结果
                depth_map = depth_estimator.estimate_depth(curr_im0_depth)
                depth_worker.seed(depth_map)
        road_main_results, road_aux_results, road_model_label = road_detector.detect(
            curr_im0_depth,
            conf_thres=opt.road_conf_thres,
        )
        surface_analysis = road_analyzer.analyze(
            curr_im0_depth,
            depth_map,
            K,
            road_main_results,
            road_aux_results,
            road_model_label,
        )
        # ============================

        # 处理检测结果
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]

            current_frame_distances = {}
            risk_img = None
            dynamic_risk = 0.0
            surface_risk = 0.0
            combined_risk = 0.0
            decision_status = 'CLEAR'
            warning_text = surface_analysis.warning_text
            max_risk_id = None
            risk_sources = []  # 存储当前帧风险源（含ID）

            if det is not None and len(det):
                # 调整检测框到原图尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # DeepSort输入准备
                bbox_xywh = []
                confs = []
                classes_list = []
                bbox_3d_list = [] # 存放新的 3D pose
                
                img_h, img_w, _ = im0.shape
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    bbox_xywh.append([x_c, y_c, bbox_w, bbox_h])
                    confs.append([conf.item()])
                    classes_list.append([cls.item()])
                    
                    # === 新增：基于 Depth Anything 的 3D 估算 ===
                    # 获取深度值
                    # xyxy 是 tensor，转 numpy
                    xyxy_cpu = [x.item() for x in xyxy]
                    depth_val = depth_estimator.get_depth_in_region(depth_map, xyxy_cpu, method='median')
                    
                    # 估算 3D 框 (此时没有 ID)
                    cls_name = names[int(cls)]
                    box_3d = bbox3d_estimator.estimate_3d_box(xyxy_cpu, depth_val, cls_name)
                    
                    # 构造 DeepSort 需要的 pose (X, Z, L, W, H, Yaw)
                    pose = (
                        box_3d['location'][0], # X
                        box_3d['location'][2], # Z (深度)
                        box_3d['dimensions'][2], # L
                        box_3d['dimensions'][1], # W
                        box_3d['dimensions'][0], # H
                        box_3d['orientation']  # Yaw
                    )
                    bbox_3d_list.append(pose)
                    # ==========================================

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes_list)

                # DeepSort跟踪 (传入 3D 数据以启用 3D 欧氏距离匹配)
                outputs = deepsort.update(xywhs, confss, im0, classes, bbox_3d=bbox_3d_list)

                # 过滤有效车辆类型（bicycle, car, motorcycle, bus, truck）
                valid_classes = {0, 1, 2, 3, 5, 7}
                valid_outputs = [out for out in outputs if int(out[5]) in valid_classes]

                # 速度计算（每5帧更新）
                box_centers = []
                for each_box in outputs:
                    if int(each_box[5]) in valid_classes:
                        center_x = (each_box[0] + each_box[2]) / 2
                        center_y = (each_box[1] + each_box[3]) / 2
                        box_centers.append([
                            center_x, center_y, each_box[4], each_box[5], each_box[2] - each_box[0]
                        ])
                location = box_centers
                locations.append(location)
                if len(locations) == 5:
                    if len(locations[0]) and len(locations[-1]) != 0:
                        locations = [locations[0], locations[-1]]
                        speed_list = Estimated_speed(locations, fps, width)
                        current_frame_speeds.clear()
                        for sp in speed_list:
                            current_frame_speeds[sp[1]] = sp[0]
                    locations = []

                # 收集风险源信息（含ID）
                # ---------------------------------------------------------
                # NEW LOGIC: Multi-stage Processing for Demo Visualization
                # ---------------------------------------------------------
                
                # Stage 1: Data Collection & 3D Refinement
                risk_sources = []
                
                for out in valid_outputs:
                    x1, y1, x2, y2, current_id, cls_id = out
                    
                    # Validation
                    if cls_id < 0 or cls_id >= len(names): continue
                    current_class = names[int(cls_id)]
                    car_type = 'truck' if current_class in ['bus', 'truck'] else 'car'
                    
                    # Confidence
                    match_idx = np.where(classes.numpy().flatten() == cls_id)[0]
                    conf = confss[match_idx[0]].item() if len(match_idx) > 0 else 0.0
                    conf2 = round(conf, 2)

                    # 2D Box Drawing (Base Layer)
                    xyxy = [x1, y1, x2, y2]
                    label = f'{current_class} {conf2:.2f}'
                    dis_m=plot_one_box(
                        xyxy, im0, speed, outputs, time_person,
                        label=label, color=[0, 0, 255], line_thickness=3,
                        name=current_class
                    )
                    
                    # 3D Tracking & Refinement
                    track = next((t for t in deepsort.tracker.tracks if t.track_id == current_id), None)
                    if track and track.pose_3d is not None:
                         X_raw, Z_raw, L, W, H, Yaw_raw = track.pose_3d
                         
                         # Kalman Filter Refinement
                         box_3d_raw = {
                             'location': np.array([X_raw, 0, Z_raw]),
                             'dimensions': np.array([H, W, L]),
                             'orientation': Yaw_raw,
                             'class_name': current_class
                         }
                         
                         if hasattr(bbox3d_estimator, 'refine_box_3d'):
                            box_3d_refined = bbox3d_estimator.refine_box_3d(box_3d_raw, current_id)
                            X, _, Z = box_3d_refined['location']
                            Yaw = box_3d_refined['orientation']
                         else:
                            X, Z, Yaw = X_raw, Z_raw, Yaw_raw
                            
                         # Speed
                         src_speed = current_frame_speeds.get(current_id, 0.0)
                         
                         # Prepare Draw Data
                         box_3d_draw = {
                             'bbox_2d': [x1, y1, x2, y2],
                             'location': np.array([X, 0, Z]), 
                             'dimensions': np.array([H, W, L]),
                             'orientation': Yaw,
                             'class_name': current_class,
                             'object_id': current_id,
                             'depth_value': (Z - 1.0) / 9.0 
                         }
                         
                         # Add to list
                         risk_sources.append({
                             'id': current_id,
                             'x': X, 'z': Z, 'speed': src_speed, 'yaw': Yaw,
                             'type': car_type, 'dims': np.array([H, W, L]),
                             'box_3d_draw': box_3d_draw,
                             'xyxy': [x1, y1, x2, y2],
                             'conf': conf,
                             'confidence': conf,
                             'class_name': current_class,
                             'distance': dis_m
                         })
                         
                         # Update History
                         track_history[current_id].append((X, Z))
                         if len(track_history[current_id]) > 50:
                             track_history[current_id].pop(0)
                             
                         current_frame_distances[current_id] = round(Z, 2)

                # Stage 2: Risk Field Calculation (Global)
                risk_img = None
                target_h = im0.shape[0]
                target_w = 600
                
                # Initialize Accumulators
                total_risk_map = np.zeros((risk_engine.grid_h, risk_engine.grid_w))
                vis_risk_map = np.zeros((risk_engine.grid_h, risk_engine.grid_w)) # For visualization (larger sigma)
                
                ego_v_x, ego_v_z = 0, 5 # Ego velocity (m/s)
                current_weather_factor = 1.0 
                
                max_scf = 0.0
                max_risk_id = None
                max_dist_m = 10.0 # Track max distance for Dynamic Zoom
                
                frame_scf_values = []

                if len(risk_sources) > 0:
                    for src in risk_sources:
                        # Physics Calculation
                        t_x, t_z = src['x'], src['z']
                        t_speed = src['speed'] / 3.6 # km/h -> m/s
                        t_yaw = src['yaw']
                        
                        # Update max distance
                        if t_z > max_dist_m: max_dist_m = t_z
                        
                        t_v_x = t_speed * np.sin(t_yaw)
                        t_v_z = t_speed * np.cos(t_yaw)
                        
                        # SCF Calculation (Scientific)
                        scf, ego_f, target_f, overlap = risk_engine.calculate_scf(
                            (0, 0), (t_x, t_z),
                            (ego_v_x, ego_v_z), (t_v_x, t_v_z),
                            weather_factor=current_weather_factor
                        )
                        
                        # Visualization Field (Artistic - Larger)
                        vis_target_f = risk_engine.get_visualization_field(
                            t_x, t_z, t_v_x, t_v_z, 
                            sigma_x=1.8, sigma_z=5.0, # Much larger for "Radar" feel
                            weather_factor=current_weather_factor
                        )
                        
                        src['scf'] = scf
                        frame_scf_values.append(scf)
                        total_risk_map += target_f
                        vis_risk_map = np.maximum(vis_risk_map, vis_target_f) # Use max to avoid over-saturation
                        
                        if scf > max_scf:
                            max_scf = scf
                            max_risk_id = src['id']

                surface_risk_map, surface_vis_map, surface_risk = road_fuser.build_surface_maps(surface_analysis, risk_engine)
                total_risk_map += surface_risk_map
                vis_risk_map = np.maximum(vis_risk_map, surface_vis_map)
                dynamic_risk = max_scf
                # 动态风险按 SCF 分级；路面风险按 0–1 分级后转等级；两者取"更严重者"
                # （阈值统一来自 risk_field.RISK_THRESHOLDS，不再在此写死数字）
                dynamic_level = risk_level(dynamic_risk)
                surface_level = level_from_road_danger(surface_analysis.road_danger_level)
                decision_status = worse_level(dynamic_level, surface_level)
                # combined_risk 仅用于展示/记录，把 0–1 的路面风险换算到 SCF 尺度
                combined_risk = max(dynamic_risk, surface_risk * SURFACE_TO_SCF)
                surface_dominates = (
                    RISK_LEVEL_ORDER[surface_level] >= RISK_LEVEL_ORDER[dynamic_level]
                    and bool(surface_analysis.hazards)
                )
                if decision_status == 'CLEAR':
                    warning_text = 'Path is clear'
                elif surface_dominates:
                    warning_text = f'{decision_status} road surface risk: {surface_analysis.warning_text}'
                else:
                    warning_text = f'{decision_status} dynamic object risk ahead'
                if surface_dominates:
                    max_risk_id = 'ROAD'
                max_scf = combined_risk
                for hazard in surface_analysis.hazards:
                    if hazard.z_m > max_dist_m:
                        max_dist_m = hazard.z_m

                # Stage 3: Visualization (Main View - 3D Box with Shadow)
                for src in risk_sources:
                    # Draw Box
                    im0 = bbox3d_estimator.draw_box_3d(im0, src['box_3d_draw'], risk_score=src.get('scf', 0))
                    # Draw AR Risk Projection (Ellipse on ground)
                    # DISABLED: User feedback "Cancel red in video area"
                    # im0 = bbox3d_estimator.draw_risk_projection(im0, src['box_3d_draw'], risk_score=src.get('scf', 0))

                # Stage 4: BEV Generation (Demo Mode)
                # Always generate BEV even if no sources, to show empty grid
                
                # Update Dynamic Zoom
                bev_visualizer.update_scale(max_dist_m)
                bev_visualizer.reset()
                
                if len(risk_sources) > 0:
                    # 1. Draw Trajectories & Future Sectors (Background)
                    for src in risk_sources:
                        # Trajectory Fading
                        hist = track_history[src['id']]
                        bev_visualizer.draw_trajectory_fading(hist)
                        
                        # Future Sector
                        bev_visualizer.draw_future_sector(
                            src['x'], src['z'], 
                            src['speed']/3.6, 
                            src['yaw'], 
                            risk_level=src.get('scf', 0)
                        )
                    
                    # 2. Draw Vehicles
                    for src in risk_sources:
                        box_3d_bev = {
                            'class_name': src['type'],
                            'location': np.array([src['x'], 0, src['z']]),
                            'dimensions': src['dims'],
                            'orientation': src['yaw'],
                            'object_id': src['id'],
                            'depth_value': (src['z'] - 1.0) / 9.0 
                        }
                        # 补传 risk_score（SCF），否则 BEV 配色与中文提示永远不会触发
                        bev_visualizer.draw_box(box_3d_bev, risk_score=src.get('scf', 0))
                        
                    # 3. Draw Risk Heatmap (Glowing Overlay)
                    # Use vis_risk_map for better visual
                    total_risk_map_flipped = cv2.flip(vis_risk_map, 0)
                    bev_visualizer.draw_risk_heatmap(total_risk_map_flipped)
                    road_visualizer.draw_on_bev(bev_visualizer, surface_analysis)
                    
                    # 4. HUD
                    bev_visualizer.draw_hud(max_scf, max_risk_id, ego_speed=ego_v_z*3.6)
                    
                    # Log
                    avg_scf = sum(frame_scf_values) / len(frame_scf_values) if frame_scf_values else 0.0
                    try:
                        with open('vehicle_info.txt', 'a') as f:
                            for src in risk_sources:
                                 f.write(f"Frame: {frame}, ID: {src['id']}, Type: {src['type']}, X: {src['x']:.2f}, Z: {src['z']:.2f}, Speed: {src['speed']:.2f}, SCF: {src.get('scf', 0):.4f}, Avg_SCF: {avg_scf:.4f}\n")
                    except Exception as e:
                        print(f"Error writing log: {e}")
                else:
                    if np.max(vis_risk_map) > 0:
                        bev_visualizer.draw_risk_heatmap(cv2.flip(vis_risk_map, 0))
                    road_visualizer.draw_on_bev(bev_visualizer, surface_analysis)
                    bev_visualizer.draw_hud(max_scf, max_risk_id, ego_speed=ego_v_z*3.6)
                
                # Get Image
                risk_img = bev_visualizer.get_image()
                
                # Resize
                if risk_img.shape[0] != target_h or risk_img.shape[1] != target_w:
                    risk_img = cv2.resize(risk_img, (target_w, target_h))



            # 打印耗时
            if risk_img is None:
                target_h = im0.shape[0]
                target_w = 600
                ego_v_z = 5
                max_dist_m = 10.0
                vis_risk_map = np.zeros((risk_engine.grid_h, risk_engine.grid_w))
                surface_risk_map, surface_vis_map, surface_risk = road_fuser.build_surface_maps(surface_analysis, risk_engine)
                vis_risk_map = np.maximum(vis_risk_map, surface_vis_map)
                dynamic_level = risk_level(dynamic_risk)
                surface_level = level_from_road_danger(surface_analysis.road_danger_level)
                decision_status = worse_level(dynamic_level, surface_level)
                combined_risk = max(dynamic_risk, surface_risk * SURFACE_TO_SCF)
                if decision_status == 'CLEAR':
                    warning_text = 'Path is clear'
                elif (
                    RISK_LEVEL_ORDER[surface_level] >= RISK_LEVEL_ORDER[dynamic_level]
                    and surface_analysis.hazards
                ):
                    warning_text = f'{decision_status} road surface risk: {surface_analysis.warning_text}'
                    max_risk_id = 'ROAD'
                else:
                    warning_text = f'{decision_status} dynamic object risk ahead'
                for hazard in surface_analysis.hazards:
                    if hazard.z_m > max_dist_m:
                        max_dist_m = hazard.z_m

                bev_visualizer.update_scale(max_dist_m)
                bev_visualizer.reset()
                if np.max(vis_risk_map) > 0:
                    bev_visualizer.draw_risk_heatmap(cv2.flip(vis_risk_map, 0))
                road_visualizer.draw_on_bev(bev_visualizer, surface_analysis)
                bev_visualizer.draw_hud(combined_risk, max_risk_id, ego_speed=ego_v_z*3.6)
                risk_img = bev_visualizer.get_image()
                if risk_img.shape[0] != target_h or risk_img.shape[1] != target_w:
                    risk_img = cv2.resize(risk_img, (target_w, target_h))

            im0 = road_visualizer.draw_on_frame(im0, surface_analysis)
            im0 = draw_chinese_risk_prompt(im0, decision_status)

            frame_record = None
            if structured_writer is not None or sound_alerter.enabled:
                frame_record = build_frame_record(
                    source=str(p),
                    stream_index=i,
                    frame_index=frame_idx,
                    source_frame_index=frame,
                    image_shape=im0.shape,
                    dynamic_targets=risk_sources,
                    surface_analysis=surface_analysis,
                    dynamic_risk=dynamic_risk,
                    surface_risk=surface_risk,
                    combined_risk=combined_risk,
                    decision_status=decision_status,
                    warning_text=warning_text,
                    max_risk_source=max_risk_id,
                )

            if structured_writer is not None and frame_record is not None:
                structured_writer.write_frame(frame_record)
            if frame_record is not None:
                sound_alerter.handle_frame_record(frame_record)

            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 显示结果
            if view_img:
                cv_show(str(p), im0, risk_img)
            if callback is not None:
                callback(im0, risk_img, frame_idx=frame_idx, risk_sources=risk_sources, frame_risk=combined_risk)

            # 保存结果
            if save_img:
                if dataset.mode == 'image':
                    if risk_img is not None:
                        combined_img = np.hstack((im0, risk_img))
                        cv2.imwrite(save_path.replace('.jpg', '_with_risk.jpg'), combined_img)
                    else:
                        cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps_vid = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if risk_img is not None:
                                w += risk_img.shape[1]  # 拼接后宽度增加
                        else:
                            fps_vid, w, h = 30, im0.shape[1], im0.shape[0]
                            if risk_img is not None:
                                w += risk_img.shape[1]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (w, h)
                        )
                    if risk_img is not None:
                        combined_img = np.hstack((im0, risk_img))
                        vid_writer.write(combined_img)
                    else:
                        vid_writer.write(im0)

        # 【关键】更新上一帧车辆位置（供下一帧计算方向）
        # risk_sources 现在使用 'z' 代表纵向位置
        current_centers = {src['id']: (src['x'], src['z']) for src in risk_sources}
        prev_centers = current_centers

    if depth_worker is not None:
        depth_worker.shutdown()
        print(depth_worker.status_line())
    print(pacer.status_line())

    if structured_writer is not None:
        structured_writer.close()

    # 保存信息
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        structured_msg = f"\nStructured jsonl saved to {structured_writer.output_path}" if structured_writer is not None else ''
        print(f"Results saved to {save_dir}{s}{structured_msg}")
    elif structured_writer is not None:
        print(f"Structured jsonl saved to {structured_writer.output_path}")

    # 总耗时
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov10s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='lanechange.mp4', help='source')  # 输入视频路径
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', dest='view_img', action='store_true', help='display results')
    parser.add_argument('--no-view-img', dest='view_img', action='store_false', help='disable display results')
    parser.set_defaults(view_img=True, save_jsonl=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument('--road-model-dir', type=str, default=str(Path(__file__).resolve().parent / 'code' / 'models'))
    parser.add_argument('--road-conf-thres', type=float, default=0.25)
    parser.add_argument('--road-roi-top', type=float, default=0.5,
                        help='路面模型只推理画面下方该比例以上的区域（0=关闭 ROI，用整幅）')
    parser.add_argument('--no-road-parallel', action='store_true',
                        help='关闭主/辅路面模型并行推理（退回串行，便于对照排查）')
    parser.add_argument('--depth-backend', choices=['depth-anything'], default='depth-anything')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--depth-onnx', action='store_true', help='use the exported ONNX depth model instead of the transformers pipeline')
    parser.add_argument('--save-jsonl', dest='save_jsonl', action='store_true', help='enable per-frame structured jsonl export')
    parser.add_argument('--no-save-jsonl', dest='save_jsonl', action='store_false', help='disable per-frame structured jsonl export')
    parser.add_argument('--structured-dir', type=str, default='structured', help='structured output subdirectory name')
    # 需求 5/6：固定帧率节拍 + 异步深度
    parser.add_argument('--fps', type=float, default=25.0, help='目标帧率，写死不动态调整（默认 25）')
    parser.add_argument('--no-pacing', action='store_true', help='关闭固定帧率节拍（不限速、不丢帧）')
    parser.add_argument('--no-depth-async', action='store_true', help='关闭深度异步 worker，退回每帧同步推理')
    opt = parser.parse_args()
    print(opt)
    check_requirements('requirements_common.txt', exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
