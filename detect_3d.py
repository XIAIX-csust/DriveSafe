import time
import math
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov10.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolov10.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from counter.draw_counter import draw_up_down_counter
import argparse
import platform
import shutil
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import os
from PIL import Image
from pylab import *
from matplotlib.pyplot import ginput, ion, ioff
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# 新增引用：YOLO-3D 核心模块
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from risk_field import RiskFieldEngine
from runtime_governor import RuntimeGovernor

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
# 风险场热力图生成函数（优化版）
# --------------------------
def generate_bev_map(risk_sources, width_meter=20, depth_meter=100, map_w=400, map_h=800, tracks=None):
    """
    生成物理坐标系的 BEV 风险场热力图
    :param tracks: 车辆轨迹字典 {id: [(x, z), ...]}
    """
    # 1. 论文标定参数初始化
    m = 1500;
    mu = 0.9;
    psi_delta = 1.457;
    psi_rho_tau = 1.000;
    DR_i = 0.5
    R_common = calc_R_b(psi_delta, psi_rho_tau, mu)
    M_i = calc_M_b('car', m, v=0)
    K_risk = 0.1;
    k1 = 1.5;
    k3 = 160

    # 2. 生成物理网格 (X: -10m~10m, Z: 0m~100m)
    # 显著增加横向分辨率，确保左右车道区分清晰
    grid_res_w = 400
    grid_res_h = 400
    x_phi = np.linspace(-width_meter / 2, width_meter / 2, grid_res_w)
    y_phi = np.linspace(0, depth_meter, grid_res_h)
    x_phi, y_phi = np.meshgrid(x_phi, y_phi)

    # 3. 计算总风险场
    U_total = np.zeros_like(x_phi)
    for src in risk_sources:
        x_obs, z_obs = src['x'], src['z']
        v_b = src['speed']
        car_type = src['type']

        M_b = calc_M_b(car_type, m, v_b)

        # 运动物体 (使用 yaw 作为方向)
        if v_b > 1e-3:
            phi_v = src.get('yaw', 0.0)
            # 注意：调整角度方向以匹配网格坐标系
            single_risk = SPE_v_bi(x_phi, y_phi, x_obs, z_obs, v_b, phi_v, K_risk, k1, k3, R_common, M_b, R_common, M_i,
                                   DR_i)
        else:
            single_risk = SPE_r_ai(x_phi, y_phi, x_obs, z_obs, K_risk, k1, R_common, M_b, R_common, M_i, DR_i)

        U_total += single_risk

    # 4. 归一化
    if U_total.max() > 0:
        threshold = np.percentile(U_total, 98) if np.any(U_total) else 1.0
        U_total_clipped = np.clip(U_total, 0, threshold)
        p30 = np.percentile(U_total_clipped, 30) if np.any(U_total_clipped) else 0.1
        if p30 == 0: p30 = 0.1

        U_norm = np.zeros_like(U_total_clipped)
        mask_low = U_total_clipped <= p30
        mask_high = ~mask_low
        U_norm[mask_low] = (U_total_clipped[mask_low] / p30) * 0.3
        denom = (threshold - p30)
        if denom == 0: denom = 1
        U_norm[mask_high] = ((U_total_clipped[mask_high] - p30) / denom) * 0.7 + 0.3
    else:
        U_norm = U_total

    # 5. 绘制热力图
    plt.switch_backend('Agg')
    dpi = 100
    fig = plt.figure(figsize=(map_w / dpi, map_h / dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.set_xlim(-width_meter / 2, width_meter / 2)
    ax.set_ylim(0, depth_meter)  # 0(近处)在下, 100(远处)在上

    # 配色方案
    levels = 100
    colors = [(0.1, 0.1, 0.8), (0.2, 0.7, 0.2), (1.0, 0.5, 0), (1.0, 0, 0)]
    cmap_custom = LinearSegmentedColormap.from_list('high_contrast', colors, N=levels)

    ax.contourf(x_phi, y_phi, U_norm, levels=levels, cmap=cmap_custom, alpha=0.9, antialiased=True)

    # 绘制轨迹
    if tracks:
        for src in risk_sources:
            tid = src['id']
            if tid in tracks:
                hist = tracks[tid]
                if len(hist) > 1:
                    # hist is list of (x, z)
                    px = [p[0] for p in hist]
                    py = [p[1] for p in hist]
                    ax.plot(px, py, color='white', linewidth=1, alpha=0.4, linestyle='--')

    # 标记风险源
    for src in risk_sources:
        # ax.scatter(src['x'], src['z'], c='red', s=150, edgecolors='white', linewidth=2, zorder=10)

        # 使用“水滴型”标记 (Marker) 代替丑陋的箭头
        # 这种标记本身就带有方向性，看起来更像雷达界面
        # 使用自定义 MarkerPath 或者简单的 matplotlib 预设
        # '^' 是正三角形，我们可以旋转它来表示方向

        x, z = src['x'], src['z']
        yaw = src.get('yaw', 0)

        # 将 yaw 转换为度数 (matplotlib 旋转是逆时针为正，0度是向右)
        # 我们的坐标系：0度是Z轴正向(向上)
        # 所以转换公式：angle = -yaw_deg + 90
        yaw_deg = np.degrees(yaw)
        marker_angle = -yaw_deg + 90

        # 使用自定义的 Marker 绘制带方向的圆点
        # 这里的 marker=(3, 0, angle) 表示 3边形(三角形)，0是风格，angle是旋转
        # 但 scatter 的 marker 旋转比较麻烦，不如直接用 RegularPolygon

        # 方案B：绘制一个圆点 + 一个短线指示方向 (经典的雷达图画法)
        # 圆点：缩小尺寸，避免视觉上“粘连”
        # 原来 radius=0.4 -> 现在 0.3
        circle = plt.Circle((x, z), radius=0.3, fc='red', ec='white', lw=1.5, zorder=10)
        ax.add_patch(circle)

        # 方向线 (从圆心伸出)
        line_len = 0.8
        dx = line_len * np.sin(yaw)
        dz = line_len * np.cos(yaw)
        ax.plot([x, x + dx], [z, z + dz], color='white', lw=2, zorder=11)

    fig.canvas.draw()

    # 兼容不同版本 Matplotlib 获取图像数据
    try:
        if hasattr(fig.canvas, 'buffer_rgba'):
            # Matplotlib 3.8+
            buf = fig.canvas.buffer_rgba()
            risk_rgb = np.asarray(buf)
            # RGBA 转 BGR
            if risk_rgb.shape[2] == 4:
                risk_bgr = cv2.cvtColor(risk_rgb, cv2.COLOR_RGBA2BGR)
            else:
                risk_bgr = cv2.cvtColor(risk_rgb, cv2.COLOR_RGB2BGR)
        else:
            # 旧版 Matplotlib
            buf = fig.canvas.tostring_rgb()
            risk_rgb = np.frombuffer(buf, dtype=np.uint8).reshape(map_h, map_w, 3)
            risk_bgr = cv2.cvtColor(risk_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error generating BEV map: {e}")
        risk_bgr = np.zeros((map_h, map_w, 3), dtype=np.uint8)

    plt.close(fig)
    return risk_bgr


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
    track_history = defaultdict(list)  # ID -> list of (x, z)
    # ====================

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # YOLO-3D 初始化：Depth Anything v2 和 3D BBox Estimator
    print("Initializing Depth Anything v2...")
    # 使用与 YOLO 相同的 device，但如果显存不足，DepthEstimator 会自动 fallback 到 CPU
    depth_estimator = DepthEstimator(model_size='small', device=device.type if device.type != 'cpu' else 'cpu')

    # 3D BBox Estimator (使用已加载的相机参数)
    if cam_params:
        K = np.array([
            [cam_params['camera_matrix']['fx'], 0, cam_params['camera_matrix']['cx']],
            [0, cam_params['camera_matrix']['fy'], cam_params['camera_matrix']['cy']],
            [0, 0, 1]
        ])
    else:
        # Fallback: 自动估算内参
        vid_w, vid_h = 1920, 1080  # Default
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

    # 运行时负载调度：按实测 FPS 动态决定深度/REID/BEV 是否本帧执行
    governor = RuntimeGovernor(
        target_fps=getattr(opt, 'target_fps', None),
        depth_strategy=getattr(opt, 'depth_strategy', None),
        road_strategy=getattr(opt, 'road_strategy', None),
        reid_strategy=getattr(opt, 'reid_strategy', None),
        bev_strategy=getattr(opt, 'bev_strategy', None),
    )
    last_risk_img = None  # BEV 抽帧渲染时复用上一帧画面

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
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device, weights_only=False)['model']).to(
            device).eval()

    # 数据加载器
    vid_path, vid_writer = None, None
    if webcam:
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
            target_w = 600  # 增加宽度，使风险图更大更清晰

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
    risk_engine = RiskFieldEngine(width_meter=16, depth_meter=25, grid_res=0.1)

    # BEV 视图：宽400px，高600px (对应16m x 24m => 25 px/m)
    # 稍微调整高度以匹配比例
    bev_visualizer = BirdEyeView(size=(400, 625), scale=25, camera_height=1.5)

    t0 = time.time()

    # 遍历视频帧
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        loop_t0 = time.time()

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

        # === 深度估计 (Governor 动态调控：负载高时隔帧/降采样，帧间复用深度图) ===
        # 注意：im0s 可能是列表（多摄像头）或单张图
        # 这里假设是单张图处理，如果是多路视频可能需要遍历
        curr_im0_depth = im0s.copy() if not webcam else im0s[0].copy()

        depth_estimator.input_scale = governor.depth_input_scale()
        if governor.should_run_depth() or depth_estimator.last_depth_map is None:
            depth_map = depth_estimator.estimate_depth(curr_im0_depth)
        else:
            depth_map = depth_estimator.last_depth_map
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
            risk_sources = []  # 存储当前帧风险源（含ID）
            risk_img = None  # 无检测/未渲染帧时兜底（避免 NameError）

            if det is not None and len(det):
                # 调整检测框到原图尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # DeepSort输入准备
                bbox_xywh = []
                confs = []
                classes_list = []
                bbox_3d_list = []  # 存放新的 3D pose

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
                        box_3d['location'][0],  # X
                        box_3d['location'][2],  # Z (深度)
                        box_3d['dimensions'][2],  # L
                        box_3d['dimensions'][1],  # W
                        box_3d['dimensions'][0],  # H
                        box_3d['orientation']  # Yaw
                    )
                    bbox_3d_list.append(pose)
                    # ==========================================

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes_list)

                # DeepSort跟踪 (传入 3D 数据以启用 3D 欧氏距离匹配；负载高时按 Governor 复用特征)
                outputs = deepsort.update(
                    xywhs, confss, im0, classes,
                    bbox_3d=bbox_3d_list,
                    extract_reid=governor.should_run_reid(),
                )

                # 过滤有效车辆类型（bicycle, car, motorcycle, bus, truck）
                valid_classes = {1, 2, 3, 5, 7}
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
                vis_risk_map = np.zeros((risk_engine.grid_h, risk_engine.grid_w))  # For visualization (larger sigma)

                ego_v_x, ego_v_z = 0, 5  # Ego velocity (m/s)
                current_weather_factor = 1.0

                max_scf = 0.0
                max_risk_id = None
                max_dist_m = 10.0  # Track max distance for Dynamic Zoom

                frame_scf_values = []

                if len(risk_sources) > 0:
                    for src in risk_sources:
                        # Physics Calculation
                        t_x, t_z = src['x'], src['z']
                        t_speed = src['speed'] / 3.6  # km/h -> m/s
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
                            sigma_x=1.8, sigma_z=5.0,  # Much larger for "Radar" feel
                            weather_factor=current_weather_factor
                        )

                        src['scf'] = scf
                        frame_scf_values.append(scf)
                        total_risk_map += target_f
                        vis_risk_map = np.maximum(vis_risk_map, vis_target_f)  # Use max to avoid over-saturation

                        if scf > max_scf:
                            max_scf = scf
                            max_risk_id = src['id']

                # Stage 3: Visualization (Main View - 3D Box with Shadow)
                for src in risk_sources:
                    # Draw Box
                    im0 = bbox3d_estimator.draw_box_3d(im0, src['box_3d_draw'], risk_score=src.get('scf', 0))
                    # Draw AR Risk Projection (Ellipse on ground)
                    # DISABLED: User feedback "Cancel red in video area"
                    # im0 = bbox3d_estimator.draw_risk_projection(im0, src['box_3d_draw'], risk_score=src.get('scf', 0))

                # Stage 4: BEV Generation (Demo Mode)
                # Always generate BEV even if no sources, to show empty grid
                render_bev = governor.should_render_bev()

                # Update Dynamic Zoom
                bev_visualizer.update_scale(max_dist_m)
                bev_visualizer.reset()

                if render_bev and len(risk_sources) > 0:
                    # 1. Draw Trajectories & Future Sectors (Background)
                    for src in risk_sources:
                        # Trajectory Fading
                        hist = track_history[src['id']]
                        bev_visualizer.draw_trajectory_fading(hist)

                        # Future Sector
                        bev_visualizer.draw_future_sector(
                            src['x'], src['z'],
                            src['speed'] / 3.6,
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
                        bev_visualizer.draw_box(box_3d_bev)

                    # 3. Draw Risk Heatmap (Glowing Overlay)
                    # Use vis_risk_map for better visual
                    total_risk_map_flipped = cv2.flip(vis_risk_map, 0)
                    bev_visualizer.draw_risk_heatmap(total_risk_map_flipped)

                    # 4. HUD
                    bev_visualizer.draw_hud(max_scf, max_risk_id, ego_speed=ego_v_z * 3.6)

                    # Log
                    avg_scf = sum(frame_scf_values) / len(frame_scf_values) if frame_scf_values else 0.0
                    try:
                        with open('vehicle_info.txt', 'a') as f:
                            for src in risk_sources:
                                f.write(
                                    f"Frame: {frame}, ID: {src['id']}, Type: {src['type']}, X: {src['x']:.2f}, Z: {src['z']:.2f}, Speed: {src['speed']:.2f}, SCF: {src.get('scf', 0):.4f}, Avg_SCF: {avg_scf:.4f}\n")
                    except Exception as e:
                        print(f"Error writing log: {e}")

                if render_bev:
                    # Get Image
                    risk_img = bev_visualizer.get_image()

                    # Resize
                    if risk_img.shape[0] != target_h or risk_img.shape[1] != target_w:
                        risk_img = cv2.resize(risk_img, (target_w, target_h))
                    last_risk_img = risk_img
                else:
                    # 抽帧渲染：跳过帧复用上一帧 BEV，省掉 matplotlib 开销
                    risk_img = last_risk_img
                    if risk_img is not None and (risk_img.shape[0] != target_h or risk_img.shape[1] != target_w):
                        risk_img = cv2.resize(risk_img, (target_w, target_h))

            # 打印耗时
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 显示结果
            if view_img:
                cv_show(str(p), im0, risk_img)
            if callback is not None:
                callback(im0, risk_img, frame_idx=frame_idx, risk_sources=risk_sources)

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

        # 记录本帧耗时给 Governor（只算推理，不含显示/保存）
        governor.tick(time.time() - loop_t0)

    # 保存信息
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

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
    parser.add_argument('--view-img', action='store_true', help='display results', default=True)
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
    # ── 运行时负载调度（RuntimeGovernor）──
    parser.add_argument('--target-fps', type=float, default=None,
                        help='期望最低 FPS（缺省 15，可用环境变量 DRIVESAFE_TARGET_FPS 覆盖）')
    parser.add_argument('--depth-strategy', choices=['auto', 'always', 'off'], default='auto',
                        help='深度估计策略：auto=按负载动态隔帧/降采样；always=每帧；off=关闭')
    parser.add_argument('--road-strategy', choices=['auto', 'always', 'off'], default='auto',
                        help='路面双模型策略（本入口未使用路面模型，仅为统一参数）')
    parser.add_argument('--reid-strategy', choices=['auto', 'always', 'off'], default='auto',
                        help='DeepSort REID 特征策略：auto=负载高时按 IoU 复用上帧特征')
    parser.add_argument('--bev-strategy', choices=['auto', 'always', 'off'], default='auto',
                        help='BEV 热力图渲染策略：auto=负载高时抽帧渲染')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
