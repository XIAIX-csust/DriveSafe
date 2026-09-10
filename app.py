import streamlit as st
import cv2
import numpy as np
import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 风险等级阈值统一来源（SCF 尺度）
from risk_field import RISK_THRESHOLDS, risk_level



# ─────────────────────────────────────────────
#  DummyOpt
# ─────────────────────────────────────────────
class DummyOpt:
    def __init__(self):
        self.weights = 'yolov10s.pt'
        self.source = 'lanechange.mp4'
        self.img_size = 640
        # 与命令行默认值保持一致（detect_3d_with_surface.py: --conf-thres 0.25 / --iou-thres 0.45）
        # 网页端不走 argparse，这里就是网页端实际生效的口径
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = ''
        self.view_img = False
        self.save_txt = False
        self.save_conf = False
        self.nosave = True
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.project = 'runs/detect'
        self.name = 'exp'
        self.exist_ok = False
        self.config_deepsort = 'deep_sort/configs/deep_sort.yaml'
        self.show_img = False
        self.save_img = False
        self.road_model_dir = 'code/models'
        self.road_conf_thres = 0.25
        self.max_frames = None
        self.save_jsonl = False
        self.structured_dir = 'structured'
        # 需求 5/6：固定帧率节拍 + 异步深度
        self.fps = 25.0
        self.no_pacing = False
        self.no_depth_async = False

import detect_3d_with_surface as detect_3d
detect_3d.opt = DummyOpt()


# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="智驭安 DriveSafe",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Session State Init
# ─────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'detection_done' not in st.session_state:
    st.session_state.detection_done = False
if 'stop_detection' not in st.session_state:
    st.session_state.stop_detection = False
if 'login_error' not in st.session_state:
    st.session_state.login_error = False
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0
if 'frames_processed' not in st.session_state:
    st.session_state.frames_processed = 0
# --- 新增：初始化主题模式 ---
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark' # 默认为暗色主题


# ─────────────────────────────────────────────
#  Global CSS — Tesla Dark Tech Theme
# ─────────────────────────────────────────────
# 深色主题 CSS
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:    #0a0c10;
    --bg-secondary:  #0d1117;
    --bg-card:       #111620;
    --bg-card2:      #141b26;
    --accent-blue:   #00bfff;
    --accent-red:    #e8303a;
    --accent-green:  #00e5a0;
    --accent-amber:  #ffb300;
    --border:        rgba(0,191,255,0.18);
    --border-hot:    rgba(0,191,255,0.55);
    --text-primary:  #e8eaf0;
    --text-muted:    #6b7a99;
    --glow-blue:     0 0 18px rgba(0,191,255,0.45);
    --glow-red:      0 0 18px rgba(232,48,58,0.55);
    --glow-green:    0 0 18px rgba(0,229,160,0.4);
}

/* ── App background ── */
.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,191,255,0.08) 0%, transparent 70%),
        linear-gradient(180deg, #0a0c10 0%, #080b0f 100%) !important;
}

/* ── Animated grid overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,191,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,191,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1018 0%, #090d14 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── All text ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Main headings ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.06em !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 0 20px rgba(0,191,255,0.06) !important;
    transition: box-shadow .3s, border-color .3s !important;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-hot) !important;
    box-shadow: var(--glow-blue) !important;
}
[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #00e5ff !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: var(--accent-blue) !important;
    border: 1px solid var(--accent-blue) !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 10px 22px !important;
    transition: all .3s ease !important;
}
.stButton > button:hover {
    background: var(--accent-blue) !important;
    color: #000 !important;
    box-shadow: var(--glow-blue) !important;
    transform: translateY(-1px) !important;
}

/* ── Primary action button (start) ── */
.btn-start > button {
    background: linear-gradient(135deg, #00bfff22, #00bfff11) !important;
    border: 1px solid var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    box-shadow: inset 0 0 20px rgba(0,191,255,0.05) !important;
}

/* ── Stop button ── */
.btn-stop > button {
    border-color: var(--accent-red) !important;
    color: var(--accent-red) !important;
}
.btn-stop > button:hover {
    background: var(--accent-red) !important;
    color: #fff !important;
    box-shadow: var(--glow-red) !important;
}

/* ── Video container styling ── */
.video-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 0 20px rgba(0,191,255,0.08);
    height: 100%;
}

.video-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #aabbcc;
    letter-spacing: 0.14em;
    margin-bottom: 10px;
    text-align: center;
    text-transform: uppercase;
}

[data-testid="stImage"] {
    width: 100% !important;
    min-height: 320px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 20px rgba(0,191,255,0.08) !important;
    overflow: hidden !important;
}

[data-testid="stImage"] > div,
[data-testid="stImage"] img {
    width: 100% !important;
}

[data-testid="stImage"] img {
    height: 320px !important;
    object-fit: fill !important;
    background: var(--bg-card) !important;
    display: block !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.dataframe {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-blue), #0080ff) !important;
    box-shadow: 0 0 8px var(--accent-blue) !important;
}
.stProgress > div {
    background: rgba(0,191,255,0.08) !important;
    border-radius: 4px !important;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 20px;
    border: 1px solid;
}
.badge-online  { color: var(--accent-green); border-color: var(--accent-green); background: rgba(0,229,160,0.08); }
.badge-warning { color: var(--accent-amber); border-color: var(--accent-amber); background: rgba(255,179,0,0.08); }
.badge-danger  { color: var(--accent-red);   border-color: var(--accent-red);   background: rgba(232,48,58,0.08); }

/* ── Risk level bar ── */
.risk-bar-wrap {
    background: rgba(0,191,255,0.06);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-green), var(--accent-amber), var(--accent-red));
}

/* ── Section header ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.95rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00d4ff;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 18px;
}
.compact-section-header {
    margin-bottom: 6px !important;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--accent-blue);
    text-align: center;
    letter-spacing: 0.06em;
    padding: 8px 0 16px 0;
    text-shadow: var(--glow-blue);
}
.sidebar-version {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.2em;
    margin-top: -12px;
    margin-bottom: 20px;
}

/* ── Page title area ── */
.page-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.45rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    margin: 0;
}
.page-title span { color: var(--accent-blue); }
.page-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.22em;
    color: #99aabb;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Login form ── */
.login-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 48px 56px;
    box-shadow: 0 0 60px rgba(0,191,255,0.08), 0 0 120px rgba(0,0,0,0.6);
    max-width: 460px;
    width: 100%;
    text-align: center;
}
.login-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    color: var(--accent-blue);
    text-shadow: var(--glow-blue);
    margin-bottom: 4px;
}
.login-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
    margin: 24px 0;
    opacity: 0.4;
}

/* ── Logout btn small ── */
.logout-btn > button {
    background: transparent !important;
    border: 1px solid var(--accent-red) !important;
    color: var(--accent-red) !important;
    font-size: 0.65rem !important;
    padding: 6px 14px !important;
}
.logout-btn > button:hover {
    background: var(--accent-red) !important;
    color: #fff !important;
}
/* ── FileUploader 全部强制白色 ── */
[data-testid="stFileUploader"] {
    color: #ffffff !important;
}

[data-testid="stFileUploader"] * {
    color: #ffffff !important;
}

/* 文件名 */
[data-testid="stFileUploader"] span {
    color: #ffffff !important;
}

/* 文件大小 */
[data-testid="stFileUploader"] small {
    color: #ffffff !important;
}

/* 防止被 markdown 或 label 覆盖 */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] p {
    color: #ffffff !important;
}
/* ── 文件名改成蓝色（只改这一行） ── */
[data-testid="stFileUploader"] span {
    color: #00e5ff !important;   /* 亮科技蓝 */
    font-weight: 600 !important;
    text-shadow: 0 0 6px rgba(0,191,255,0.6); /* 可选：发光 */
}

/* Hide Streamlit native chrome */
[data-testid="stAppDeployButton"],
.stAppDeployButton,
#MainMenu,
[data-testid="stMainMenu"],
footer {
    visibility: hidden !important;
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background: transparent !important;
    pointer-events: none !important;
    min-height: 0 !important;
}

[data-testid="stHeader"] {
    height: 0 !important;
}

[data-testid="stExpandSidebarButton"] {
    pointer-events: auto !important;
    visibility: visible !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
}
</style>
"""



# 浅色/白天主题 CSS
LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── Light Mode Root palette ── */
:root {
    --bg-primary:    #ffffff; /* 白色主背景 */
    --bg-secondary:  #f8f9fa; /* 次级背景 */
    --bg-card:       #ffffff; /* 卡片背景 */
    --bg-card2:      #f0f2f6; /* 卡片背景2 */
    --accent-blue:   #007bff; /* 主色调 */
    --accent-red:    #dc3545; /* 危险色 */
    --accent-green:  #28a745; /* 成功色 */
    --accent-amber:  #ffc107; /* 警告色 */
    --border:        rgba(0,0,0,0.1); /* 边框 */
    --border-hot:    rgba(0,123,255,0.5); /* 悬停边框 */
    --text-primary:  #212529; /* 主要文字 */
    --text-muted:    #6c757d; /* 次要文字 */
    --glow-blue:     0 0 8px rgba(0,123,255,0.4); /* 蓝色发光 */
    --glow-red:      0 0 8px rgba(220,53,69,0.5); /* 红色发光 */
    --glow-green:    0 0 8px rgba(40,167,69,0.4); /* 绿色发光 */
}

/* ── App background ── */
.stApp {
    background: var(--bg-primary) !important;
    background-image: none; /* 移除深色主题的渐变背景 */
}

/* ── Remove animated grid overlay ── */
.stApp::before {
    display: none; /* 隐藏深色主题的网格 */
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.05) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── All text ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Main headings ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.06em !important;
    color: var(--text-primary); /* 确保标题在浅色背景下可见 */
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
    transition: box-shadow .3s, border-color .3s !important;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-hot) !important;
    box-shadow: var(--glow-blue) !important;
}
[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--accent-blue) !important; /* 使用主色调 */
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--text-muted) !important; /* 使用次要文字颜色 */
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: var(--accent-blue) !important;
    border: 1px solid var(--accent-blue) !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 10px 22px !important;
    transition: all .3s ease !important;
}
.stButton > button:hover {
    background: var(--accent-blue) !important;
    color: #ffffff !important; /* 文字变为白色 */
    box-shadow: var(--glow-blue) !important;
    transform: translateY(-1px) !important;
}

/* ── Primary action button (start) ── */
.btn-start > button {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-blue) 100%) !important;
    border: 1px solid var(--accent-blue) !important;
    color: white !important; /* 白色文字 */
    box-shadow: inset 0 0 20px rgba(0,0,0,0.05) !important;
}
.btn-start > button:hover {
    background: linear-gradient(135deg, #0056b3 0%, #0056b3 100%) !important; /* 深一些的蓝色 */
    box-shadow: var(--glow-blue) !important;
}

/* ── Stop button ── */
.btn-stop > button {
    border-color: var(--accent-red) !important;
    color: var(--accent-red) !important;
}
.btn-stop > button:hover {
    background: var(--accent-red) !important;
    color: #fff !important;
    box-shadow: var(--glow-red) !important;
}

/* ── Video container styling ── */
.video-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    height: 100%;
}

.video-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-muted);
    letter-spacing: 0.14em;
    margin-bottom: 10px;
    text-align: center;
    text-transform: uppercase;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.dataframe {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-blue), #0056b3) !important;
    box-shadow: 0 0 8px rgba(0,123,255,0.2) !important;
}
.stProgress > div {
    background: rgba(0,123,255,0.1) !important;
    border-radius: 4px !important;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 20px;
    border: 1px solid;
}
.badge-online  { color: var(--accent-green); border-color: var(--accent-green); background: rgba(40,167,69,0.1); }
.badge-warning { color: var(--accent-amber); border-color: var(--accent-amber); background: rgba(255,193,7,0.1); }
.badge-danger  { color: var(--accent-red);   border-color: var(--accent-red);   background: rgba(220,53,69,0.1); }

/* ── Risk level bar ── */
.risk-bar-wrap {
    background: rgba(0,0,0,0.05);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-green), var(--accent-amber), var(--accent-red));
}

/* ── Section header ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.95rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-blue);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 18px;
}
.compact-section-header {
    margin-bottom: 6px !important;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--accent-blue);
    text-align: center;
    letter-spacing: 0.06em;
    padding: 8px 0 16px 0;
    text-shadow: none; /* 移除深色主题的发光效果 */
}
.sidebar-version {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.2em;
    margin-top: -12px;
    margin-bottom: 20px;
}

/* ── Page title area ── */
.page-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.45rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    margin: 0;
}
.page-title span { color: var(--accent-blue); }
.page-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.22em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Login form ── */
.login-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 48px 56px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1), 0 4px 40px rgba(0,0,0,0.05);
    max-width: 460px;
    width: 100%;
    text-align: center;
}
.login-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    color: var(--accent-blue);
    text-shadow: none; /* 移除发光效果 */
    margin-bottom: 4px;
}
.login-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
    margin: 24px 0;
    opacity: 0.4;
}

/* ── Logout btn small ── */
.logout-btn > button {
    background: transparent !important;
    border: 1px solid var(--accent-red) !important;
    color: var(--accent-red) !important;
    font-size: 0.65rem !important;
    padding: 6px 14px !important;
}
.logout-btn > button:hover {
    background: var(--accent-red) !important;
    color: #fff !important;
}
/* ── FileUploader 文字颜色调整 ── */
[data-testid="stFileUploader"] {
    color: var(--text-primary) !important;
}
[data-testid="stFileUploader"] * {
    color: var(--text-primary) !important;
}
[data-testid="stFileUploader"] span {
    color: var(--accent-blue) !important;   /* 文件名改为蓝色 */
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] small {
    color: var(--text-muted) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] p {
    color: var(--text-primary) !important;
}

/* Hide Streamlit native chrome */
[data-testid="stAppDeployButton"],
.stAppDeployButton,
#MainMenu,
[data-testid="stMainMenu"],
footer {
    visibility: hidden !important;
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background: transparent !important;
    pointer-events: none !important;
    min-height: 0 !important;
}

[data-testid="stHeader"] {
    height: 0 !important;
}

[data-testid="stExpandSidebarButton"] {
    pointer-events: auto !important;
    visibility: visible !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
}
</style>
"""

# 根据当前主题模式返回对应的CSS
def get_global_css():
    if st.session_state.theme_mode == 'light':
        return LIGHT_CSS
    else:
        return DARK_CSS

# 应用当前CSS
st.markdown(get_global_css(), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────
def render_login():
    user_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user")

    def is_valid_username(name):
        if not name or ".." in name:
            return False
        invalid_chars = '<>:"/\\|?*'
        return not any(ch in name for ch in invalid_chars)

    def user_file_path(name):
        return os.path.join(user_dir, f"{name}.txt")

    def check_user_password(name, pwd):
        if name == "123" and pwd == "123":
            return True
        if not is_valid_username(name):
            return False

        path = user_file_path(name)
        if not os.path.exists(path):
            return False

        try:
            with open(path, "r") as f:
                saved_pwd = f.read().strip()
        except OSError:
            return False
        return saved_pwd == pwd

    # Center using columns
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown("""
        <div style="height:60px"></div>
        <div class="login-card">
            <div class="login-logo">&#9651; DriveSafe</div>
            <div style="font-family:'Orbitron',monospace;font-size:0.78rem;
                        letter-spacing:0.18em;color:#6b7a99;margin-bottom:2px;">
                智驭安DriveSafe智能驾驶风险预警系统
            </div>
            <div class="login-divider"></div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                        letter-spacing:0.2em;color:#00bfff;text-align:center;
                        margin-bottom:6px;">用户认证</div>
        </div>
        """, unsafe_allow_html=True)

        login_tab, register_tab = st.tabs(["登录", "注册"])

        with login_tab:
            username = st.text_input("账号", placeholder="输入账号", key="login_user",
                                     label_visibility="collapsed")

            password = st.text_input("密码", placeholder="输入密码", type="password",
                                     key="login_pass", label_visibility="collapsed")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("登录", use_container_width=True):
                if check_user_password(username.strip(), password):
                    st.session_state.logged_in = True
                    st.session_state.login_error = False
                    st.rerun()
                else:
                    st.session_state.login_error = True
                    st.rerun()

            if st.session_state.login_error:
                st.markdown("""
                <div style="margin-top:12px;font-family:'Share Tech Mono',monospace;
                            font-size:0.75rem;color:#e8303a;letter-spacing:0.1em;
                            text-align:center;">
                    访问被拒绝 — 凭证无效
                </div>
                """, unsafe_allow_html=True)

        with register_tab:
            reg_username = st.text_input("注册账号", placeholder="输入新账号", key="register_user",
                                         label_visibility="collapsed")
            reg_password = st.text_input("注册密码", placeholder="输入密码", type="password",
                                         key="register_pass", label_visibility="collapsed")
            reg_confirm = st.text_input("确认密码", placeholder="再次输入密码", type="password",
                                        key="register_confirm", label_visibility="collapsed")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("注册", use_container_width=True):
                account = reg_username.strip()
                if not account or not reg_password:
                    st.error("账号和密码不能为空")
                elif not is_valid_username(account):
                    st.error("账号不能包含 /、\\、.. 或 Windows 文件名非法字符")
                elif reg_password != reg_confirm:
                    st.error("两次输入的密码不一致")
                else:
                    os.makedirs(user_dir, exist_ok=True)
                    path = user_file_path(account)
                    if os.path.exists(path):
                        st.error("账号已存在，请直接登录")
                    else:
                        try:
                            with open(path, "w") as f:
                                f.write(reg_password)
                            st.session_state.login_error = False
                            st.success("注册成功，请切换到登录")
                        except OSError as exc:
                            st.error(f"注册失败：{exc}")

        st.markdown("""
        <div style="margin-top:28px;font-family:'Share Tech Mono',monospace;
                    font-size:0.62rem;color:#3a4560;letter-spacing:0.16em;
                    text-align:center;border-top:1px solid rgba(0,191,255,0.1);
                    padding-top:16px;">
            DRIVESAFE v2.1.0 &nbsp;|&nbsp; YOLOV10 + DEEPSORT &nbsp;|&nbsp; 3D风险引擎
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">&#9651; DRIVESAFE</div>
        <div class="sidebar-version">智能驾驶风险预警系统 </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # --- 新增：主题切换按钮 ---
        if st.button("🔄 切换主题"):
            if st.session_state.theme_mode == 'dark':
                st.session_state.theme_mode = 'light'
            else:
                st.session_state.theme_mode = 'dark'
            st.rerun() # 重新运行以应用新主题

        # System status
        st.markdown("""
        <div class="section-header">系统状态</div>
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 4px;font-family:'Share Tech Mono',monospace;
                    font-size:0.78rem;color:#6b7a99;">
            <span>AI引擎</span>
            <span class="status-badge badge-online">在线</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 4px;font-family:'Share Tech Mono',monospace;
                    font-size:0.78rem;color:#6b7a99;">
            <span>GPU加速</span>
            <span class="status-badge badge-warning">待命</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 4px;font-family:'Share Tech Mono',monospace;
                    font-size:0.78rem;color:#6b7a99;">
            <span>深度排序</span>
            <span class="status-badge badge-online">就绪</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-header">视频源</div>', unsafe_allow_html=True)
        source_option = st.radio("", ["摄像头", "视频文件"], label_visibility="collapsed")

        source = "lanechange.mp4"
        if source_option == "视频文件":
            video_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
            if video_file:
                with open(video_file.name, "wb") as f:
                    f.write(video_file.getbuffer())
                source = video_file.name
        else:
            cam_id = st.text_input("摄像头编号", value="0")
            source = cam_id

        st.markdown("---")
        st.markdown('<div class="section-header">检测参数</div>', unsafe_allow_html=True)

        with st.expander("高级参数"):
            # 默认值取自 DummyOpt（单一来源），避免与命令行默认值再次漂移
            conf_val = st.slider("置信度阈值", 0.01, 1.0, float(detect_3d.opt.conf_thres), 0.01)
            iou_val = st.slider("IOU 阈值", 0.01, 1.0, float(detect_3d.opt.iou_thres), 0.01)
            detect_3d.opt.conf_thres = conf_val
            detect_3d.opt.iou_thres = iou_val

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="btn-start">', unsafe_allow_html=True)
            start_btn = st.button("开始", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="btn-stop">', unsafe_allow_html=True)
            stop_btn = st.button("停止", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Logout
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.detection_done = False
            st.session_state.stop_detection = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:24px;font-family:'Share Tech Mono',monospace;
                    font-size:0.58rem;color:#2a3448;letter-spacing:0.14em;
                    text-align:center;">
            (C) 2026 智驭安智能科技
        </div>
        """, unsafe_allow_html=True)

    return source_option, source, start_btn, stop_btn


def build_media_placeholder_frame(accent_rgb):
    frame = np.full((360, 640, 3), (17, 22, 32), dtype=np.uint8)
    grid_color = np.array((20, 31, 43), dtype=np.uint8)
    accent = np.array(accent_rgb, dtype=np.uint8)

    frame[::40, :, :] = grid_color
    frame[:, ::40, :] = grid_color
    frame[0:2, :, :] = accent
    frame[-2:, :, :] = accent
    frame[:, 0:2, :] = accent
    frame[:, -2:, :] = accent

    cx, cy = 320, 180
    frame[cy - 2:cy + 2, cx - 54:cx + 54, :] = accent
    frame[cy - 36:cy + 36, cx - 2:cx + 2, :] = accent
    frame[cy - 26:cy + 26, cx - 42:cx + 42, :] = (
        frame[cy - 26:cy + 26, cx - 42:cx + 42, :] // 2 + accent // 5
    )
    return frame


VIDEO_PLACEHOLDER_FRAME = build_media_placeholder_frame((0, 191, 255))
RISK_PLACEHOLDER_FRAME = build_media_placeholder_frame((232, 48, 58))


# ─────────────────────────────────────────────
#  METRIC CARDS ROW
# ─────────────────────────────────────────────
def render_metrics(placeholder=None):
    from data_store import data_store

    stats = data_store.get_stats()

    track_count = str(stats['track_count'])
    alert_count = str(stats['total_alerts'])

    html = """
    <div style="display: flex; gap: 6px;">
        <div style="flex: 1; background: var(--bg-card); border: 1px solid var(--border); 
                    border-radius: 12px; padding: 8px 10px; text-align: center;
                    box-shadow: 0 0 20px rgba(0,191,255,0.06);">
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.44rem; 
                        color: #00e5ff; letter-spacing: 0.1em; text-transform: uppercase;
                        margin-bottom: 3px;">目标追踪数</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 0.96rem; 
                        font-weight: 800; color: #e8eaf0;">""" + track_count + """</div>
        </div>
        <div style="flex: 1; background: var(--bg-card); border: 1px solid var(--border); 
                    border-radius: 12px; padding: 8px 10px; text-align: center;
                    box-shadow: 0 0 20px rgba(0,191,255,0.06);">
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.44rem; 
                        color: #00e5ff; letter-spacing: 0.1em; text-transform: uppercase;
                        margin-bottom: 3px;">告警次数</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 0.96rem; 
                        font-weight: 800; color: #e8eaf0;">""" + alert_count + """</div>
        </div>
    </div>
    """

    if placeholder is None:
        st.markdown(html, unsafe_allow_html=True)
    else:
        with placeholder:
            st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 风险概览组件
# ═══════════════════════════════════════════════════════════════
def render_risk_overview(placeholder=None):
    from data_store import data_store

    stats = data_store.get_stats()
    current_frame = data_store.get_current_frame()

    if current_frame:
        risk_index = current_frame.max_risk_score
        avg_risk = current_frame.avg_risk_score
    else:
        risk_index = 0.0
        avg_risk = 0.0

    # 阈值统一来自 risk_field.RISK_THRESHOLDS（SCF 尺度），与画面 decision_status 同源
    risk_progress = min(max(0.0, risk_index / RISK_THRESHOLDS["HIGH"] * 100), 100)

    _risk_label = risk_level(risk_index)
    if _risk_label == "HIGH":
        risk_text = "高风险"
        risk_color = "#e8303a"
    elif _risk_label == "MEDIUM":
        risk_text = "中风险"
        risk_color = "#ffb300"
    elif _risk_label == "LOW":
        risk_text = "低风险"
        risk_color = "#ffd54f"
    else:
        risk_text = "安全"
        risk_color = "#00e5a0"

    risk_html = f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border); 
                border-radius: 12px; padding: 8px; text-align: center; height: 100%;">
        <div style="font-family: 'Orbitron', monospace; font-size: 1.16rem; 
                    font-weight: 900; color: {risk_color};">
            {risk_index:.3f}
        </div>
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.44rem; 
                    color: #aaddff; margin-top: 2px;">
            风险指数
        </div>
        <div style="margin-top: 6px; padding: 0 16px;">
            <div class="risk-bar-wrap">
                <div class="risk-bar-fill" style="width: {risk_progress}%;"></div>
            </div>
        </div>
        <div style="margin-top: 5px; font-family: 'Orbitron', monospace; 
                    font-size: 0.56rem; color: {risk_color};">
            {risk_text}
        </div>
        <div style="margin-top: 6px; font-family: 'Share Tech Mono', monospace; 
                    font-size: 0.44rem; color: #aabbcc;">
            跟踪目标数: {stats['track_count']}<br>
            平均风险: {avg_risk:.4f}
        </div>
    </div>
    """

    if placeholder is None:
        st.markdown(risk_html, unsafe_allow_html=True)
    else:
        with placeholder:
            st.markdown(risk_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  RISK TREND CHART
# ─────────────────────────────────────────────
def render_risk_trend(placeholder=None):
    from data_store import data_store
    import altair as alt
    import pandas as pd

    def _render():
        trend = data_store.get_risk_trend()

        if not trend:
            st.markdown(
                '<div style="background:var(--bg-card);border:1px solid var(--border);'
                'border-radius:12px;height:168px;display:flex;align-items:center;'
                'justify-content:center;text-align:center;">'
                '<span style="font-family:Share Tech Mono,monospace;color:#6b7a99;">'
                '等待风险趋势数据...</span></div>',
                unsafe_allow_html=True
            )
            return

        df = pd.DataFrame(trend).sort_values("frame_idx")

        base = alt.Chart(df).encode(
            x=alt.X(
                "frame_idx:Q",
                title="帧序号",
                axis=alt.Axis(format="d", tickMinStep=1)
            ),
            y=alt.Y(
                "risk_score:Q",
                title="风险大小",
                scale=alt.Scale(zero=False, nice=True)
            ),
            tooltip=[
                alt.Tooltip("frame_idx:Q", title="帧序号", format="d"),
                alt.Tooltip("risk_score:Q", title="整帧风险", format=".3f")
            ]
        )

        line = base.mark_line(color="#00bfff", strokeWidth=2, opacity=0.9)
        points = base.mark_circle(
            size=46,
            color="#ffffff",
            stroke="#ffffff",
            strokeWidth=1.2,
            opacity=1.0
        )

        chart = (line + points).properties(height=154).configure_axis(
            labelColor="#aabbcc",
            titleColor="#aaddff",
            gridColor="#1b3444",
            domainColor="#26495f",
            tickColor="#26495f",
            labelFont="Share Tech Mono",
            titleFont="Share Tech Mono"
        ).configure_view(
            stroke=None
        ).configure(
            background="transparent"
        )

        st.altair_chart(chart, use_container_width=True)

    if placeholder is None:
        _render()
    else:
        with placeholder:
            _render()


# ─────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────
def render_dashboard():
    source_option, source, start_btn, stop_btn = render_sidebar()

    # ── Page header ──
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown("""
        <div class="page-title">&#9651; DRIVE<span>SAFE</span></div>
        <div class="page-subtitle">智能驾驶风险预警系统</div>
        """, unsafe_allow_html=True)
    with h_col2:
        now = time.strftime("%Y-%m-%d  %H:%M:%S")
        st.markdown(f"""
        <div style="text-align:right;font-family:'Share Tech Mono',monospace;
                    font-size:0.72rem;color:#6b7a99;letter-spacing:0.1em;
                    padding-top:6px;">
            {now}<br>
            <span style="color:#00e5a0;">&#9679;</span> 系统运行中
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # 1. 顶部信息区：左侧检测统计 + 右侧风险概览
    # ═══════════════════════════════════════════════════════════════
    top_left_col, top_right_col = st.columns([1, 1], gap="medium")

    with top_left_col:
        with st.container(height=224, border=False):
            st.markdown('<div class="section-header compact-section-header">检测数据统计</div>', unsafe_allow_html=True)
            metrics_placeholder = st.empty()
            with metrics_placeholder:
                render_metrics()

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            st.markdown('<div class="section-header compact-section-header">风险概览</div>', unsafe_allow_html=True)
            risk_overview_placeholder = st.empty()
            with risk_overview_placeholder:
                render_risk_overview()

    with top_right_col:
        with st.container(height=224, border=False):
            st.markdown('<div class="section-header">实时风险趋势折线图</div>', unsafe_allow_html=True)
            risk_trend_placeholder = st.empty()
            with risk_trend_placeholder:
                render_risk_trend()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # 2. 中部主显示区：左侧实时画面 + 右侧热力图
    # ═══════════════════════════════════════════════════════════════
    bottom_left_col, bottom_right_col = st.columns([1, 1], gap="medium")

    with bottom_left_col:
        with st.container(height=448, border=False):
            st.markdown('<div class="section-header">实时画面</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="video-container">
                <div class="video-header">📹 摄像头画面</div>
            </div>
            """, unsafe_allow_html=True)
            video_placeholder = st.empty()
            video_placeholder.markdown(
                '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;'
                'height:320px;display:flex;align-items:center;justify-content:center;text-align:center;">'
                '<span style="font-family:Share Tech Mono,monospace;color:#6b7a99;">等待视频源...</span>'
                '</div>',
                unsafe_allow_html=True
            )

    with bottom_right_col:
        with st.container(height=448, border=False):
            st.markdown('<div class="section-header">风险热力图</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="video-container">
                <div class="video-header">🔥 风险热力图</div>
            </div>
            """, unsafe_allow_html=True)
            risk_placeholder = st.empty()
            risk_placeholder.markdown(
                '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;'
                'height:320px;display:flex;align-items:center;justify-content:center;text-align:center;">'
                '<span style="font-family:Share Tech Mono,monospace;color:#99aabb;">等待风险数据...</span>'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # 3. 底部详情区：当前检测目标实时数据
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">当前检测目标</div>', unsafe_allow_html=True)
    realtime_placeholder = st.empty()
    with realtime_placeholder:
        render_realtime_panel()

    # ── Footer ──
    st.markdown("""
    <div style="margin-top:32px;padding-top:16px;border-top:1px solid rgba(0,191,255,0.1);
                font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                color:#667788;letter-spacing:0.16em;text-align:center;">
        DRIVESAFE 智能驾驶风险预警系统 &nbsp;|&nbsp; YOLOV10S + DEEPSORT + 3D风险引擎
        &nbsp;|&nbsp; (C) 2026 智驭安智能科技
    </div>
    """, unsafe_allow_html=True)

    # ─── DETECTION LOGIC ───
    if stop_btn:
        st.session_state.stop_detection = True
        st.session_state.detection_done = True
        st.rerun()

    if start_btn and not st.session_state.detection_done:
        st.session_state.detection_done = True
        st.session_state.stop_detection = False
        st.info("[ 系统 ] 检测已启动 — 正在处理帧流...")

        detect_3d.opt.source = source
        detect_3d.opt.view_img = False

        from data_store import data_store
        data_store.reset()

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        def callback(im0, risk_img, frame_idx=None, total_frames=None, risk_sources=None, frame_risk=None):
            import streamlit as st
            from data_store import data_store

            if st.session_state.get('stop_detection', False):
                return False

            if im0 is not None:
                im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                video_placeholder.image(im0_rgb, channels="RGB", use_container_width=True)
                if frame_idx is not None:
                    st.session_state.frames_processed = frame_idx + 1

            if risk_img is not None:
                risk_rgb = cv2.cvtColor(risk_img, cv2.COLOR_BGR2RGB)
                risk_rgb = cv2.resize(risk_rgb, (640, 320), interpolation=cv2.INTER_LINEAR)
                risk_placeholder.image(risk_rgb, channels="RGB", use_container_width=True)

            if risk_sources is not None:
                alert_triggered = any(src.get('scf', 0) > 510 for src in risk_sources)

                data_store.update_frame(
                    frame_idx if frame_idx is not None else 0,
                    risk_sources,
                    alert_triggered,
                    frame_risk_score=frame_risk
                )

                # 更新所有占位符
                with metrics_placeholder:
                    render_metrics()
                with risk_overview_placeholder:
                    render_risk_overview()
                with risk_trend_placeholder:
                    render_risk_trend()
                with realtime_placeholder:
                    render_realtime_panel()

            if frame_idx is not None and total_frames is not None:
                progress = (frame_idx + 1) / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.markdown(
                    f"<span style='font-family:Share Tech Mono,monospace;"
                    f"font-size:0.72rem;color:#00bfff;'>"
                    f"处理中: {frame_idx + 1} / {total_frames}</span>",
                    unsafe_allow_html=True
                )
            return True

        try:
            if source_option != "摄像头":
                cap = cv2.VideoCapture(source)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if total_frames <= 0:
                    total_frames = None
            else:
                total_frames = None

            detect_3d.detect(save_img=False, callback=callback)

            if not st.session_state.get('stop_detection', False):
                st.success("[ 完成 ] 检测完成！")
            else:
                st.warning("[ 停止 ] 检测已停止")

        except Exception as e:
            st.error(f"[ 错误 ] 识别出错: {e}")
        finally:
            st.session_state.detection_done = False
            st.session_state.stop_detection = False
            progress_bar.empty()
            status_text.empty()

    if st.session_state.detection_done and not st.session_state.get('stop_detection', False):
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        if st.sidebar.button("重新开始"):
            st.session_state.detection_done = False
            st.session_state.stop_detection = False
            st.rerun()
    else:
        if not start_btn:
            st.info("[ 就绪 ] 点击左侧「开始」按钮启动检测")


def render_realtime_panel():
    from data_store import data_store

    st.markdown("---")
    st.markdown('<div class="section-header">当前检测目标</div>', unsafe_allow_html=True)

    current_frame = data_store.get_current_frame()
    current_objects = data_store.get_current_objects()

    if current_objects:
        html = ""
        for obj in current_objects:
            risk_color = "#e8303a" if obj.risk_level == "高风险" else "#ffb300" if obj.risk_level == "中风险" else "#00e5a0"

            html += f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border); 
                        border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'Share Tech Mono', monospace; color: #00bfff; font-size: 1.1rem; font-weight: bold;">
                        ID: {obj.track_id} | {obj.class_name}
                    </span>
                    <span style="font-family: 'Orbitron', monospace; font-size: 0.8rem; 
                                color: {risk_color}; border: 1px solid {risk_color}; 
                                padding: 4px 12px; border-radius: 4px;">
                        {obj.risk_level}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 12px; 
                            font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; color: #6b7a99;">
                    <span style="flex: 1;">距离: {obj.distance:.2f}m</span>
                    <span style="flex: 1;">速度: {obj.speed:.1f} km/h</span>
                    <span style="flex: 1;">风险类型: {obj.risk_type}</span>
                    <span style="flex: 1;">风险值: {obj.risk_score:.4f}</span>
                </div>
            </div>
            """
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border); 
                    border-radius: 12px; padding: 40px; text-align: center;">
            <span style="font-family: 'Share Tech Mono', monospace; color: #6b7a99;">
                等待检测数据...
            </span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────
if not st.session_state.logged_in:
    render_login()
else:
    render_dashboard()
