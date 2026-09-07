import os
import sys
import time
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextBrowser, QFileDialog, QDialog, QLineEdit, QGridLayout, QMessageBox)

# 注入opt 对象
import argparse
class DummyOpt:
    def __init__(self):
        self.weights = 'yolov10s.pt'
        self.source = 'lanechange.mp4'
        self.img_size = 640
        self.conf_thres = 0.01
        self.iou_thres = 0.01
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
        # 运行时负载调度参数（auto/always/off）
        self.target_fps = None
        self.depth_strategy = 'auto'
        self.road_strategy = 'auto'
        self.reid_strategy = 'auto'
        self.bev_strategy = 'auto'

import detect_3d
# 注入 opt
detect_3d.opt = DummyOpt()

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
        
    def flush(self):
        pass

class DetectThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True

    def run(self):
        detect_3d.opt.source = self.source
        detect_3d.opt.view_img = False

        # 调用 detect 并传入 callback，同时设置 save_img=False 以提高性能
        try:
            detect_3d.detect(save_img=False, callback=self.frame_callback)
        except Exception as e:
            print(f"Detection error: {e}")
        self.finished_signal.emit()

    def frame_callback(self, im0, risk_img):
        if not self.running:
            raise InterruptedError("Stopped by user")
        if risk_img is None:
            risk_img = np.zeros_like(im0) # fallback
        self.change_pixmap_signal.emit(im0, risk_img)

    def stop(self):
        self.running = False


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('欢迎登录')
        self.resize(400, 300)
        self.setFixedSize(self.width(), self.height())
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("驭安drivesafe")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        layout.addWidget(title)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("请输入账号")
        self.user_input.setFont(QFont("Microsoft YaHei", 12))
        self.user_input.setMinimumHeight(40)
        layout.addWidget(self.user_input)

        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("请输入密码")
        self.pwd_input.setEchoMode(QLineEdit.Password)
        self.pwd_input.setFont(QFont("Microsoft YaHei", 12))
        self.pwd_input.setMinimumHeight(40)
        layout.addWidget(self.pwd_input)

        btn_layout = QHBoxLayout()
        self.btn_login = QPushButton("登录")
        self.btn_login.setMinimumHeight(40)
        self.btn_login.setFont(QFont("Microsoft YaHei", 12))
        
        self.btn_register = QPushButton("注册")
        self.btn_register.setMinimumHeight(40)
        self.btn_register.setFont(QFont("Microsoft YaHei", 12))
        
        btn_layout.addWidget(self.btn_login)
        btn_layout.addWidget(self.btn_register)
        layout.addLayout(btn_layout)

        self.btn_login.clicked.connect(self.login)
        self.btn_register.clicked.connect(self.register)

        self.user_dir = './user'
        if not os.path.exists(self.user_dir):
            os.makedirs(self.user_dir)

    def register(self):
        user = self.user_input.text().strip()
        pwd = self.pwd_input.text().strip()
        if not user or not pwd:
            QMessageBox.warning(self, "错误", "账号或密码不能为空！")
            return
        path = os.path.join(self.user_dir, f"{user}.txt")
        if os.path.exists(path):
            QMessageBox.warning(self, "错误", "账号已存在！")
            return
        with open(path, "w") as f:
            f.write(pwd)
        QMessageBox.information(self, "成功", "注册成功，请登录！")

    def login(self):
        user = self.user_input.text().strip()
        pwd = self.pwd_input.text().strip()
        if not user or not pwd:
            QMessageBox.warning(self, "错误", "账号或密码不能为空！")
            return
        path = os.path.join(self.user_dir, f"{user}.txt")
        if not os.path.exists(path):
            QMessageBox.warning(self, "错误", "账号不存在！")
            return
        with open(path, "r") as f:
            saved_pwd = f.read().strip()
            if saved_pwd == pwd:
                self.accept()
            else:
                QMessageBox.warning(self, "错误", "密码错误！")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("驭安 drivesafe")
        self.resize(1280, 800)

        # Main Widget and Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Title Label
        self.title_label = QLabel("驭安 drivesafe")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        self.title_label.setStyleSheet("color: #333; margin: 10px;")
        self.main_layout.addWidget(self.title_label)

        # Video & Heatmap Area
        self.video_layout = QHBoxLayout()
        self.main_layout.addLayout(self.video_layout, stretch=5)

        self.video_label = QLabel("实时画面\n\n请点击以添加视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #f0f0f0; border: 2px solid #ccc; font-size: 20px; color: red;")
        self.video_layout.addWidget(self.video_label, stretch=2)

        self.heatmap_label = QLabel("热力图画面\n\n生成动态图像")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setStyleSheet("background-color: #f0f0f0; border: 2px solid #ccc; font-size: 20px; color: red;")
        self.video_layout.addWidget(self.heatmap_label, stretch=1)

        # Bottom Area
        self.bottom_layout = QHBoxLayout()
        self.main_layout.addLayout(self.bottom_layout, stretch=2)

        # Log TextBrowser
        self.text_browser = QTextBrowser()
        self.text_browser.setFont(QFont("Microsoft YaHei", 10))
        self.text_browser.setStyleSheet("border: 2px solid #ccc; padding: 5px;")
        self.text_browser.append("欢迎用户使用驭安drivesafe！！！")
        self.bottom_layout.addWidget(self.text_browser, stretch=3)

        # Buttons Area
        self.btn_layout = QGridLayout()
        self.bottom_layout.addLayout(self.btn_layout, stretch=1)

        self.btn_select_file = QPushButton("选择文件")
        self.btn_start = QPushButton("开始识别")
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_exit = QPushButton("退出系统")

        btn_style = "QPushButton { background-color: #358eff; color: white; border-radius: 10px; font-size: 18px; padding: 15px; } QPushButton:hover { background-color: #1e70d6; }"
        for btn in [self.btn_select_file, self.btn_start, self.btn_camera, self.btn_exit]:
            btn.setStyleSheet(btn_style)

        self.btn_layout.addWidget(self.btn_select_file, 0, 0)
        self.btn_layout.addWidget(self.btn_start, 0, 1)
        self.btn_layout.addWidget(self.btn_camera, 1, 0)
        self.btn_layout.addWidget(self.btn_exit, 1, 1)

        # Connections
        self.btn_select_file.clicked.connect(self.select_file)
        self.btn_start.clicked.connect(self.start_recognition)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_exit.clicked.connect(self.close)

        # Redirect stdout
        sys.stdout = EmittingStream(textWritten=self.normal_output_written)
        sys.stderr = EmittingStream(textWritten=self.normal_output_written)

        self.current_source = None
        self.thread = None

    def normal_output_written(self, text):
        cursor = self.text_browser.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.text_browser.setTextCursor(cursor)
        self.text_browser.ensureCursorVisible()

    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if fname:
            self.current_source = os.path.normpath(fname)
            print(f"当前选择的文件路径是：{self.current_source}")

    def open_camera(self):
        self.current_source = "0"
        print("已选择打开摄像头 (设备0)")

    def start_recognition(self):
        if not self.current_source:
            print("请先选择视频文件或打开摄像头！")
            return
            
        if self.thread is not None and self.thread.isRunning():
            print("识别正在进行中...")
            return

        print(f"开始识别: {self.current_source}")
        self.thread = DetectThread(self.current_source)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.finished_signal.connect(self.recognition_finished)
        self.thread.start()

    def recognition_finished(self):
        print("识别结束")

    def update_image(self, im0, risk_img):
        if im0 is None or risk_img is None or im0.size == 0 or risk_img.size == 0:
            return
            
        # Convert im0 to QPixmap
        if len(im0.shape) == 3:
            h, w, ch = im0.shape
            bytes_per_line = ch * w
            im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            qimg1 = QImage(im0_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap1 = QPixmap.fromImage(qimg1)
            self.video_label.setPixmap(pixmap1.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

        # Convert risk_img to QPixmap
        if len(risk_img.shape) == 3:
            h2, w2, ch2 = risk_img.shape
            bytes_per_line2 = ch2 * w2
            # Handle RGBA just in case
            if ch2 == 4:
                risk_img_rgb = cv2.cvtColor(risk_img, cv2.COLOR_BGRA2RGBA)
                fmt = QImage.Format_RGBA8888
            else:
                risk_img_rgb = cv2.cvtColor(risk_img, cv2.COLOR_BGR2RGB)
                fmt = QImage.Format_RGB888
            qimg2 = QImage(risk_img_rgb.data, w2, h2, bytes_per_line2, fmt)
            pixmap2 = QPixmap.fromImage(qimg2)
            self.heatmap_label.setPixmap(pixmap2.scaled(self.heatmap_label.width(), self.heatmap_label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    login = LoginDialog()
    if login.exec_() == QDialog.Accepted:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
