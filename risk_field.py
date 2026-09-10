import numpy as np
import cv2

class RiskFieldEngine:
    def __init__(self, width_meter=16, depth_meter=25, backward_meter=10, grid_res=0.1):
        """
        Initialize the Risk Field Engine.
        :param width_meter: Width of the field in meters (X-axis)
        :param depth_meter: Depth of the field in meters (Z-axis forward)
        :param backward_meter: Depth of the field in meters (Z-axis backward)
        :param grid_res: Grid resolution in meters
        """
        self.width_meter = width_meter
        self.depth_meter = depth_meter
        self.backward_meter = backward_meter
        self.grid_res = grid_res
        
        # Grid dimensions
        self.grid_w = int(width_meter / grid_res)
        self.grid_h = int((depth_meter + backward_meter) / grid_res)
        
        # Create meshgrid for vectorized calculations
        # X: -width/2 to width/2
        # Z: -backward to depth
        # Note: We use meshgrid 'xy' indexing by default in numpy, but for image we want (row, col)
        # Here we align with image coordinates: row -> Z (up/down), col -> X (left/right)
        
        # X axis (columns): -width/2 ... width/2
        x = np.linspace(-width_meter/2, width_meter/2, self.grid_w)
        # Z axis (rows): -backward ... depth
        z = np.linspace(-backward_meter, depth_meter, self.grid_h)
        
        self.X, self.Z = np.meshgrid(x, z)
        
    def get_gaussian_field(self, x_c, z_c, v_x=0, v_z=0, sigma_x=0.8, sigma_z=2.0, v_stretch_factor=0.2, weather_factor=1.0):
        """
        Generate a Gaussian risk field for a vehicle.
        :param x_c, z_c: Center position
        :param v_x, v_z: Velocity vector
        :param sigma_x, sigma_z: Base standard deviations
        :param v_stretch_factor: How much velocity stretches the field
        :param weather_factor: Environmental risk multiplier (e.g., 1.2 for rain)
        :return: Grid of risk values
        """
        # Apply weather factor to base sigma
        eff_sigma_x = sigma_x * weather_factor
        eff_sigma_z_base = sigma_z * weather_factor
        
        # Calculate velocity magnitude
        v_mag = np.sqrt(v_x**2 + v_z**2)
        
        # Stretch sigma_z based on velocity (Driving Risk Field concept)
        eff_sigma_z = eff_sigma_z_base + v_mag * v_stretch_factor
        
        # Rotation angle from Z-axis (which is up in our plot)
        # v_z is forward, v_x is lateral
        angle = np.arctan2(v_x, v_z) 
        
        # Rotate covariance matrix
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        
        # Covariance in local frame (aligned with velocity)
        # We assume X is lateral, Z is longitudinal
        Sigma_local = np.array([[eff_sigma_x**2, 0], [0, eff_sigma_z**2]])
        
        # Sigma_global = R @ Sigma_local @ R.T
        # Note: This standard rotation formula applies if X, Z are standard Cartesian.
        Sigma_global = R @ Sigma_local @ R.T
        
        # Inverse of covariance matrix
        try:
            Sigma_inv = np.linalg.inv(Sigma_global)
        except:
            Sigma_inv = np.eye(2) * (1.0 / (eff_sigma_x * eff_sigma_z))
            
        # Vectorized Mahalanobis distance
        # Stack points: (N, 2)
        # We flatten X and Z to make a list of points
        points = np.stack([self.X.ravel() - x_c, self.Z.ravel() - z_c], axis=1)
        
        # Calculate (x-mu)^T * Sigma^-1 * (x-mu)
        # Term 1: points @ Sigma_inv -> (N, 2)
        term1 = points @ Sigma_inv
        # Term 2: sum(term1 * points, axis=1) -> (N,)
        mahal_sq = np.sum(term1 * points, axis=1)
        
        # Gaussian: exp(-0.5 * mahal_sq)
        grid_vals = np.exp(-0.5 * mahal_sq)
        
        # Reshape back to grid
        return grid_vals.reshape(self.grid_h, self.grid_w)

    def calculate_scf(self, ego_pos, target_pos, v_ego, v_target, weather_factor=1.0):
        """
        Calculate Surrogate Conflict Field (SCF) between ego and target.
        :return: scf_value, ego_field, target_field, overlap_field
        """
        # Ego field (Self)
        ego_field = self.get_gaussian_field(
            ego_pos[0], ego_pos[1], 
            v_ego[0], v_ego[1], 
            sigma_x=1.0, sigma_z=3.0, # Ego field is usually larger
            weather_factor=weather_factor
        )
        
        # Target field
        target_field = self.get_gaussian_field(
            target_pos[0], target_pos[1], 
            v_target[0], v_target[1], 
            sigma_x=0.8, sigma_z=2.0,
            weather_factor=weather_factor
        )
        
        # Conflict overlap
        overlap_field = ego_field * target_field
        
        # Integral (Sum)
        scf_value = np.sum(overlap_field)
        
        return scf_value, ego_field, target_field, overlap_field

    def get_visualization_field(self, x_c, z_c, v_x=0, v_z=0, sigma_x=1.5, sigma_z=4.0, weather_factor=1.0):
        """
        Generate a larger, smoother Gaussian field specifically for visualization (Radar-like feel).
        Parameters are tuned for visual impact (larger sigma).
        """
        return self.get_gaussian_field(
            x_c, z_c, v_x, v_z, 
            sigma_x=sigma_x, sigma_z=sigma_z, 
            v_stretch_factor=0.3, # More stretch for visual effect
            weather_factor=weather_factor
        )

    def get_trajectory_risk_field(self, trajectory, velocities, time_decay=None, weather_factor=1.0):
        """
        Generate a risk field based on predicted trajectory.
        :param trajectory: List of predicted positions [(x1, z1), (x2, z2), ...]
        :param velocities: List of corresponding velocities [(vx1, vz1), (vx2, vz2), ...]
        :param time_decay: List of time decay factors for each predicted point
        :param weather_factor: Environmental risk multiplier
        :return: Combined risk field
        """
        if not trajectory:
            return np.zeros((self.grid_h, self.grid_w))
        
        # If time decay not provided, generate default
        if time_decay is None:
            steps = len(trajectory)
            time_decay = []
            for i in range(steps):
                weight = np.exp(-i * 0.5)  # 0.5 is decay coefficient
                time_decay.append(weight)
            # Normalize
            total = sum(time_decay)
            time_decay = [w / total for w in time_decay]
        
        # Initialize combined field
        combined_field = np.zeros((self.grid_h, self.grid_w))
        
        # Add risk for each predicted point
        for i, ((x, z), (vx, vz), weight) in enumerate(zip(trajectory, velocities, time_decay)):
            # Generate risk field for this point
            field = self.get_gaussian_field(
                x, z, vx, vz, 
                weather_factor=weather_factor
            )
            # Add to combined field with time decay weight
            combined_field += field * weight
        
        # Normalize to [0, 1]
        max_val = np.max(combined_field)
        if max_val > 0:
            combined_field /= max_val
        
        return combined_field

    def calculate_trajectory_risk(self, ego_pos, ego_vel, trajectories, velocities_list, time_decay=None, weather_factor=1.0):
        """
        Calculate risk based on predicted trajectories.
        :param ego_pos: Ego vehicle position (x, z)
        :param ego_vel: Ego vehicle velocity (vx, vz)
        :param trajectories: List of trajectories for multiple targets
        :param velocities_list: List of velocity lists for multiple targets
        :param time_decay: Time decay factors
        :param weather_factor: Environmental risk multiplier
        :return: Total risk score and risk field
        """
        total_risk = 0
        combined_field = np.zeros((self.grid_h, self.grid_w))
        
        for trajectory, velocities in zip(trajectories, velocities_list):
            if not trajectory:
                continue
            
            # Get trajectory risk field
            traj_field = self.get_trajectory_risk_field(trajectory, velocities, time_decay, weather_factor)
            
            # Get ego field
            ego_field = self.get_gaussian_field(
                ego_pos[0], ego_pos[1], 
                ego_vel[0], ego_vel[1], 
                sigma_x=1.0, sigma_z=3.0, 
                weather_factor=weather_factor
            )
            
            # Calculate overlap
            overlap = ego_field * traj_field
            risk = np.sum(overlap)
            total_risk += risk
            combined_field += traj_field
        
        # Normalize combined field
        max_val = np.max(combined_field)
        if max_val > 0:
            combined_field /= max_val
        
        return total_risk, combined_field


# ─────────────────────────────────────────────────────────────
#  风险等级阈值（SCF 尺度）—— 单一真源
#
#  标定依据：lanechange.mp4 300 帧 / conf=0.25 / iou=0.45
#  （见 docs/测试结果.md §2.1：P50=22.6 P95=26.4 P99=42.8 max=62.5，
#    分布极窄，分位数阈值会退化成"在噪声上划线"，故锚定尾部）
#
#  ⚠️ 所有消费端都必须引用这里，不要再各写一套数字：
#     - detect_3d_with_surface.py 的 decision_status
#     - data_store.py 的风险分级 / 高风险计数
#     - bbox3d_utils.py 的画框配色与 BEV 提示
#     - app.py 仪表盘的进度与颜色
#  否则会出现"画面判 HIGH、侧边栏显示安全"这类口径分裂。
# ─────────────────────────────────────────────────────────────
RISK_THRESHOLDS = {"HIGH": 50.0, "MEDIUM": 40.0, "LOW": 30.0}
RISK_LEVEL_ORDER = {"CLEAR": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
# 路面风险是 0–1 的独立尺度，用它换算成"SCF 等效值"以便合并展示：
# surface_risk = 1.0（最严重）等价于 HIGH 阈值
SURFACE_TO_SCF = RISK_THRESHOLDS["HIGH"]


def risk_level(scf: float) -> str:
    """SCF → 'HIGH' / 'MEDIUM' / 'LOW' / 'CLEAR'"""
    if scf >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    if scf >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    if scf >= RISK_THRESHOLDS["LOW"]:
        return "LOW"
    return "CLEAR"


def worse_level(a: str, b: str) -> str:
    """取两个等级中更严重的一个"""
    return a if RISK_LEVEL_ORDER.get(a, 0) >= RISK_LEVEL_ORDER.get(b, 0) else b


def level_from_road_danger(danger_level) -> str:
    """路面 road_danger_level(int 0–3) → 等级字符串"""
    return {3: "HIGH", 2: "MEDIUM", 1: "LOW"}.get(int(danger_level or 0), "CLEAR")
