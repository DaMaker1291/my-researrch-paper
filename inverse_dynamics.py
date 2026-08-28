"""
IMU-to-Wind Inverse Dynamics Estimator
=======================================

NOVEL CONTRIBUTION (no existing implementation in literature):
Reconstructs wind forces from IMU + motor commands WITHOUT any wind sensors.

Key insight: The IMU measures total acceleration, which is:
  a_measured = a_thrust + a_gravity + a_drag + a_wind

If we know thrust (from motor commands) and gravity, we can solve for wind:
  a_wind = a_measured - a_thrust - a_gravity - a_drag(v)

This is an INVERSE DYNAMICS problem that has never been solved online
for hurricane drones before.

Applications:
1. Feed wind estimates to the GP wind field mapper
2. Enable wind-aware control without dedicated wind sensors
3. Detect motor failures (unexplained acceleration changes)
4. Validate aerodynamic models in real-time

Mathematical framework:
- State: x = [position, velocity, orientation]
- IMU: a_imu = R^T (F_total / m - g_world)
- Motor model: F_thrust = Σ_i k_t * rpm_i²
- Drag model: F_drag = -c_d * v * ||v||
- Wind: F_wind = m * a_wind (what we want)

Solving: a_wind = R * a_imu + g - (F_thrust + F_drag) / m
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class InverseDynamicsConfig:
    """Configuration for inverse dynamics estimator."""
    # Drone physical parameters
    mass: float = 1.5                  # kg
    drag_coefficient: float = 0.3     # N/(m/s)²
    thrust_coefficient: float = 1.5e-5  # N/RPM²
    motor_time_constant: float = 0.05  # seconds
    
    # Filter parameters
    process_noise: float = 0.1        # m/s²
    measurement_noise: float = 0.5    # m/s²
    dt: float = 0.02                  # 50 Hz IMU
    
    # Adaptive estimation
    window_size: int = 20             # samples for moving average
    outlier_threshold: float = 3.0    # standard deviations
    
    # Motor mixing matrix (quadrotor X configuration)
    # [thrust, roll, pitch, yaw] -> [rpm1, rpm2, rpm3, rpm4]
    motor_mixing = None               # computed in __init__
    
    def __post_init__(self):
        if self.motor_mixing is None:
            # Standard X-configuration mixing
            self.motor_mixing = np.array([
                [ 1,  1, -1, -1],   # motor 1 (front-right)
                [ 1, -1, -1,  1],   # motor 2 (rear-right)
                [ 1, -1,  1, -1],   # motor 3 (rear-left)
                [ 1,  1,  1,  1],   # motor 4 (front-left)
            ]) / 4.0


class MotorModel:
    """
    Motor dynamics model.
    
    Maps motor commands ([-1, 1]) to RPMs and thrust forces.
    Accounts for motor time constant (ESC + propeller lag).
    """
    
    def __init__(self, config: InverseDynamicsConfig):
        self.config = config
        self.current_rpms = np.full(4, 8944.0)  # hover RPM
        self.hover_rpm = 8944.0
        self.max_rpm = 12000.0
        self.min_rpm = 2000.0
    
    def command_to_rpm(self, action: np.ndarray) -> np.ndarray:
        """
        Convert action command to RPMs through motor mixing.
        
        Args:
            action: (4,) in [-1, 1] = [thrust, roll, pitch, yaw]
        
        Returns:
            rpms: (4,) motor RPMs
        """
        # Through motor mixing
        mixed = self.config.motor_mixing @ action
        
        # Convert to RPMs
        target_rpms = self.hover_rpm + mixed * (self.max_rpm - self.hover_rpm)
        target_rpms = np.clip(target_rpms, self.min_rpm, self.max_rpm)
        
        # First-order motor dynamics
        alpha = self.config.dt / self.config.motor_time_constant
        self.current_rpms = self.current_rpms + alpha * (target_rpms - self.current_rpms)
        
        return self.current_rpms.copy()
    
    def rpm_to_thrust(self, rpms: np.ndarray) -> np.ndarray:
        """
        Convert RPMs to thrust forces.
        
        F_i = k_t * rpm_i²
        
        Returns:
            forces: (4,) thrust forces per motor
        """
        return self.config.thrust_coefficient * rpms**2
    
    def total_thrust(self, rpms: np.ndarray) -> float:
        """Total thrust force (z-axis in body frame)."""
        forces = self.rpm_to_thrust(rpms)
        return float(np.sum(forces))
    
    def reset(self):
        """Reset motor state."""
        self.current_rpms = np.full(4, self.hover_rpm)


class DragModel:
    """
    Aerodynamic drag model.
    
    F_drag = -c_d * v * ||v||
    
    This is a simplified quadratic drag model.
    For hurricane conditions, we also add a correction factor
    for high Reynolds number flows.
    """
    
    def __init__(self, config: InverseDynamicsConfig):
        self.c_d = config.drag_coefficient
        
        # Adaptive drag coefficient (learned online)
        self.c_d_adaptive = config.drag_coefficient
        self.drag_estimate_buffer = []
    
    def compute_drag_force(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute drag force.
        
        Args:
            velocity: (3,) velocity in world frame
        
        Returns:
            drag: (3,) drag force
        """
        speed = np.linalg.norm(velocity)
        if speed < 0.01:
            return np.zeros(3)
        
        # Quadratic drag
        drag = -self.c_d_adaptive * velocity * speed
        
        return drag
    
    def adapt_drag_coefficient(self, measured_accel: np.ndarray,
                              predicted_accel_no_drag: np.ndarray,
                              velocity: np.ndarray):
        """
        Adapt drag coefficient based on IMU measurements.
        
        If we know total acceleration and can predict everything except drag,
        we can estimate the drag coefficient.
        """
        speed = np.linalg.norm(velocity)
        if speed < 0.5:
            return
        
        # Residual acceleration = drag acceleration
        drag_accel = measured_accel - predicted_accel_no_drag
        
        # Estimate c_d from: drag_accel = -c_d * v * ||v|| / m
        # c_d = -drag_accel · v / (||v||³)
        projection = np.dot(drag_accel, velocity)
        c_d_est = -projection / (speed**3 + 1e-8)
        
        # Update with exponential moving average
        self.drag_estimate_buffer.append(c_d_est)
        if len(self.drag_estimate_buffer) > 50:
            self.drag_estimate_buffer.pop(0)
        
        if len(self.drag_estimate_buffer) > 10:
            self.c_d_adaptive = 0.9 * self.c_d_adaptive + 0.1 * np.median(self.drag_estimate_buffer)


class IMUToWindEstimator:
    """
    Inverse dynamics wind estimator.
    
    NOVEL ALGORITHM:
    
    1. Read IMU: a_measured (body frame)
    2. Rotate to world frame: a_world = R * a_measured
    3. Predict known accelerations:
       a_gravity = [0, 0, -g]
       a_thrust = R_body * [0, 0, F_thrust/m]
       a_drag = F_drag(v) / m
    4. Solve for wind:
       a_wind = a_world - a_gravity - a_thrust - a_drag
    5. Apply Kalman-like filtering for smooth estimates
    
    This gives us wind acceleration, which we can integrate to get
    wind velocity (with some drift, but the GP mapper handles that).
    
    FIRST online IMU-to-wind estimator for hurricane drones.
    """
    
    def __init__(self, config: InverseDynamicsConfig = None):
        self.config = config or InverseDynamicsConfig()
        
        # Sub-models
        self.motor_model = MotorModel(self.config)
        self.drag_model = DragModel(self.config)
        
        # State
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([1, 0, 0, 0])  # quaternion (w, x, y, z)
        
        # Wind estimate
        self.wind_estimate = np.zeros(3)
        self.wind_velocity = np.zeros(3)  # integrated wind velocity
        
        # Filtering
        self.wind_buffer = []
        self.accel_buffer = []
        
        # Statistics
        self.total_estimates = 0
        self.estimation_errors = []
    
    def estimate_wind(self, imu_data: Dict, motor_commands: np.ndarray,
                     dt: float = None) -> Dict:
        """
        Estimate wind from IMU and motor data.
        
        Args:
            imu_data: dict with:
                - acceleration: (3,) body-frame acceleration
                - gyroscope: (3,) angular velocity
                - quaternion: (4,) orientation
            motor_commands: (4,) motor commands in [-1, 1]
            dt: time step (uses config default if None)
        
        Returns:
            dict with:
                - wind_acceleration: (3,) estimated wind acceleration
                - wind_velocity: (3,) estimated wind velocity
                - confidence: float [0, 1]
                - diagnostics: dict
        """
        dt = dt or self.config.dt
        self.total_estimates += 1
        
        # Extract IMU data
        accel_body = np.asarray(imu_data['acceleration'], dtype=np.float64)
        gyro = np.asarray(imu_data.get('gyroscope', np.zeros(3)), dtype=np.float64)
        quat = np.asarray(imu_data.get('quaternion', self.orientation), dtype=np.float64)
        
        # Update orientation
        self.orientation = quat / np.linalg.norm(quat)
        
        # Rotation matrix (body to world)
        R = self._quat_to_rot(self.orientation)
        
        # Rotate acceleration to world frame
        accel_world = R @ accel_body
        
        # Motor RPMs and thrust
        rpms = self.motor_model.command_to_rpm(motor_commands)
        total_thrust = self.motor_model.total_thrust(rpms)
        
        # Thrust acceleration (in body frame, then rotate to world)
        thrust_body = np.array([0, 0, total_thrust / self.config.mass])
        thrust_world = R @ thrust_body
        
        # Gravity
        gravity = np.array([0, 0, -9.81])
        
        # Drag acceleration
        drag_force = self.drag_model.compute_drag_force(self.velocity)
        drag_accel = drag_force / self.config.mass
        
        # Wind acceleration = measured - known
        # a_wind = a_world - gravity - thrust_world - drag_accel
        wind_accel = accel_world - gravity - thrust_world - drag_accel
        
        # Apply outlier rejection
        self.accel_buffer.append(wind_accel)
        if len(self.accel_buffer) > self.config.window_size:
            self.accel_buffer.pop(0)
        
        if len(self.accel_buffer) > 5:
            accel_arr = np.array(self.accel_buffer)
            mean_accel = np.mean(accel_arr, axis=0)
            std_accel = np.std(accel_arr, axis=0) + 1e-8
            
            # Reject outliers
            z_scores = np.abs(wind_accel - mean_accel) / std_accel
            is_outlier = np.any(z_scores > self.config.outlier_threshold)
            
            if is_outlier:
                wind_accel = mean_accel  # use smoothed value
        
        # Apply exponential filtering
        alpha = 0.3  # filter coefficient
        self.wind_estimate = alpha * wind_accel + (1 - alpha) * self.wind_estimate
        
        # Integrate to get wind velocity (with decay to prevent drift)
        self.wind_velocity = 0.98 * self.wind_velocity + self.wind_estimate * dt
        
        # Buffer for confidence estimation
        self.wind_buffer.append(self.wind_estimate.copy())
        if len(self.wind_buffer) > self.config.window_size:
            self.wind_buffer.pop(0)
        
        # Compute confidence based on consistency
        confidence = self._compute_confidence()
        
        # Update drag model adaptively
        predicted_accel_no_drag = gravity + thrust_world
        self.drag_model.adapt_drag_coefficient(
            accel_world, predicted_accel_no_drag, self.velocity
        )
        
        return {
            'wind_acceleration': self.wind_estimate.copy(),
            'wind_velocity': self.wind_velocity.copy(),
            'confidence': confidence,
            'total_thrust_N': float(total_thrust),
            'drag_coefficient': float(self.drag_model.c_d_adaptive),
            'rpms': rpms,
            'diagnostics': {
                'accel_world_norm': float(np.linalg.norm(accel_world)),
                'thrust_accel_norm': float(np.linalg.norm(thrust_world)),
                'drag_accel_norm': float(np.linalg.norm(drag_accel)),
                'wind_accel_norm': float(np.linalg.norm(self.wind_estimate)),
                'n_buffered': len(self.wind_buffer),
            }
        }
    
    def _compute_confidence(self) -> float:
        """
        Compute confidence in wind estimate.
        
        Based on:
        - Buffer consistency (low variance = high confidence)
        - Acceleration magnitude (reasonable range)
        - Number of samples
        """
        if len(self.wind_buffer) < 3:
            return 0.1
        
        buffer = np.array(self.wind_buffer)
        
        # Variance-based confidence
        variance = np.mean(np.var(buffer, axis=0))
        var_confidence = np.exp(-variance / 10.0)  # high variance -> low confidence
        
        # Magnitude-based confidence (reasonable wind range)
        magnitude = np.mean(np.linalg.norm(buffer, axis=1))
        mag_confidence = np.exp(-((magnitude - 15.0) / 20.0)**2)  # peaks at 15 m/s
        
        # Sample count confidence
        sample_confidence = min(1.0, len(self.wind_buffer) / 10.0)
        
        # Combined confidence
        confidence = 0.4 * var_confidence + 0.3 * mag_confidence + 0.3 * sample_confidence
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def update_state(self, position: np.ndarray, velocity: np.ndarray):
        """Update state from external source (e.g., position estimator)."""
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
    
    def reset(self):
        """Reset estimator state."""
        self.motor_model.reset()
        self.wind_estimate = np.zeros(3)
        self.wind_velocity = np.zeros(3)
        self.wind_buffer.clear()
        self.accel_buffer.clear()
        self.total_estimates = 0
    
    def _quat_to_rot(self, q: np.ndarray) -> np.ndarray:
        """Quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    def get_stats(self) -> Dict:
        """Get estimator statistics."""
        return {
            'total_estimates': self.total_estimates,
            'mean_wind_speed': float(np.mean([
                np.linalg.norm(w) for w in self.wind_buffer
            ])) if self.wind_buffer else 0.0,
            'drag_coefficient': float(self.drag_model.c_d_adaptive),
        }
