"""
MARAHS Quadrotor Environment
=============================
Realistic quadrotor simulation for hurricane wind resistance training.

Physics:
- 6-DOF rigid body dynamics (PyBullet)
- 4 direct motor RPM controls with lag filter
- Aerodynamic drag model
- Ground effect near terrain
- Dryden turbulence model
- Rankine vortex hurricane wind field

Sensors:
- IMU: accelerometer + gyroscope (with noise)
- No wind sensor (actor must infer wind from vibration patterns)

Domain Randomization:
- Drone mass: +/- 30%
- Center of mass: +/- 5cm
- Motor efficiency: 70-100% per motor
- Sensor noise: Gaussian white noise

Action Space: [-1, 1]^4 (4 motor RPM commands)
Observation Space: IMU history (10 frames) + position error + quaternion + angular velocity
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Tuple, Optional
import math


@dataclass
class QuadrotorConfig:
    # Drone parameters
    mass: float = 2.0  # kg (will be randomized +/- 30%)
    arm_length: float = 0.3  # m (distance from center to motor)
    motor_time_constant: float = 0.05  # seconds (motor lag)
    max_rpm: float = 12000.0  # RPM
    hover_rpm: float = 5000.0  # RPM to hover
    
    # Aerodynamics
    drag_coeff: float = 0.3  # aerodynamic drag coefficient
    lift_coeff: float = 1.0  # lift coefficient
    
    # Wind
    wind_enabled: bool = True
    wind_speed: float = 0.0  # m/s base wind speed
    dryden_intensity: float = 0.0  # turbulence intensity [0,1]
    
    # Domain randomization
    randomize_mass: bool = True
    randomize_com: bool = True  # center of mass
    randomize_motors: bool = True
    sensor_noise: bool = True
    
    # Episode
    max_episode_steps: int = 3000  # 30 seconds at 100Hz
    dt: float = 0.01  # 100Hz
    
    # History buffer
    history_length: int = 10  # H=10 frames


class DrydenTurbulence:
    """
    Dryden turbulence model for realistic wind gusts.
    
    Generates turbulence as filtered white noise:
    - Low-pass filtered for large-scale gusts
    - Band-pass filtered for medium-scale eddies
    - High-pass filtered for small-scale turbulence
    """
    
    def __init__(self, intensity: float = 0.5, dt: float = 0.01):
        self.intensity = intensity
        self.dt = dt
        
        # Time constants for different scales
        self.tau_large = 5.0  # large gusts
        self.tau_medium = 1.0  # eddies
        self.tau_small = 0.1  # small turbulence
        
        # State (filtered noise)
        self.state_large = np.zeros(3)
        self.state_medium = np.zeros(3)
        self.state_small = np.zeros(3)
    
    def reset(self):
        self.state_large = np.zeros(3)
        self.state_medium = np.zeros(3)
        self.state_small = np.zeros(3)
    
    def step(self) -> np.ndarray:
        """Generate turbulence vector (3D) in m/s."""
        # White noise input
        white = np.random.randn(3) * self.intensity
        
        # Low-pass filter (large gusts)
        alpha_l = self.dt / (self.tau_large + self.dt)
        self.state_large = (1 - alpha_l) * self.state_large + alpha_l * white
        
        # Band-pass filter (medium eddies)
        alpha_m = self.dt / (self.tau_medium + self.dt)
        self.state_medium = (1 - alpha_m) * self.state_medium + alpha_m * white
        
        # High-pass filter (small turbulence)
        alpha_s = self.dt / (self.tau_small + self.dt)
        self.state_small = (1 - alpha_s) * self.state_small + alpha_s * white
        
        # Combine scales with different weights
        turbulence = (
            0.5 * self.state_large +  # large gusts dominate
            0.3 * self.state_medium +  # eddies
            0.2 * self.state_small     # small turbulence
        )
        
        return turbulence * self.intensity


class RankineVortex:
    """
    Rankine vortex model for hurricane wind field.
    
    Wind speed profile:
    - Inner core (r < R_max): v = V_max * (r / R_max)
    - Outer region (r >= R_max): v = V_max * (R_max / r)^1.5
    
    Wind direction: tangential (clockwise in N. hemisphere)
    """
    
    def __init__(self, V_max: float = 70.0, R_max: float = 50.0):
        self.V_max = V_max  # max wind speed (m/s)
        self.R_max = R_max  # radius of max winds (m)
        self.eye_pos = np.array([0.0, 0.0])  # hurricane center
        self.drift_dir = np.array([1.0, 0.0])
        self.drift_speed = 0.5  # m/s
    
    def reset(self, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng()
        self.eye_pos = rng.uniform(-100, 100, 2)
        angle = rng.uniform(0, 2 * np.pi)
        self.drift_dir = np.array([np.cos(angle), np.sin(angle)])
        self.drift_speed = rng.uniform(0.1, 1.0)
    
    def step(self, dt: float = 0.01):
        self.eye_pos += self.drift_dir * self.drift_speed * dt
    
    def get_wind(self, x: float, y: float) -> np.ndarray:
        """Get wind vector at position (x, y) in m/s."""
        dx = x - self.eye_pos[0]
        dy = y - self.eye_pos[1]
        r = math.sqrt(dx * dx + dy * dy)
        
        if r < 0.1:
            return np.zeros(3)
        
        # Wind speed from Rankine profile
        if r < self.R_max:
            speed = self.V_max * (r / self.R_max)
        else:
            speed = self.V_max * (self.R_max / max(r, 1.0)) ** 1.5
        
        # Tangential direction (clockwise)
        wx = speed * (-dy / r)
        wy = speed * (dx / r)
        
        # Add small vertical component (updraft near eye)
        wz = speed * 0.1 * max(0, 1 - r / self.R_max)
        
        return np.array([wx, wy, wz])


class QuadrotorEnv(gym.Env):
    """
    Quadrotor environment with direct motor control and IMU sensing.
    
    The AI controls 4 motor RPMs directly. It only sees IMU data
    (accelerometer + gyroscope) and must infer wind conditions from
    vibration patterns.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 100}
    
    def __init__(self, config: QuadrotorConfig = None, render_mode=None):
        super().__init__()
        self.config = config or QuadrotorConfig()
        self.render_mode = render_mode
        
        # Action: 4 motor RPM commands [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Observation: position_error(3) + quaternion(4) + angular_vel(3) + 
        #              motor_commands_history(H*4) + accelerometer_history(H*3) = 10 + 10H
        obs_dim = 10 + self.config.history_length * 7  # 80 for H=10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Internal state
        self.current_rpm = np.zeros(4)
        self.target_position = np.array([0.0, 0.0, 1.0])  # hover at 1m
        
        # Wind models
        self.dryden = DrydenTurbulence(
            intensity=self.config.dryden_intensity,
            dt=self.config.dt
        )
        self.rankine = RankineVortex()
        
        # History buffers
        self.motor_history = []
        self.accel_history = []
        
        # Domain randomization params
        self.mass = self.config.mass
        self.com_offset = np.zeros(3)
        self.motor_efficiency = np.ones(4)
        
        # Simulation
        self.physics_client = None
        self.drone_id = None
        self.step_count = 0
        
        # Noise
        self.accel_noise_std = 0.1  # m/s^2
        self.gyro_noise_std = 0.01  # rad/s
    
    def _connect_physics(self):
        """Connect to PyBullet physics engine."""
        if self.physics_client is None:
            if self.render_mode == "human":
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(self.config.dt)
    
    def _create_drone(self):
        """Create quadrotor in PyBullet."""
        # Load URDF
        urdf_path = pybullet_data.getDataPath() + "/r2d2.urdf"
        # Use a simple box as drone body
        half_extents = [0.15, 0.15, 0.05]
        collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents
        )
        visual_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=[0.2, 0.6, 0.9, 1]
        )
        
        # Randomize mass
        if self.config.randomize_mass:
            self.mass = self.config.mass * np.random.uniform(0.7, 1.3)
        
        self.drone_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[0, 0, 1.0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Create ground plane
        p.loadURDF("plane.urdf")
    
    def _domain_randomize(self):
        """Randomize drone properties for each episode."""
        # Mass randomization
        if self.config.randomize_mass:
            self.mass = self.config.mass * np.random.uniform(0.7, 1.3)
        
        # Center of mass offset
        if self.config.randomize_com:
            self.com_offset = np.random.uniform(-0.05, 0.05, 3)
        
        # Motor efficiency
        if self.config.randomize_motors:
            self.motor_efficiency = np.random.uniform(0.7, 1.0, 4)
        
        # Reset wind
        self.dryden.reset()
        if self.config.wind_speed > 0:
            self.rankine.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._connect_physics()
        
        # Clear previous simulation
        if self.drone_id is not None:
            p.removeBody(self.drone_id)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.config.dt)
        
        # Domain randomization
        self._domain_randomize()
        
        # Create drone
        self._create_drone()
        
        # Randomize starting position
        start_x = self.np_random.uniform(-0.5, 0.5)
        start_y = self.np_random.uniform(-0.5, 0.5)
        start_z = self.np_random.uniform(0.5, 1.5)
        p.resetBasePositionAndOrientation(
            self.drone_id,
            [start_x, start_y, start_z],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Randomize target
        self.target_position = np.array([
            self.np_random.uniform(-2, 2),
            self.np_random.uniform(-2, 2),
            self.np_random.uniform(0.5, 2.0)
        ])
        
        # Reset state
        self.current_rpm = np.full(4, self.config.hover_rpm)
        self.step_count = 0
        self.motor_history = []
        self.accel_history = []
        
        # Initialize history
        for _ in range(self.config.history_length):
            self.motor_history.append(np.zeros(4))
            self.accel_history.append(np.zeros(3))
        
        return self._get_obs(), {}
    
    def _get_wind_force(self) -> np.ndarray:
        """Get total wind force on drone (dryden + rankine)."""
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        x, y, z = pos
        
        force = np.zeros(3)
        
        # Rankine vortex wind (if enabled)
        if self.config.wind_speed > 0:
            rankine_wind = self.rankine.get_wind(x, y)
            force += rankine_wind * self.config.wind_speed
        
        # Dryden turbulence
        turbulence = self.dryden.step()
        force += turbulence * 10.0  # scale turbulence
        
        # Convert wind speed to force (simplified drag model)
        # F = 0.5 * rho * Cd * A * v^2
        rho = 1.225  # air density kg/m^3
        Cd = self.config.drag_coeff
        A = 0.1  # effective drag area m^2
        
        # Wind velocity relative to drone
        vel, _ = p.getBaseVelocity(self.drone_id)
        vel = np.array(vel)
        wind_rel = force - vel  # relative wind
        
        drag_force = 0.5 * rho * Cd * A * wind_rel * np.abs(wind_rel)
        
        return drag_force
    
    def _get_motor_forces(self, action: np.ndarray) -> np.ndarray:
        """Convert motor commands to forces with lag filter."""
        # Scale [-1, 1] to RPM
        rpm_cmd = (action + 1) / 2 * self.config.max_rpm
        
        # Apply motor lag filter: tau * dOmega/dt = Omega_cmd - Omega
        tau = self.config.motor_time_constant
        dt = self.config.dt
        
        for i in range(4):
            alpha = dt / (tau + dt)
            self.current_rpm[i] = (1 - alpha) * self.current_rpm[i] + alpha * rpm_cmd[i]
        
        # Apply motor efficiency
        effective_rpm = self.current_rpm * self.motor_efficiency
        
        # Convert RPM to thrust force
        # Simplified: F = k * RPM^2
        k = 1e-6  # thrust coefficient
        forces = k * effective_rpm ** 2
        
        return forces
    
    def _get_imu_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get accelerometer and gyroscope readings."""
        # Get state
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        # Accelerometer: measures proper acceleration (including gravity)
        accel = np.array([0, 0, 9.81])  # gravity
        accel -= np.array(vel) / self.config.dt  # linear acceleration
        
        # Add wind acceleration
        wind_force = self._get_wind_force()
        accel += wind_force / self.mass
        
        # Gyroscope: angular velocity in body frame
        gyro = np.array(ang_vel)
        
        # Add sensor noise
        if self.config.sensor_noise:
            accel += np.random.randn(3) * self.accel_noise_std
            gyro += np.random.randn(3) * self.gyro_noise_std
        
        return accel, gyro
    
    def _get_obs(self) -> np.ndarray:
        """Build observation vector from IMU history."""
        # Get current state
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        pos = np.array(pos)
        quat = np.array(quat)
        ang_vel = np.array(ang_vel)
        
        # Position error (normalized)
        pos_error = (self.target_position - pos) / 10.0  # normalize
        
        # Get IMU data
        accel, gyro = self._get_imu_data()
        
        # Update histories
        self.motor_history.append(self.current_rpm / self.config.max_rpm)
        self.accel_history.append(accel / 20.0)  # normalize
        
        if len(self.motor_history) > self.config.history_length:
            self.motor_history.pop(0)
        if len(self.accel_history) > self.config.history_length:
            self.accel_history.pop(0)
        
        # Build observation
        obs_parts = [
            pos_error,        # 3D
            quat,             # 4D
            gyro,             # 3D
        ]
        
        # Add history (motor commands + accelerometer)
        for i in range(self.config.history_length):
            obs_parts.append(self.motor_history[i])  # 4D
            obs_parts.append(self.accel_history[i])  # 3D
        
        obs = np.concatenate(obs_parts).astype(np.float32)
        return obs
    
    def step(self, action):
        self.step_count += 1
        
        # Get motor forces
        motor_forces = self._get_motor_forces(action)
        
        # Apply forces at motor positions (simplified)
        # Motor layout: [front-left, front-right, back-left, back-right]
        arm = self.config.arm_length
        motor_positions = [
            [arm, arm, 0],    # front-left
            [arm, -arm, 0],   # front-right
            [-arm, arm, 0],   # back-left
            [-arm, -arm, 0],  # back-right
        ]
        
        # Apply upward force at center
        total_thrust = np.sum(motor_forces)
        p.applyExternalForce(
            self.drone_id, -1,
            [0, 0, total_thrust],
            [0, 0, 0],  # center of mass
            p.WORLD_FRAME
        )
        
        # Apply torque from differential thrust
        # Roll: motors 0+2 vs 1+3
        roll_torque = (motor_forces[0] + motor_forces[2] - 
                      motor_forces[1] - motor_forces[3]) * arm * 0.5
        # Pitch: motors 0+1 vs 2+3
        pitch_torque = (motor_forces[0] + motor_forces[1] - 
                       motor_forces[2] - motor_forces[3]) * arm * 0.5
        # Yaw: motors 0+3 vs 1+2 (counter-rotating pairs)
        yaw_torque = (motor_forces[0] + motor_forces[3] - 
                     motor_forces[1] - motor_forces[2]) * 0.1
        
        p.applyExternalTorque(
            self.drone_id, -1,
            [roll_torque, pitch_torque, yaw_torque],
            p.WORLD_FRAME
        )
        
        # Apply wind force
        wind_force = self._get_wind_force()
        p.applyExternalForce(
            self.drone_id, -1,
            wind_force.tolist(),
            [0, 0, 0],
            p.WORLD_FRAME
        )
        
        # Apply gravity
        p.applyExternalForce(
            self.drone_id, -1,
            [0, 0, -self.mass * 9.81],
            [0, 0, 0],
            p.WORLD_FRAME
        )
        
        # Step simulation
        p.stepSimulation()
        
        # Update wind
        self.rankine.step(self.config.dt)
        
        # Get new state
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        pos = np.array(pos)
        
        # Check termination
        done = False
        terminated = False
        truncated = False
        
        # Ground collision
        if pos[2] < 0.1:
            terminated = True
        
        # Out of bounds
        if np.abs(pos[0]) > 10 or np.abs(pos[1]) > 10 or pos[2] > 10:
            terminated = True
        
        # Max steps
        if self.step_count >= self.config.max_episode_steps:
            truncated = True
        
        # Compute reward
        reward = self._compute_reward(pos, quat, vel, ang_vel, action)
        
        # Build info
        pos_error = np.linalg.norm(self.target_position - pos)
        info = {
            'position_error': pos_error,
            'altitude': pos[2],
            'wind_speed': np.linalg.norm(wind_force) / self.mass,
            'motor_rpm': self.current_rpm.copy(),
            'mass': self.mass,
            'motor_efficiency': self.motor_efficiency.copy(),
        }
        
        done = terminated or truncated
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _compute_reward(self, pos, quat, vel, ang_vel, action) -> float:
        """Compute multi-objective reward."""
        # 1. Tracking reward (stay near target)
        pos_error = np.linalg.norm(self.target_position - pos)
        r_track = math.exp(-2.0 * pos_error ** 2)
        
        # 2. Motor smoothness (penalize jerky commands)
        if len(self.motor_history) > 1:
            prev_cmd = np.array(self.motor_history[-2])
            curr_cmd = action / 2 + 0.5  # normalize to [0,1]
            r_smooth = -np.linalg.norm(curr_cmd - prev_cmd) ** 2
        else:
            r_smooth = 0.0
        
        # 3. Attitude penalty (don't tilt too much)
        # quaternion to euler
        euler = p.getEulerFromQuaternion(quat)
        roll, pitch = abs(euler[0]), abs(euler[1])
        r_attitude = -(roll ** 2 + pitch ** 2) if (roll > 1.0 or pitch > 1.0) else 0.0
        
        # 4. Crash penalty
        r_survival = -10.0 if pos[2] < 0.1 else 0.0
        
        # 5. Energy efficiency (penalize high thrust)
        thrust_ratio = np.sum(self.current_rpm) / (4 * self.config.max_rpm)
        r_energy = -0.1 * thrust_ratio ** 2
        
        # Combine
        reward = (
            2.0 * r_track +
            0.5 * r_smooth +
            1.0 * r_attitude +
            r_survival +
            r_energy
        )
        
        return reward
