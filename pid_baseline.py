"""
Baseline Controllers for Hurricane Drone Coverage
==================================================
Provides multiple baselines for comparison:
1. RandomBaseline: Random actions
2. HoverBaseline: Stay in place
3. PIDBaseline: PID controller for station-keeping + coverage
4. GreedyBaseline: Move toward nearest uncovered cell

These baselines are essential for proving the AI actually learns.
"""

import numpy as np
from typing import Dict, Optional


class RandomBaseline:
    """Random actions — lower bound for performance."""
    
    def __init__(self, action_dim=4):
        self.action_dim = action_dim
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
    
    def reset(self):
        pass


class HoverBaseline:
    """Stay in place — tests if wind resistance works."""
    
    def __init__(self):
        pass
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # Output zero action (hover)
        return np.zeros(4, dtype=np.float32)
    
    def reset(self):
        pass


class PIDBaseline:
    """
    PID controller for station-keeping + coverage.
    
    Uses:
    - PID for altitude hold
    - PID for position hold (against wind)
    - Greedy coverage: move toward nearest uncovered cell
    """
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.5, 
                 coverage_kp=0.5, altitude_target=15.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.coverage_kp = coverage_kp
        self.altitude_target = altitude_target
        
        # Error accumulators
        self.alt_error_sum = 0.0
        self.prev_alt_error = 0.0
        self.pos_error_sum = np.zeros(2)
        self.prev_pos_error = np.zeros(2)
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get PID action from observation.
        
        obs contains:
        - pos(3), vel(3), orient(4), ang_vel(3), motors(4), wind(3),
        - debris_radar(12), coverage_frac(1), target_dir(2), target_dist(1),
        - alt_err(1), step(1)
        """
        # Extract relevant info
        position = obs[:3]
        velocity = obs[3:6]
        target_dir = obs[27:29]
        target_dist = obs[29]
        alt_error = obs[31]
        
        # PID for altitude
        self.alt_error_sum += alt_error
        self.alt_error_sum = np.clip(self.alt_error_sum, -10, 10)
        alt_d = alt_error - self.prev_alt_error
        self.prev_alt_error = alt_error
        
        # Throttle: maintain altitude
        throttle = -(self.kp * alt_error + self.ki * self.alt_error_sum + self.kd * alt_d)
        throttle = np.clip(throttle, -0.3, 0.3)
        
        # PID for position (move toward uncovered cells)
        if target_dist > 0.01:
            # Move toward nearest uncovered cell
            roll = -target_dir[1] * self.coverage_kp
            pitch = target_dir[0] * self.coverage_kp
        else:
            # Hover if all cells covered
            roll = 0.0
            pitch = 0.0
        
        # Wind compensation (use wind info from obs)
        wind = obs[13:16]
        roll += wind[1] * 0.01  # compensate for wind
        pitch -= wind[0] * 0.01
        
        # Yaw: face target
        yaw = 0.0
        
        action = np.array([throttle, roll, pitch, yaw], dtype=np.float32)
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.alt_error_sum = 0.0
        self.prev_alt_error = 0.0
        self.pos_error_sum = np.zeros(2)
        self.prev_pos_error = np.zeros(2)


class GreedyBaseline:
    """Greedy coverage: always move toward nearest uncovered cell."""
    
    def __init__(self):
        pass
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        target_dir = obs[27:29]
        target_dist = obs[29]
        alt_error = obs[31]
        
        # Move toward target
        if target_dist > 0.01:
            pitch = target_dir[0] * 0.5
            roll = -target_dir[1] * 0.5
        else:
            pitch = 0.0
            roll = 0.0
        
        # Altitude control
        throttle = -alt_error * 0.3
        
        return np.array([throttle, roll, pitch, 0.0], dtype=np.float32)
    
    def reset(self):
        pass
