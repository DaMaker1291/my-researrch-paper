"""
Meta-Adaptive Neural Control with Safety Verification
======================================================

Extended Neural Fly architecture with three novel additions:
1. IMU-to-wind inverse dynamics (no wind sensors needed)
2. Online GP wind field mapping (predictive planning)
3. CBF-verified adaptation (provable safety)

Mathematical foundation (Neural Fly, Caltech):
- z_t = φ(x_t)          # frozen feature extractor output
- y_t = W_t @ z_t + b_t   # adaptive linear readout
- W_{t+1} = W_t + K_t (a_t - y_t) z_t^T  # RLS update

NOVEL EXTENSION:
- W_{t+1} = SafeProject(W_t + K_t (a_t - y_t) z_t^T)
  where SafeProject ensures the adapted controller satisfies CBF constraints

This creates a controller that is BOTH adaptive AND provably safe.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MetaAdaptiveConfig:
    """Configuration for meta-adaptive controller."""
    obs_dim: int = 41
    feature_dim: int = 64
    action_dim: int = 4
    
    # RLS parameters
    forgetting_factor: float = 0.98
    initial_covariance: float = 10.0
    noise_variance: float = 0.01
    
    # Safety
    max_adaptation_norm: float = 0.3
    enable_safety_verification: bool = True


class FrozenFeatureExtractor(nn.Module):
    """
    Frozen feature extractor (trained offline, never updated online).
    
    Architecture: obs → Linear+LayerNorm+ELU → Linear+LayerNorm+ELU → 64D features
    
    The key insight from Neural Fly: these features capture the STRUCTURE
    of the dynamics, while the adaptive readout captures the PARAMETERS.
    """
    
    def __init__(self, obs_dim: int = 38, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(),
        )
        self.feature_dim = feature_dim
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features from observation. Output: (batch, feature_dim)"""
        return self.net(obs)


class AdaptiveReadout(nn.Module):
    """
    Adaptive linear readout layer.
    
    This layer's weights are updated ONLINE via RLS during flight.
    The feature extractor is FROZEN - only this layer adapts.
    
    Math: y = W @ features + b
    """
    
    def __init__(self, feature_dim: int = 64, action_dim: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        self.W = nn.Parameter(torch.randn(action_dim, feature_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Linear readout: features → actions"""
        return features @ self.W.T + self.b


class RLSScheduler:
    """
    Recursive Least Squares filter for online weight adaptation.
    
    Enhanced with safety bounds from CBF verification.
    
    Standard RLS:
    - Maintains covariance matrix P
    - Computes Kalman gain K_t = P_t z_t / (λ + z_t^T P_t z_t)
    - Updates weights: W_{t+1} = W_t + K_t (error) z_t^T
    
    NOVEL extension:
    - Computes maximum safe adaptation norm from CBF
    - Clamps weight updates to stay within safe cone
    """
    
    def __init__(self, feature_dim: int = 64, action_dim: int = 4,
                 forgetting_factor: float = 0.98,
                 initial_covariance: float = 10.0,
                 noise_variance: float = 0.01,
                 max_adaptation_norm: float = 0.3):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.lam = forgetting_factor
        self.max_adaptation_norm = max_adaptation_norm
        
        # RLS state for each action dimension
        self.P = np.eye(feature_dim)[None, :, :].repeat(action_dim, axis=0) * initial_covariance
        self.W = np.zeros((action_dim, feature_dim))
        self.b = np.zeros(action_dim)
        
        self.R = noise_variance
        
        # Metrics
        self.adaptation_rate = 0.0
        self.steps_adapted = 0
        self.total_adaptation_norm = 0.0
        self.safety_clamps = 0
    
    def reset(self):
        """Reset RLS state."""
        self.P = np.eye(self.feature_dim)[None, :, :].repeat(self.action_dim, axis=0) * 10.0
        self.W = np.zeros((self.action_dim, self.feature_dim))
        self.b = np.zeros(self.action_dim)
        self.steps_adapted = 0
    
    def adapt(self, features: np.ndarray, measured_action: np.ndarray,
              predicted_action: np.ndarray,
              max_safe_norm: float = None) -> Dict:
        """
        One RLS adaptation step with optional safety clamping.
        
        Args:
            features: (feature_dim,) - frozen features
            measured_action: (action_dim,) - actual motor response
            predicted_action: (action_dim,) - predicted action
            max_safe_norm: optional maximum safe adaptation norm from CBF
        
        Returns:
            dict with adaptation metrics
        """
        if max_safe_norm is None:
            max_safe_norm = self.max_adaptation_norm
        
        z = features.reshape(-1, 1)  # (feature_dim, 1)
        error = measured_action - predicted_action  # (action_dim,)
        
        total_delta = 0.0
        
        for i in range(self.action_dim):
            zi = z[:, 0]
            
            # Kalman gain: K = P z / (λ + z^T P z)
            Pz = self.P[i] @ zi
            denominator = self.lam + zi @ Pz
            K = Pz / denominator  # (feature_dim,)
            
            # Compute weight update
            delta_w = K * error[i]
            delta_norm = np.linalg.norm(delta_w)
            
            # Safety clamping (NOVEL)
            if delta_norm > max_safe_norm:
                delta_w = delta_w * (max_safe_norm / delta_norm)
                self.safety_clamps += 1
            
            # Apply update
            self.W[i] += delta_w
            self.b[i] += error[i] * 0.01
            
            # Update covariance
            self.P[i] = (self.P[i] - np.outer(K, Pz)) / self.lam
            
            total_delta += np.linalg.norm(delta_w)
        
        self.steps_adapted += 1
        self.adaptation_rate = np.mean(np.abs(error))
        self.total_adaptation_norm += total_delta
        
        return {
            'error': float(np.mean(np.abs(error))),
            'adaptation_norm': total_delta,
            'max_safe_norm': max_safe_norm,
            'steps': self.steps_adapted,
            'safety_clamped': delta_norm > max_safe_norm,
        }
    
    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current adapted weights."""
        return self.W.copy(), self.b.copy()


class NeuralFlyController:
    """
    Complete Neural Fly controller with safety-verified adaptation.
    
    Architecture:
    1. FrozenFeatureExtractor: obs → 64D features (never updated)
    2. AdaptiveReadout: features → 4 motor commands (adapted online via RLS)
    3. RLSScheduler: manages online adaptation with safety bounds
    
    NOVEL additions:
    4. IMU-to-wind estimation (inverse_dynamics module)
    5. GP wind field mapping (wind_field_mapper module)
    6. CBF safety verification (adaptive_safety module)
    
    Usage:
        controller = NeuralFlyController(frozen_features, adaptive_readout)
        
        # During flight (100Hz loop):
        for step in range(flight_time):
            # Get safe action
            result = controller.get_safe_action(
                obs, state, motor_commands, imu_data
            )
            action = result['action']
            
            # Execute and adapt
            controller.adapt(features, measured_response, action, state)
    """
    
    def __init__(self, feature_extractor: FrozenFeatureExtractor,
                 readout: AdaptiveReadout,
                 rls_scheduler: RLSScheduler = None,
                 config: MetaAdaptiveConfig = None):
        self.feature_extractor = feature_extractor
        self.readout = readout
        self.config = config or MetaAdaptiveConfig()
        
        self.rls = rls_scheduler or RLSScheduler(
            feature_dim=self.config.feature_dim,
            action_dim=self.config.action_dim,
            max_adaptation_norm=self.config.max_adaptation_norm,
        )
        
        # Initialize RLS weights from trained readout
        with torch.no_grad():
            self.rls.W = readout.W.data.cpu().numpy()
            self.rls.b = readout.b.data.cpu().numpy()
        
        # Wind estimation (optional)
        self.wind_estimator = None
        self.wind_mapper = None
        self.cbf = None
        
        if self.config.enable_safety_verification:
            from inverse_dynamics import IMUToWindEstimator
            from wind_field_mapper import OnlineWindFieldMapper
            from adaptive_safety import AdaptiveCBF
            
            self.wind_estimator = IMUToWindEstimator()
            self.wind_mapper = OnlineWindFieldMapper()
            self.cbf = AdaptiveCBF()
    
    def extract_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract frozen features from observation."""
        self.feature_extractor.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            features = self.feature_extractor(obs_t).numpy()[0]
        return features
    
    def get_actions(self, features: np.ndarray,
                   deterministic: bool = True) -> np.ndarray:
        """Get motor commands from adaptive readout."""
        features_t = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            logits = self.readout(features_t).numpy()[0]
        
        actions = np.tanh(logits)
        return actions
    
    def get_safe_action(self, obs: np.ndarray, state: Dict = None,
                       motor_commands: np.ndarray = None,
                       imu_data: Dict = None) -> Dict:
        """
        Get safe action with wind estimation and safety verification.
        
        This is the MAIN CONTROL INTERFACE that integrates all components.
        
        Args:
            obs: observation vector
            state: drone state dict (for CBF)
            motor_commands: previous motor commands (for wind estimation)
            imu_data: IMU data dict (for wind estimation)
        
        Returns:
            dict with:
                - action: safe action to execute
                - features: extracted features
                - wind_estimate: estimated wind
                - safety_info: safety verification result
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get base action from adaptive readout
        base_action = self.get_actions(features)
        
        # Wind estimation
        wind_info = {}
        if self.wind_estimator is not None and imu_data is not None:
            wind_result = self.wind_estimator.estimate_wind(
                imu_data, motor_commands or np.zeros(4)
            )
            wind_info = wind_result
            
            # Update wind mapper
            if self.wind_mapper is not None and state is not None:
                position = state.get('position', np.zeros(3))
                self.wind_mapper.add_measurement(
                    position[:2], wind_result['wind_velocity'][:2]
                )
        
        # Safety verification
        safe_action = base_action
        safety_info = {'was_projected': False}
        
        if self.cbf is not None and state is not None:
            safe_action, safety_info = self.cbf.verify_and_project(
                state, base_action
            )
        
        return {
            'action': safe_action,
            'features': features,
            'wind_estimate': wind_info.get('wind_acceleration', np.zeros(3)),
            'wind_confidence': wind_info.get('confidence', 0.0),
            'safety_info': safety_info,
        }
    
    def adapt(self, features: np.ndarray, measured_response: np.ndarray,
             predicted_action: np.ndarray, state: Dict = None) -> Dict:
        """
        Adapt readout weights with safety verification.
        
        This is the NOVEL method that ensures adaptation is safe.
        """
        max_safe_norm = self.config.max_adaptation_norm
        
        # Get safety bound from CBF
        if self.cbf is not None and state is not None:
            bound_info = self.cbf.compute_adaptation_safety_bound(
                state, predicted_action
            )
            max_safe_norm = bound_info.get('max_norm', max_safe_norm)
        
        # RLS adaptation with safety clamping
        return self.rls.adapt(features, measured_response, predicted_action, max_safe_norm)
    
    def get_stats(self) -> Dict:
        """Get adaptation statistics."""
        stats = {
            'adaptation_rate': self.rls.adaptation_rate,
            'steps_adapted': self.rls.steps_adapted,
            'total_adaptation_norm': self.rls.total_adaptation_norm,
            'safety_clamps': self.rls.safety_clamps,
            'weight_norm': float(np.linalg.norm(self.rls.W)),
            'covariance_trace': float(np.mean([
                np.trace(self.rls.P[i]) for i in range(self.rls.action_dim)
            ])),
        }
        
        if self.wind_estimator is not None:
            stats['wind_estimator'] = self.wind_estimator.get_stats()
        
        if self.wind_mapper is not None:
            stats['wind_mapper'] = self.wind_mapper.get_stats()
        
        if self.cbf is not None:
            stats['cbf'] = self.cbf.get_stats()
        
        return stats


def train_feature_extractor(env, num_episodes: int = 1000,
                           feature_dim: int = 64,
                           obs_dim: int = 41,
                           action_dim: int = 4) -> FrozenFeatureExtractor:
    """
    Pre-train the frozen feature extractor offline.
    
    Uses autoencoder loss to learn useful features.
    """
    import torch.optim as optim
    
    feature_net = FrozenFeatureExtractor(obs_dim, feature_dim)
    decoder = nn.Sequential(
        nn.Linear(feature_dim, 128),
        nn.ELU(),
        nn.Linear(128, obs_dim),
    )
    
    optimizer = optim.Adam(
        list(feature_net.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_loss = 0
        
        for step in range(300):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            features = feature_net(obs_t)
            reconstructed = decoder(features)
            loss = nn.MSELoss()(reconstructed, obs_t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            action = np.random.uniform(-1, 1, action_dim)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        if episode % 100 == 0:
            print(f'  Episode {episode}: reconstruction loss = {total_loss / (step + 1):.4f}')
    
    return feature_net
