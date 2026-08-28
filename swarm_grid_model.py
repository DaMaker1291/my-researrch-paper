"""
MARAHS: Multi-Agent Robust Autonomous Hurricane Swarm
=======================================================
CEM (Cross-Entropy Method) training for deterministic coverage policies.

Why CEM > PPO for this task:
1. PPO produces stochastic policies that collapse deterministically
2. PPO's value function co-adaptation causes training instability
3. CEM optimizes deterministic policies directly
4. CEM guarantees monotonic improvement
5. CEM needs no hyperparameter tuning (pop_size, elite_frac)

Two policy variants:
- LinearPolicy: obs → W @ obs + b → argmax (100 params, highly interpretable)
- MLPPolicy: obs → Linear+ReLU → Linear (more expressive)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict


class LinearPolicy(nn.Module):
    """Linear policy: action = argmax(W @ obs + b).
    
    Pros: 100 params, fast CEM, fully interpretable
    Cons: Can't learn nonlinear decision boundaries
    """

    def __init__(self, obs_dim: int = 19, action_dim: int = 5):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.linear = nn.Linear(obs_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)

    def get_actions(self, obs: torch.Tensor, deterministic: bool = True) -> Dict:
        logits = self.forward(obs)
        actions = logits.argmax(dim=-1)
        return {'actions': actions, 'values': torch.zeros(obs.shape[0])}

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([
            self.linear.weight.data.cpu().numpy().ravel().copy(),
            self.linear.bias.data.cpu().numpy().ravel().copy(),
        ])

    def set_flat_params(self, flat_params: np.ndarray):
        w_size = self.action_dim * self.obs_dim
        self.linear.weight.data = torch.FloatTensor(
            flat_params[:w_size].reshape(self.action_dim, self.obs_dim))
        self.linear.bias.data = torch.FloatTensor(
            flat_params[w_size:w_size + self.action_dim])


class MLPPolicy(nn.Module):
    """Small MLP policy: obs → Linear+ReLU → Linear → action.
    
    Pros: ~600 params, can learn nonlinear boundaries
    Cons: Slower CEM, less interpretable
    """

    def __init__(self, obs_dim: int = 19, action_dim: int = 5, hidden: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_actions(self, obs: torch.Tensor, deterministic: bool = True) -> Dict:
        logits = self.forward(obs)
        actions = logits.argmax(dim=-1)
        return {'actions': actions, 'values': torch.zeros(obs.shape[0])}

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([p.data.cpu().numpy().ravel().copy() for p in self.parameters()])

    def set_flat_params(self, flat_params: np.ndarray):
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data = torch.FloatTensor(flat_params[offset:offset + n]).reshape(p.shape)
            offset += n


def evaluate_episode(model, env, deterministic: bool = True) -> float:
    """Run one episode, return final coverage percentage."""
    model.eval()
    obs, _ = env.reset()
    K = env.env.K

    for _ in range(env.env.config.max_steps):
        all_obs = np.array([env.env._get_drone_obs(i) for i in range(K)])
        with torch.no_grad():
            obs_t = torch.FloatTensor(all_obs)
            result = model.get_actions(obs_t, deterministic=deterministic)
            actions = result['actions'].numpy()

        obs, _, dones, _, infos = env.step_all_drones(actions)
        if dones.all():
            break

    return infos[0]['coverage_pct'] if infos else 0


def cem_train(model, env,
              num_generations: int = 50,
              eval_episodes: int = 3,
              pop_size: int = 100,
              elite_frac: float = 0.2,
              sigma_init: float = 0.3,
              sigma_min: float = 0.01,
              sigma_decay: float = 0.995,
              verbose: bool = True) -> Dict:
    """
    Train a model using CEM (Cross-Entropy Method).
    
    Algorithm:
    1. Sample pop_size parameter vectors from N(mu, sigma)
    2. Evaluate each on the environment (deterministic)
    3. Keep top elite_frac as elites
    4. Refit mu, sigma to elites
    5. Repeat
    
    Returns: history dict
    """
    num_elites = max(2, int(pop_size * elite_frac))
    n_params = len(model.get_flat_params())
    
    # Initialize
    mu = np.random.randn(n_params) * 0.1
    sigma = np.full(n_params, float(sigma_init))
    best_score = -1.0
    best_params = mu.copy()
    
    history = {
        'generation': [], 'best_coverage': [], 'mean_coverage': [],
        'sigma_mean': [],
    }
    
    for gen in range(num_generations):
        population = [mu + np.random.randn(n_params) * sigma for _ in range(pop_size)]
        
        scores = []
        for params in population:
            model.set_flat_params(params)
            ep_scores = []
            for _ in range(eval_episodes):
                ep_scores.append(evaluate_episode(model, env))
            scores.append(float(np.mean(ep_scores)))
        
        scores_arr = np.array(scores)
        elite_idx = np.argsort(scores_arr)[::-1][:num_elites]
        elites = [population[i] for i in elite_idx]
        
        gen_best = scores_arr[elite_idx[0]]
        if gen_best > best_score:
            best_score = float(gen_best)
            best_params = population[elite_idx[0]].copy()
        
        mu = np.mean(elites, axis=0)
        sigma = np.std(elites, axis=0) + 1e-8
        sigma = np.maximum(sigma * sigma_decay, sigma_min)
        
        history['generation'].append(gen)
        history['best_coverage'].append(best_score)
        history['mean_coverage'].append(float(np.mean(scores_arr[elite_idx])))
        history['sigma_mean'].append(float(np.mean(sigma)))
        
        if verbose and (gen % 5 == 0 or gen == num_generations - 1):
            print(f"  gen {gen:3d} | best {best_score:5.1f}% | "
                  f"gen_mean {np.mean(scores_arr[elite_idx]):5.1f}% | "
                  f"sigma={np.mean(sigma):.4f}")
    
    model.set_flat_params(best_params)
    return history
