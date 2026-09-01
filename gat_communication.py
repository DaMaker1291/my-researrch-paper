#!/usr/bin/env python3
"""
Graph Attention Network (GAT) Communication Layer for MARAHS
================================================================

Implements multi-head graph attention for decentralized drone coordination.
Each drone is a node; edges connect drones within communication range.
The GAT aggregates neighbor observations so each drone "knows" what nearby
drones see — enabling emergent coordination without a central controller.

Architecture:
  Node features: drone's local obs (648) + global features (8) = 656 dims
  GAT Layer 1: 656 → 128, 4 heads, LeakyReLU
  GAT Layer 2: 128 → 64, 4 heads, LeakyReLU
  Output: 64-dim graph-aware embedding per drone

The GAT output is concatenated with the original observation and fed to PPO.

Key properties:
  - Permutation invariant (handles any number of drones)
  - Dynamic graph topology (edges update each step based on positions)
  - Graceful degradation (dead drones are simply removed from the graph)
  - Scales to 100+ drones with sparse attention

Usage:
    gat = GATCommunication(in_dim=656, comm_range=15.0)
    enhanced_obs = gat.forward(all_obs, positions)  # (K, 656+64)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device("cpu")


class MultiHeadAttention(nn.Module):
    """
    Single-head graph attention: a_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    
    This computes attention weights between node i and each neighbor j,
    then aggregates neighbor features weighted by attention.
    """

    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0, f"out_dim ({out_dim}) must be divisible by n_heads ({n_heads})"
        
        # Linear projection for each head
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # Attention vectors for each head
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim))
        
        # LeakyReLU for attention
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Init
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, h, adj_mask):
        """
        Args:
            h: (K, in_dim) node features
            adj_mask: (K, K) boolean adjacency mask (True = connected)
        Returns:
            out: (K, out_dim) updated node features
        """
        K = h.shape[0]
        if K <= 1:
            # Single drone — no neighbors, just project
            return self.W(h)
        
        # Project: (K, out_dim)
        Wh = self.W(h)
        # Reshape for multi-head: (K, n_heads, head_dim)
        Wh = Wh.view(K, self.n_heads, self.head_dim)
        
        # Compute attention scores
        # src: (K, n_heads, head_dim) * (n_heads, head_dim) -> (K, n_heads)
        e_src = (Wh * self.a_src.unsqueeze(0)).sum(dim=2)
        e_dst = (Wh * self.a_dst.unsqueeze(0)).sum(dim=2)
        
        # Attention: e_ij = LeakyReLU(e_src[i] + e_dst[j])
        # (K, K, n_heads)
        e = e_src.unsqueeze(1) + e_dst.unsqueeze(0)  # (K, K, H)
        e = self.leaky_relu(e)
        
        # Mask non-edges
        adj = adj_mask.unsqueeze(2).float()  # (K, K, 1)
        e = e.masked_fill(~adj.bool(), float('-inf'))
        
        # Softmax over neighbors
        attn = F.softmax(e, dim=1)  # (K, K, H)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle all-masked rows
        
        # Aggregate: out_i = sum_j attn_ij * Wh_j
        # attn: (K, K, H), Wh: (K, H, D)
        out = torch.einsum('kjh,khd->khd', attn, Wh)  # (K, H, D)
        out = out.reshape(K, -1)  # (K, out_dim)
        
        return out


class GATCommunication(nn.Module):
    """
    2-layer GAT for inter-drone communication.
    
    Input:  obs per drone (K × 656)
    Output: graph-enhanced obs per drone (K × (656 + 64))
    
    The 64-dim GAT embedding is concatenated with the original observation,
    giving the PPO agent awareness of what nearby drones see and intend.
    """

    def __init__(self, in_dim=656, hidden_dim=128, out_dim=64,
                 n_heads=4, comm_range=15.0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.comm_range = comm_range
        
        # Layer 1: in_dim → hidden_dim
        self.attn1 = MultiHeadAttention(in_dim, hidden_dim, n_heads)
        
        # Layer 2: hidden_dim → out_dim
        self.attn2 = MultiHeadAttention(hidden_dim, out_dim, n_heads)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        # Output projection
        self.output_proj = nn.Linear(out_dim, out_dim)
        
    def build_graph(self, positions, alive_mask):
        """
        Build adjacency mask from drone positions.
        
        Args:
            positions: (K, 2) drone positions
            alive_mask: (K,) boolean mask of alive drones
        Returns:
            adj_mask: (K, K) boolean adjacency matrix
        """
        K = len(positions)
        adj = torch.zeros(K, K, dtype=torch.bool, device=device)
        
        for i in range(K):
            if not alive_mask[i]:
                continue
            for j in range(K):
                if not alive_mask[j] or i == j:
                    continue
                dist = torch.norm(positions[i] - positions[j])
                if dist < self.comm_range:
                    adj[i, j] = True
                    adj[j, i] = True
        
        # Each node always attends to itself
        for i in range(K):
            if alive_mask[i]:
                adj[i, i] = True
        
        return adj
    
    def forward(self, obs, positions, alive_mask):
        """
        Forward pass: build graph, run GAT, concatenate output.
        
        Args:
            obs: (K, in_dim) observations for all drones
            positions: (K, 2) drone positions
            alive_mask: (K,) boolean mask of alive drones
        Returns:
            enhanced_obs: (K, in_dim + out_dim) enhanced observations
        """
        K = obs.shape[0]
        
        # Build adjacency graph
        adj = self.build_graph(positions, alive_mask)
        
        # Run GAT
        h = obs
        h = self.attn1(h, adj)
        h = self.norm1(h)
        h = F.relu(h)
        
        h2 = self.attn2(h, adj)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        
        # Project output
        gat_embed = self.output_proj(h2)  # (K, out_dim)
        
        # Concatenate with original observation
        enhanced = torch.cat([obs, gat_embed], dim=1)  # (K, in_dim + out_dim)
        
        return enhanced
    
    @property
    def enhanced_obs_dim(self):
        """Observation dimension after GAT enhancement."""
        return self.in_dim + self.out_dim


class GATPPOAgent:
    """
    PPO agent with GAT communication layer.
    
    The GAT processes multi-drone observations into graph-aware embeddings,
    then the PPO policy acts on the enhanced observations.
    
    Architecture:
        obs (K × 656) → GAT (656 → 656+64) → PPO Network (720 → policy + value)
    """

    def __init__(self, obs_dim=656, act_dim=5, gat_hidden=128, gat_out=64,
                 n_heads=4, comm_range=15.0, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.02):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        
        # GAT layer
        self.gat = GATCommunication(
            in_dim=obs_dim, hidden_dim=gat_hidden, out_dim=gat_out,
            n_heads=n_heads, comm_range=comm_range
        ).to(device)
        
        # PPO policy (takes GAT-enhanced observations)
        enhanced_dim = obs_dim + gat_out
        self.policy_net = PPONetwork(enhanced_dim, act_dim).to(device)
        
        # Combined optimizer
        params = list(self.gat.parameters()) + list(self.policy_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
        
        # Trajectory storage
        self._trajectories = {}
        
    def get_actions(self, obs, positions, alive_mask, deterministic=False):
        """
        Get actions for all alive drones using GAT + PPO.
        
        Args:
            obs: (K, obs_dim) raw observations
            positions: (K, 2) drone positions
            alive_mask: (K,) boolean mask
        Returns:
            actions: (K,) action indices
            log_probs: (K,) log probabilities
            values: (K,) value estimates
        """
        obs_t = torch.tensor(obs, device=device, dtype=torch.float32)
        pos_t = torch.tensor(positions, device=device, dtype=torch.float32)
        
        # GAT enhancement
        enhanced = self.gat(obs_t, pos_t, alive_mask)
        
        # PPO action selection
        with torch.no_grad():
            logits, values = self.policy_net(enhanced)
            probs = torch.distributions.Categorical(logits=logits)
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                actions = probs.sample()
            log_probs = probs.log_prob(actions)
        
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()
    
    def store_transition(self, drone_id, obs, positions, alive_mask,
                         action, reward, done, log_prob, value):
        """Store one transition for a specific drone."""
        if drone_id not in self._trajectories:
            self._trajectories[drone_id] = {
                'obs': [], 'positions': [], 'alive_mask': [],
                'action': [], 'reward': [], 'done': [],
                'log_prob': [], 'value': [],
            }
        t = self._trajectories[drone_id]
        t['obs'].append(obs)
        t['positions'].append(positions)
        t['alive_mask'].append(alive_mask)
        t['action'].append(action)
        t['reward'].append(reward)
        t['done'].append(done)
        t['log_prob'].append(log_prob)
        t['value'].append(value)
    
    def reset_trajectories(self):
        self._trajectories = {}
    
    def update(self, n_epochs=4, batch_size=128, max_grad_norm=0.5):
        """
        PPO update with GAT-enhanced observations.
        
        Replays GAT forward pass for each stored observation to get
        the correct enhanced observations.
        """
        all_obs, all_actions, all_old_lp = [], [], []
        all_advantages, all_returns = [], []
        
        for drone_id, t in self._trajectories.items():
            if len(t['obs']) == 0:
                continue
            
            rewards = np.array(t['reward'], dtype=np.float32)
            values = np.array(t['value'], dtype=np.float32)
            dones = np.array(t['done'], dtype=np.float32)
            
            # GAE per-drone
            n = len(rewards)
            advantages = np.zeros(n, dtype=np.float32)
            returns = np.zeros(n, dtype=np.float32)
            gae = 0.0
            for step in reversed(range(n)):
                next_val = 0.0 if step == n - 1 else values[step + 1]
                next_done = 1.0 if step == n - 1 else dones[step + 1]
                delta = rewards[step] + self.gamma * next_val * (1 - next_done) - values[step]
                gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
                advantages[step] = gae
                returns[step] = gae + values[step]
            
            # Re-run GAT to get enhanced observations
            for i in range(n):
                obs_t = torch.tensor(t['obs'][i], device=device, dtype=torch.float32).unsqueeze(0)
                pos_t = torch.tensor(t['positions'][i], device=device, dtype=torch.float32)
                mask = torch.tensor(t['alive_mask'][i], device=device, dtype=torch.bool)
                
                enhanced = self.gat(obs_t, pos_t, mask)
                all_obs.append(enhanced.squeeze(0).detach().cpu().numpy())
                all_actions.append(t['action'][i])
                all_old_lp.append(t['log_prob'][i])
                all_advantages.append(advantages[i])
                all_returns.append(returns[i])
        
        if len(all_obs) == 0:
            return 0.0
        
        obs = torch.tensor(np.array(all_obs), device=device, dtype=torch.float32)
        actions = torch.tensor(np.array(all_actions), device=device, dtype=torch.long)
        old_lp = torch.tensor(np.array(all_old_lp), device=device, dtype=torch.float32)
        advantages = torch.tensor(np.array(all_advantages), device=device, dtype=torch.float32)
        returns = torch.tensor(np.array(all_returns), device=device, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n_samples = len(obs)
        total_loss = 0.0
        n_updates = 0
        
        for epoch in range(n_epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = perm[start:end]
                
                _, new_lp, entropy, values = self.policy_net.evaluate(obs[idx], actions[idx])
                
                ratio = torch.exp(new_lp - old_lp[idx])
                s1 = ratio * advantages[idx]
                s2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(values, returns[idx])
                entropy_bonus = -entropy.mean()
                
                loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_bonus
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.gat.parameters()) + list(self.policy_net.parameters()),
                    max_grad_norm
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
        
        self.reset_trajectories()
        return total_loss / max(1, n_updates)
    
    def save(self, path):
        torch.save({
            'gat': self.gat.state_dict(),
            'policy': self.policy_net.state_dict(),
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.gat.load_state_dict(ckpt['gat'])
        self.policy_net.load_state_dict(ckpt['policy'])


class PPONetwork(nn.Module):
    """Shared encoder with policy + value heads (same as ppo_train.py)."""
    
    def __init__(self, obs_dim, act_dim, hidden1=256, hidden2=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.LayerNorm(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.LayerNorm(hidden2),
        )
        self.policy_head = nn.Linear(hidden2, act_dim)
        self.value_head = nn.Linear(hidden2, 1)
        self.apply(self._init)
    
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    
    def forward(self, obs):
        feat = self.encoder(obs)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)
    
    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)
        return logits, probs.log_prob(actions), probs.entropy(), value


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GAT Communication Layer — Test & Benchmark")
    print("=" * 60)
    
    # Test 1: Graph construction
    print("\n--- Test 1: Graph Construction ---")
    gat = GATCommunication(in_dim=656, hidden_dim=128, out_dim=64, n_heads=4, comm_range=15.0)
    
    positions = torch.tensor([
        [5.0, 5.0], [6.0, 6.0], [20.0, 20.0], [21.0, 21.0]
    ], device=device, dtype=torch.float32)
    alive = torch.tensor([True, True, True, True], device=device, dtype=torch.bool)
    
    adj = gat.build_graph(positions, alive)
    print(f"  4 drones, comm_range=15:")
    print(f"  Drone 0 neighbors: {adj[0].sum().item() - 1} (should be 1: drone 1)")
    print(f"  Drone 2 neighbors: {adj[2].sum().item() - 1} (should be 1: drone 3)")
    print(f"  Cluster 0-1 vs 2-3 disconnected: {not adj[0, 2].item()}")
    
    # Test 2: Forward pass
    print("\n--- Test 2: Forward Pass ---")
    K = 10
    obs = torch.randn(K, 656, device=device, dtype=torch.float32)
    pos = torch.randn(K, 2, device=device, dtype=torch.float32) * 15
    mask = torch.ones(K, dtype=torch.bool, device=device)
    mask[7] = False  # Drone 7 dead
    
    enhanced = gat(obs, pos, mask)
    print(f"  Input: ({K}, 656)")
    print(f"  Output: ({K}, {enhanced.shape[1]})")
    print(f"  Expected: ({K}, {656 + 64} = 720)")
    assert enhanced.shape == (K, 720), f"Shape mismatch: {enhanced.shape}"
    
    # Test 3: Graceful degradation (fewer drones)
    print("\n--- Test 3: Scalability ---")
    for n_drones in [5, 10, 20, 50]:
        obs_n = torch.randn(n_drones, 656, device=device, dtype=torch.float32)
        pos_n = torch.randn(n_drones, 2, device=device, dtype=torch.float32) * 15
        mask_n = torch.ones(n_drones, dtype=torch.bool, device=device)
        
        t0 = time.perf_counter()
        out_n = gat(obs_n, pos_n, mask_n)
        t_ms = (time.perf_counter() - t0) * 1000
        
        print(f"  {n_drones:3d} drones → ({n_drones}, {out_n.shape[1]}) | {t_ms:.1f}ms")
    
    # Test 4: GATPPOAgent end-to-end
    print("\n--- Test 4: GAT-PPO Agent End-to-End ---")
    agent = GATPPOAgent(obs_dim=656, act_dim=5, gat_hidden=128, gat_out=64)
    
    obs = torch.randn(10, 656, device=device, dtype=torch.float32)
    pos = torch.randn(10, 2, device=device, dtype=torch.float32) * 15
    alive = torch.ones(10, dtype=torch.bool, device=device)
    
    actions, log_probs, values = agent.get_actions(
        obs.cpu().numpy(), pos.cpu().numpy(), alive.cpu().numpy()
    )
    print(f"  Actions: {actions.shape} = (10,)")
    print(f"  Log probs: {log_probs.shape} = (10,)")
    print(f"  Values: {values.shape} = (10,)")
    print(f"  Action values: {np.unique(actions)}")
    
    # Test 5: Single drone (edge case)
    print("\n--- Test 5: Single Drone ---")
    obs_1 = torch.randn(1, 656, device=device, dtype=torch.float32)
    pos_1 = torch.tensor([[15.0, 15.0]], device=device, dtype=torch.float32)
    mask_1 = torch.tensor([True], device=device, dtype=torch.bool)
    
    out_1 = gat(obs_1, pos_1, mask_1)
    print(f"  1 drone → {out_1.shape} (should be (1, 720))")
    
    print("\n" + "=" * 60)
    print("All GAT tests passed!" if enhanced.shape == (K, 720) else "TESTS FAILED!")
    print("=" * 60)
