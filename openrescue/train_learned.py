"""
OpenRescue — Train a real learned baseline on the OpenRescue environment
========================================================================

Purpose
-------
Every learned checkpoint in this repo (``ppo_best.pt``, ``ippo_*``, ``mappo_*``,
``gat_*``) was trained on the **wildfire perimeter task**, not on OpenRescue.
That made the earlier "learned vs hand-crafted" comparisons meaningless — the
policy and the baselines were solving different problems, and the wildfire
checkpoints turned out to be degenerate (hover collapse / axis lock) anyway.

This module trains a genuine actor for ``OpenRescueEnv`` itself, so the learned
policy can be measured against ``random`` / ``frontier`` / ``ig`` / ``resilient``
on the actual benchmark protocol (Failure Levels 1–5, 200 steps, IQM + CI).

Reward design (documented choices)
----------------------------------
The env's built-in reward is shared and sparse (POI discovery + global coverage
drift).  Pure PPO on it is hard to learn from, and a naive run tends to collapse
toward hovering (it costs no energy and the shared reward accrues regardless).
We therefore shape the training signal with per-agent, physically observable
terms and clearly record them so they can be reported in the paper:

    r_shaped = env_reward
             + 1.0 * (new cells the agent visited this step)      # coverage credit
             - 8.0 * (agent crashed into a hidden obstacle)       # survival credit

Evaluation is always on the *unshaped* env metrics via ``episode_summary()``
(the same numbers reported for the hand-crafted baselines), never on the
shaped reward.

Usage
-----
CPU (this machine, ~0.3 s/episode — a smoke or a modest run):

    python -m openrescue.train_learned --episodes 400 --eval-seeds 3

Longer CPU run (≈15 min for 2500 episodes):

    python -m openrescue.train_learned --episodes 2500 --update-every 2

Paper-grade (GPU / Kaggle): raise ``--episodes`` to 10_000+ and optionally add
a level curriculum; the script is deterministic per seed and logs a JSON of the
training curve plus a full IQM comparison table afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .environment import OpenRescueEnv
from .metrics import aggregate_runs
from .policies import make_policy

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Actor-critic (same MLP shape family as the other PPO nets in the repo)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """obs -> [256 ReLU LN] -> [128 ReLU LN] -> policy logits(5) + value(1)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden1), nn.ReLU(), nn.LayerNorm(hidden1),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.LayerNorm(hidden2),
        )
        self.policy_head = nn.Linear(hidden2, act_dim)
        self.value_head = nn.Linear(hidden2, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def _sample(logits, value, deterministic: bool):
    dist = torch.distributions.Categorical(logits=logits)
    if deterministic:
        a = logits.argmax(dim=-1)
        return a, dist.log_prob(a), value, None
    a = dist.sample()
    return a, dist.log_prob(a), value, dist


# ---------------------------------------------------------------------------
# Benchmark-compatible wrapper for a trained net (acts like policies.py)
# ---------------------------------------------------------------------------

class LearnedPolicy:
    """Deterministic wrapper exposing the benchmark's act(obs, info, rng)."""

    name = 'learned'

    def __init__(self, net: nn.Module):
        self.net = net
        self.net.eval()

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: dict, rng):
        with torch.no_grad():
            o = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            logits, _ = self.net(o)
            return logits.argmax(dim=-1).numpy().astype(np.int32)


# ---------------------------------------------------------------------------
# One training episode -> per-agent transition streams + shaped rewards
# ---------------------------------------------------------------------------

def rollout_episode(env: OpenRescueEnv, net: ActorCritic, gamma: float,
                    gae_lambda: float, visit_bonus: float, crash_penalty: float):
    """Run one episode, returning tensors ready for a PPO update.

    Transitions are stored per agent and cut at the moment the agent grounds
    (crashes or runs out of battery), so value bootstrapping never crosses a
    grounded boundary.  Shaped reward is recorded separately from env reward.
    """
    n = env.n_drones
    obs, info = env.reset(seed=None)  # env already holds its own seeded rng
    prev_visited = [len(env.drones[i]['visited']) for i in range(n)]
    prev_grounded = [d['grounded'] for d in env.drones]

    streams: list[list[dict]] = [[] for _ in range(n)]
    ep_env_reward = np.zeros(n)
    ep_shaped_reward = np.zeros(n)

    for _ in range(env.max_steps):
        o = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        logits, value = net(o)
        acts, logps, _, dist = _sample(logits, value, deterministic=False)

        actions = acts.numpy()
        obs2, rew, term, trunc, info2 = env.step(actions)
        env_done = bool(np.all(term)) or bool(np.all(trunc))

        grounded_now = [d['grounded'] for d in env.drones]
        for i in range(n):
            if prev_grounded[i]:
                continue  # already cut
            new_visits = max(0, len(env.drones[i]['visited']) - prev_visited[i])
            crash_now = grounded_now[i] and info2['agents'][i].get('cause') == 'crash'
            shaped = float(rew[i]) + visit_bonus * new_visits
            if crash_now:
                shaped -= crash_penalty
            done = grounded_now[i] or env_done
            streams[i].append({
                'obs': obs[i].copy(), 'act': int(actions[i]),
                'logp': float(logps[i].detach()), 'val': float(value[i].detach()),
                'rew': shaped, 'done': done,
            })
            ep_env_reward[i] += float(rew[i])
            ep_shaped_reward[i] += shaped
            prev_visited[i] = len(env.drones[i]['visited'])
        prev_grounded = grounded_now
        obs = obs2
        if env_done:
            break

    # GAE per stream (walk backwards, done flag belongs to the NEXT state)
    obs_b, act_b, logp_b, adv_b, ret_b = [], [], [], [], []
    for st in streams:
        if not st:
            continue
        gae = 0.0
        v_next = 0.0
        for t in range(len(st) - 1, -1, -1):
            tr = st[t]
            delta = tr['rew'] + gamma * v_next * (1.0 - tr['done']) - tr['val']
            gae = delta + gamma * gae_lambda * (1.0 - tr['done']) * gae
            v_next = tr['val']
            obs_b.append(tr['obs']); act_b.append(tr['act'])
            logp_b.append(tr['logp']); adv_b.append(gae)
            ret_b.append(gae + tr['val'])
    return {
        'obs': np.asarray(obs_b, dtype=np.float32),
        'act': np.asarray(act_b, dtype=np.int64),
        'logp': np.asarray(logp_b, dtype=np.float32),
        'adv': np.asarray(adv_b, dtype=np.float32),
        'ret': np.asarray(ret_b, dtype=np.float32),
        'env_reward': float(ep_env_reward.mean()),
        'shaped_reward': float(ep_shaped_reward.mean()),
        'steps': env.step_count,
    }


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(net: ActorCritic, opt, batch: dict, n_epochs: int, minibatch: int,
               clip_eps: float, value_coef: float, entropy_coef: float):
    obs = torch.from_numpy(batch['obs'])
    act = torch.from_numpy(batch['act'])
    old_logp = torch.from_numpy(batch['logp'])
    adv = torch.from_numpy(batch['adv'])
    ret = torch.from_numpy(batch['ret'])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idx = np.arange(obs.shape[0])
    total_loss = 0.0
    n_updates = 0
    for _ in range(n_epochs):
        np.random.shuffle(idx)
        for s in range(0, obs.shape[0], minibatch):
            mb = idx[s:s + minibatch]
            logits, value = net(obs[mb])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[mb])
            ratio = torch.exp(logp - old_logp[mb])
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = -torch.min(ratio * adv[mb], clipped * adv[mb]).mean()
            value_loss = nn.functional.mse_loss(value, ret[mb])
            entropy_loss = -dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            total_loss += float(loss)
            n_updates += 1
    return total_loss / max(n_updates, 1)


# ---------------------------------------------------------------------------
# Evaluation on the exact benchmark protocol (identical to run_episode)
# ---------------------------------------------------------------------------

def evaluate(net: ActorCritic, levels, seeds, steps=200, env_kwargs=None,
             baselines=('random', 'frontier', 'resilient', 'ig'), verbose=True):
    """Fresh-episode IQM comparison: learned vs the hand-crafted baselines."""
    env_kwargs = env_kwargs or {}
    policies = list(baselines) + ['learned']
    learned = LearnedPolicy(net)
    results: dict = {}
    for pol in policies:
        for lvl in levels:
            runs = []
            for seed in seeds:
                env = OpenRescueEnv(failure_level=lvl, max_steps=steps,
                                    seed=seed, **env_kwargs)
                policy = learned if pol == 'learned' else make_policy(pol)
                obs, info = env.reset(seed=seed)
                if hasattr(policy, 'reset'):
                    policy.reset()
                rng = np.random.default_rng(seed)
                for _ in range(steps):
                    actions = policy.act(obs, info, rng)
                    obs, _, term, trunc, info = env.step(actions)
                    if bool(np.all(term)) or bool(np.all(trunc)):
                        break
                runs.append(env.episode_summary())
            results[(pol, lvl)] = aggregate_runs(runs)
            if verbose:
                a = results[(pol, lvl)]
                print(f"  {pol:<10} L{lvl}  coverage={a['coverage']['iqm']:.3f} "
                      f"pois={a['pois_ratio']['iqm']:.3f} lost={a['lost']['iqm']:.2f} "
                      f"mean_r={a['mean_r']['iqm']:.3f}", flush=True)
    return results


def _main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--episodes', type=int, default=600)
    ap.add_argument('--update-every', type=int, default=2, help='episodes per PPO update')
    ap.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                    help='failure levels cycled during training')
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--gae-lambda', type=float, default=0.95)
    ap.add_argument('--visit-bonus', type=float, default=1.0)
    ap.add_argument('--crash-penalty', type=float, default=8.0)
    ap.add_argument('--eval-seeds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--checkpoint', default='openrescue_learned_ppo.pt')
    ap.add_argument('--log', default='openrescue_learned_log.json')
    ap.add_argument('--eval-json', default='openrescue_learned_eval.json')
    args = ap.parse_args()

    env = OpenRescueEnv(seed=0)
    net = ActorCritic(obs_dim=env.obs_dim, act_dim=env.act_dim)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    levels = [int(np.clip(l, 1, 5)) for l in args.levels]
    history = {'shaped': [], 'env_reward': [], 'coverage': [], 'pois': []}
    best_shaped = -1e18
    t0 = time.time()
    buf = {'obs': [], 'act': [], 'logp': [], 'adv': [], 'ret': []}
    episodes_since_update = 0

    print(f"Training PPO on OpenRescueEnv | {args.episodes} eps | "
          f"levels {levels} | obs {env.obs_dim} | act {env.act_dim}", flush=True)
    for ep in range(args.episodes):
        level = levels[ep % len(levels)]
        env = OpenRescueEnv(failure_level=level, seed=args.seed * 10_000 + ep)
        seg = rollout_episode(env, net, args.gamma, args.gae_lambda,
                              args.visit_bonus, args.crash_penalty)
        for k in buf:
            buf[k].append(seg[k])
        episodes_since_update += 1
        history['shaped'].append(seg['shaped_reward'])
        history['env_reward'].append(seg['env_reward'])
        history['coverage'].append(env.coverage)
        history['pois'].append(int(env.poi_found.sum()))

        if episodes_since_update >= args.update_every:
            cat = {k: np.concatenate(v) if v else np.zeros((0,), dtype=np.float32)
                   for k, v in buf.items()}
            cat['act'] = cat['act'].astype(np.int64)
            if cat['obs'].shape[0] > 16:
                ppo_update(net, opt, cat, n_epochs=3, minibatch=512,
                           clip_eps=0.2, value_coef=0.5, entropy_coef=0.01)
            for k in buf:
                buf[k] = []
            episodes_since_update = 0

        if (ep + 1) % 100 == 0:
            w = history['shaped'][-100:]
            mean_w = float(np.mean(w))
            print(f"  ep {ep + 1:>5}/{args.episodes}  shaped(100)={mean_w:6.2f} "
                  f"env_rew(100)={np.mean(history['env_reward'][-100:]):5.2f} "
                  f"cov(100)={np.mean(history['coverage'][-100:]):.3f} "
                  f"pois(100)={np.mean(history['pois'][-100:]):.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if mean_w > best_shaped:
                best_shaped = mean_w
                torch.save({'net': net.state_dict(), 'episode': ep + 1,
                            'obs_dim': env.obs_dim, 'act_dim': env.act_dim},
                           args.checkpoint)

    # flush any partial buffer
    if buf['obs']:
        cat = {k: np.concatenate(v) for k, v in buf.items()}
        cat['act'] = cat['act'].astype(np.int64)
        if cat['obs'].shape[0] > 16:
            ppo_update(net, opt, cat, n_epochs=3, minibatch=512,
                       clip_eps=0.2, value_coef=0.5, entropy_coef=0.01)

    torch.save({'net': net.state_dict(), 'episode': args.episodes,
                'obs_dim': env.obs_dim, 'act_dim': env.act_dim}, args.checkpoint)
    log = {'args': vars(args), 'runtime_s': round(time.time() - t0, 1),
           'history': history}
    with open(args.log, 'w') as f:
        json.dump(log, f)
    print(f"\nSaved checkpoint {args.checkpoint} | training log {args.log} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # Load best if it beat the final net
    best_sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    net.load_state_dict(best_sd['net'])
    print("\nEvaluation on fresh episodes, Levels 1-5 (IQM):", flush=True)
    results = evaluate(net, levels=list(range(1, 6)),
                       seeds=list(range(args.eval_seeds)))
    out = {(f"{p}__L{l}"): v for (p, l), v in results.items()}
    with open(args.eval_json, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved evaluation -> {args.eval_json}")


if __name__ == '__main__':
    _main()
