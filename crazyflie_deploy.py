#!/usr/bin/env python3
"""
Sim-to-real export + validation for the learned MARAHS grid actors
===================================================================

What this script actually is (corrected):
-----------------------------------------
The trained artifacts in this repo are *high-level grid policies*:
PPONetwork-style MLPs (obs -> 5 discrete scan actions) trained on the
wildfire perimeter env (``paper_ready_train.WildfireEnv``, obs 656) or the
kaggle_full_run env (obs 496), optionally fronted by a GAT communication
layer.  **None of them is a 6-D continuous wind-hold controller.**  A previous
version of this file fabricated that: it initialized random 6->32->4 weights,
"exported" them as a trained MARAHS policy, and its companion guide claimed
"holds position within +/-0.2 m at 25 m/s" — a number no code ever measured.
That fabrication is removed.

This script now does two honest things:

1. ``--export`` loads a **real checkpoint** (plain actor only: PPO/IPPO/MAPPO
   policy nets; GAT checkpoints are refused because the actor is only
   meaningful behind its runtime communication graph) and serializes the
   actual weights — encoder + LayerNorm + policy head, dimensions read from
   the checkpoint tensors — to a C header and (if ``onnx`` is installed) an
   ONNX model.  Nothing is exported unless it came from a ``.pt`` file on
   disk.

2. ``--benchmark`` measures the loaded actor against hand-crafted baselines
   (Random, Greedy-frontier) on the **task the checkpoint was trained for**
   (fresh episodes of the native env, identical seeds per policy), and writes
   the measured numbers to ``learned_vs_handcrafted.json``.  This is the
   deployment gate: if the learned actor does not beat the hand-crafted
   policies on its own task, there is no case for uploading it anywhere.

What this script does NOT do: it does not claim a physical position-hold
result.  Wind station-keeping at the actuator level is a flight-controller
problem (see ``macondo_hover.py``); turning a grid actor into an onboard
decision layer additionally requires an observation bridge (map telemetry ->
actor obs) and an action->velocity mapping that are not yet defined.  Until
those exist, the C header produced here is a weights artifact, not a
flight-ready binary.

Usage:
    python crazyflie_deploy.py --checkpoint ppo_best.pt --export
    python crazyflie_deploy.py --checkpoint ppo_best.pt --benchmark --episodes 6
"""
import argparse
import json
import os

import numpy as np
import torch

DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════
# 1. REAL CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════

def inspect_checkpoint(path):
    """Return (kind, obs_dim, act_dim) for a checkpoint, or raise.

    kind is 'actor' (standalone PPONetwork), 'mappo' (wrapped policy+critic
    dicts), 'gat' (needs runtime comm graph), or 'unknown'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: not a state-dict checkpoint (type {type(sd).__name__})")

    if "gat" in sd:
        kind, actor = "gat", sd.get("policy", {})
    elif "critic" in sd and "policy" in sd:
        kind, actor = "mappo", sd["policy"]
    elif "policy" in sd:
        kind, actor = "unknown-wrapped", sd["policy"]
    else:
        kind, actor = "actor", sd

    enc = actor.get("encoder.0.weight")
    head = actor.get("policy_head.weight")
    if enc is None or head is None:
        raise ValueError(
            f"{path}: no PPONetwork-style actor found "
            f"(encoder.0.weight / policy_head.weight missing; keys={list(sd.keys())[:8]})"
        )
    obs_dim = int(enc.shape[1])
    act_dim = int(head.shape[0])
    return kind, obs_dim, act_dim


def load_actor(path):
    """Load a plain PPONetwork actor from a checkpoint. GAT refused."""
    from ppo_train import PPONetwork

    kind, obs_dim, act_dim = inspect_checkpoint(path)
    if kind == "gat":
        raise ValueError(
            f"{path}: GAT checkpoint — its actor consumes graph-enhanced obs "
            f"and is meaningless without the runtime comm graph; C/ONNX export "
            f"of the standalone MLP would silently change the policy. Refusing."
        )
    sd = torch.load(path, map_location="cpu", weights_only=False)
    actor_sd = sd.get("policy", sd) if kind in ("mappo", "unknown-wrapped") else sd
    net = PPONetwork(obs_dim=obs_dim, act_dim=act_dim)
    net.load_state_dict(actor_sd)
    net.eval()
    return net, kind, obs_dim, act_dim


# ═══════════════════════════════════════════════════════════════
# 2. EXPORT OF REAL WEIGHTS
# ═══════════════════════════════════════════════════════════════

def export_onnx(net, obs_dim, path):
    """Export the real actor to ONNX. Fails loudly if onnx is not installed."""
    try:
        import torch.onnx  # noqa: F401  (import surfaces the onnx dependency)
        dummy = torch.randn(1, obs_dim, dtype=torch.float32)
        torch.onnx.export(
            net, dummy, path,
            input_names=["obs"], output_names=["logits"],
            dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=12,
        )
    except Exception as e:  # onnx missing or export failure
        raise RuntimeError(
            f"ONNX export failed ({type(e).__name__}: {e}). "
            f"The onnx package is required: pip install onnx"
        ) from e
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"ONNX export claimed success but {path} is missing/empty")
    print(f"Exported real actor -> ONNX: {path} ({os.path.getsize(path)} bytes)")


def _layer_norm_params(actor_sd, idx):
    """Return (gamma, beta) for LayerNorm at encoder.{idx}.weight/bias."""
    return (actor_sd[f"encoder.{idx}.weight"].numpy(),
            actor_sd[f"encoder.{idx}.bias"].numpy())


def export_c_header(net, obs_dim, act_dim, path, source_pt):
    """Serialize the *real* actor weights (incl. LayerNorm) to a C header."""
    actor_sd = net.state_dict()
    w1 = actor_sd["encoder.0.weight"].numpy()   # (256, obs_dim)
    b1 = actor_sd["encoder.0.bias"].numpy()
    ln1g, ln1b = _layer_norm_params(actor_sd, 2)
    w2 = actor_sd["encoder.3.weight"].numpy()   # (128, 256)
    b2 = actor_sd["encoder.3.bias"].numpy()
    ln2g, ln2b = _layer_norm_params(actor_sd, 5)
    wp = actor_sd["policy_head.weight"].numpy()  # (act_dim, 128)
    bp = actor_sd["policy_head.bias"].numpy()
    h1, h2 = w1.shape[0], w2.shape[0]

    def fmt(a):
        return ", ".join(f"{v:.9g}f" for v in a.ravel())

    lines = [
        "// Auto-generated from REAL checkpoint weights — do not edit.",
        f"// Source: {source_pt}",
        "// PPONetwork grid actor (MARAHS wildfire lineage):",
        "//   obs -> Linear(obs,256)+ReLU+LayerNorm -> Linear(256,128)+ReLU+LayerNorm",
        "//       -> policy_head(128, act) logits. LayerNorm eps = 1e-5.",
        "// NOTE: this is the high-level decision actor (grid obs -> discrete action).",
        "// It is NOT a low-level wind-hold controller; the observation bridge to",
        "// real telemetry is not yet defined (see module docstring).",
        "#pragma once",
        f"#define POLICY_INPUT_DIM {obs_dim}",
        f"#define POLICY_HIDDEN1_DIM {h1}",
        f"#define POLICY_HIDDEN2_DIM {h2}",
        f"#define POLICY_ACT_DIM {act_dim}",
        "#define POLICY_LN_EPS 1e-5f",
        "",
        f"static const float W1[{h1}][{obs_dim}] = {{",
    ]
    for i in range(h1):
        lines.append(f"    {{{fmt(w1[i])}}},")
    lines.append("};")
    lines.append(f"static const float b1[{h1}] = {{{fmt(b1)}}};")
    lines.append(f"static const float ln1_gamma[{h1}] = {{{fmt(ln1g)}}};")
    lines.append(f"static const float ln1_beta[{h1}] = {{{fmt(ln1b)}}};")
    lines.append("")
    lines.append(f"static const float W2[{h2}][{h1}] = {{")
    for i in range(h2):
        lines.append(f"    {{{fmt(w2[i])}}},")
    lines.append("};")
    lines.append(f"static const float b2[{h2}] = {{{fmt(b2)}}};")
    lines.append(f"static const float ln2_gamma[{h2}] = {{{fmt(ln2g)}}};")
    lines.append(f"static const float ln2_beta[{h2}] = {{{fmt(ln2b)}}};")
    lines.append("")
    lines.append(f"static const float Wp[{act_dim}][{h2}] = {{")
    for i in range(act_dim):
        lines.append(f"    {{{fmt(wp[i])}}},")
    lines.append("};")
    lines.append(f"static const float bp[{act_dim}] = {{{fmt(bp)}}};")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Exported real actor -> C header: {path} "
          f"({obs_dim}-{h1}-{h2}-{act_dim}, {os.path.getsize(path)} bytes)")


# ═══════════════════════════════════════════════════════════════
# 3. MEASURED LEARNED-VS-HANDCRAFTED BENCHMARK (native task)
# ═══════════════════════════════════════════════════════════════

def _rollout(env, act_fn, seed):
    """Run one episode; act_fn(drone_idx, obs, env) -> discrete action."""
    obs = env.reset(seed=seed)
    hist = np.zeros(5, dtype=int)
    for _ in range(env.max_steps):
        actions = np.zeros(env.n_drones, dtype=int)
        for i in range(env.n_drones):
            if env.drones[i]["alive"]:
                a = act_fn(i, obs[i], env)
                actions[i] = a
                hist[int(a)] += 1
        obs, _, dones, infos = env.step(actions)
        if all(dones):
            break
    return {
        "safety": sum(1 for d in env.drones if d["alive"]) / env.n_drones * 100.0,
        "total_coverage": float(infos[0].get("total_coverage", 0.0)),
        "cells": float(max((len(d["visited"]) for d in env.drones), default=0)),
        "alive_steps": float(max((d["alive_steps"] for d in env.drones), default=0)),
        "perimeter": float(np.mean([infos[i].get("perimeter_frac", 0.0)
                                     for i in range(env.n_drones)])),
        "fire_dist": float(np.mean([infos[i].get("fire_dist", 0.0)
                                     for i in range(env.n_drones)
                                     if infos[i].get("alive", False)])),
        "act_hist": [int(x) for x in hist],  # [hover, col+, col-, row+, row-]
    }


def _learned_act(net, obs_dim):
    def act_fn(i, obs, env):
        o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, _, _ = net.get_action(o, deterministic=True)
        return int(a[0])
    return act_fn


def _random_act(rng):
    def act_fn(i, obs, env):
        return int(rng.integers(0, 5))
    return act_fn


def _greedy_act():
    """Move toward the nearest unvisited cell (frontier heuristic)."""
    def act_fn(i, obs, env):
        pos = np.asarray(env.drones[i]["pos"], dtype=int)
        unvisited = np.argwhere(~env._visited_grid)
        if len(unvisited) == 0:
            return 0
        dists = np.abs(unvisited - pos).sum(axis=1)
        target = unvisited[np.argmin(dists)]
        d = np.sign(target - pos)
        # action_deltas: 0 hover | 1 [0,+1] | 2 [0,-1] | 3 [+1,0] | 4 [-1,0]
        # pos[0] is the first grid axis, pos[1] the second.
        if d[0] != 0:
            return 3 if d[0] > 0 else 4
        if d[1] != 0:
            return 1 if d[1] > 0 else 2
        return 0
    return act_fn


def benchmark(checkpoint, n_episodes, winds=(0.0, 12.0), out="learned_vs_handcrafted.json"):
    """Learned actor vs Random / Greedy on the actor's native env."""
    import paper_ready_train as prt

    net, kind, obs_dim, act_dim = load_actor(checkpoint)
    if obs_dim != prt.WildfireEnv().obs_dim:
        raise ValueError(
            f"{checkpoint}: actor expects obs_dim={obs_dim}, but the native "
            f"paper_ready_train env has obs_dim={prt.WildfireEnv().obs_dim}. "
            f"Cannot evaluate this checkpoint on that task — refusing rather "
            f"than silently mismatching observations."
        )

    policies = {
        "learned": lambda: _learned_act(net, obs_dim),
        "random": lambda: _random_act(np.random.default_rng(0)),
        "greedy_frontier": _greedy_act,
    }
    results = {"meta": {
        "checkpoint": checkpoint, "kind": kind, "obs_dim": obs_dim, "act_dim": act_dim,
        "task": "MARAHS wildfire perimeter (paper_ready_train.WildfireEnv, 30x30, 10 drones)",
        "n_episodes": n_episodes, "winds": list(winds),
        "note": "fresh episodes, identical seeds per policy; deterministic actions",
    }, "episodes": {}}
    for wind in winds:
        results["episodes"][str(wind)] = {}
        env = prt.WildfireEnv(grid=30, n_drones=10, max_steps=300, wind_speed=wind)
        for name, make in policies.items():
            acts = make()
            ep = [_rollout(env, acts, seed=10000 + e) for e in range(n_episodes)]
            agg = {k: float(np.mean([r[k] for r in ep])) for k in ep[0] if k != "act_hist"}
            agg.update({f"{k}_std": float(np.std([r[k] for r in ep])) for k in ep[0] if k != "act_hist"})
            agg["act_hist"] = np.mean([r["act_hist"] for r in ep], axis=0).round(1).tolist()
            results["episodes"][str(wind)][name] = agg
            print(f"  wind={wind:>5}: {name:<16} safety={agg['safety']:5.1f}% "
                  f"perimeter={agg['perimeter']:5.2f} coverage={agg['total_coverage']:5.1f} "
                  f"actions={agg['act_hist']}")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved measured results -> {out}")
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="ppo_best.pt",
                        help="Path to a real .pt checkpoint (default: ppo_best.pt)")
    parser.add_argument("--export", action="store_true",
                        help="Export the REAL actor weights to C header (+ ONNX if installed)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure learned actor vs Random/Greedy on its native task")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--out-header", default="crazyflie_policy.h")
    parser.add_argument("--out-onnx", default="crazyflie_policy.onnx")
    args = parser.parse_args()

    if not (args.export or args.benchmark):
        parser.print_help()
        raise SystemExit(0)

    kind, obs_dim, act_dim = inspect_checkpoint(args.checkpoint)
    print(f"Checkpoint {args.checkpoint}: kind={kind} obs_dim={obs_dim} act_dim={act_dim}")

    if args.export:
        net, kind, obs_dim, act_dim = load_actor(args.checkpoint)
        export_c_header(net, obs_dim, act_dim, args.out_header, args.checkpoint)
        try:
            export_onnx(net, obs_dim, args.out_onnx)
        except RuntimeError as e:
            print(f"  (skipped) {e}")

    if args.benchmark:
        benchmark(args.checkpoint, n_episodes=args.episodes)
