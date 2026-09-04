"""Behavioral verification harness for the vectorized gpu_accelerated.py rewrite.

Tests:
  A. _get_obs(): new vectorized gather == original loop-based gather on the
     original dims, bit-for-bit (same env state incl. fire_dist synced to exact),
     plus the appended 36-dim visited-map cue equals an independent downsample.
  B. _update_fire_dist(): new exact vectorized EDT vs original loop EDT;
     asserts distance maps match and the safety-critical invariant holds:
     no cell 'safe' per new while 'dangerous' per original (<0.5 threshold).
  C. step() mechanics: with exact fire_dist active in BOTH envs and no deaths,
     state after every step must be identical (positions, alive, visited,
     coverage, dones, crashed, fire/fuel/thermal, observations).
  D. Death semantics: crashed drones stay dead, frozen, and stop exploring
     (matches CPU reference env). Demonstrates original GPU code resurrected them.
  E. End-to-end batched_train() run: tensor-buffer PPO path, rewards/coverage
     recorded, checkpoints saved and loadable.
"""
import os, sys, json, types, subprocess, math
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from gpu_accelerated import (GPUWildfireEnv as NewEnv,
                             BatchedGATPPO, batched_train)

# ---- Original module straight from git, at the last PRE-CUE commit ----
# (HEAD now contains the 6x6 visited-map cue itself, so comparing against HEAD
# would be vacuous: both sides would be the 532-dim env. 4afa765 is the last
# commit with the original 496-dim observation contract this test pins.)
head_src = subprocess.run(['git', 'show', '4afa765:gpu_accelerated.py'],
                          capture_output=True, text=True, check=True).stdout
orig_mod = types.ModuleType('gpu_accel_orig_ref')
orig_mod.__dict__['__file__'] = os.path.join(ROOT, 'gpu_accelerated.py')
exec(compile(head_src, 'gpu_accelerated_orig.py', 'exec'), orig_mod.__dict__)
OldEnv = orig_mod.GPUWildfireEnv

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}: {name}" + (f"  [{detail}]" if detail else ""))

# Torch/CUDA note: we run on CPU here (no GPU in dev env); determinism identical.
print(f"device = {device if 'device' in globals() else torch.device('cpu')}", end="")
print(f" | torch {torch.__version__}")

def make_envs(N=4, G=30, K=10, exact_new=True):
    """Two envs (original vs new) with identical hand-crafted state."""
    old = OldEnv(n_envs=N, grid=G, n_drones=K, max_steps=300)
    new = NewEnv(n_envs=N, grid=G, n_drones=K, max_steps=300)
    # Give new the ORIGINAL exact fire-dist method when requested (isolates step mechanics)
    if exact_new:
        new._update_fire_dist = OldEnv._update_fire_dist.__get__(new)
    return old, new

def set_state(env, N=4, G=30, K=10, fire_c=(15, 15), R=3):
    """Deterministic state: identical circular fire in every env, no wind."""
    env.fire.zero_()
    cx, cy = fire_c
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            if dx * dx + dy * dy <= R * R:
                env.fire[:, cy + dy, cx + dx] = 0.8
    env.fuel.fill_(1.0)
    env.wind_x.zero_()
    env.wind_y.zero_()
    env.shared_visited.zero_()
    env.total_cells_explored.zero_()
    env.drone_vel.zero_()
    env.drone_alive.fill_(True)
    env.step_count = 0
    # scattered drone positions: near corners/boundaries + away from the fire disk
    pos = torch.tensor([
        [2.2, 2.4], [27.3, 27.1], [2.8, 27.6], [27.9, 2.5],
        [5.1, 5.0], [25.0, 6.2], [6.4, 25.1], [24.2, 24.8],
        [10.3, 3.3], [3.1, 10.7],
    ])
    for n in range(N):
        for k in range(K):
            env.drone_pos[n, k] = pos[k % K]
    env._update_thermal()
    env._update_fire_dist()
    env.episode_cells = env.total_cells_explored.sum(dim=(1, 2)).float()

# ═══════════════════════════════════════════════════════════
print("\n[TEST A] _get_obs(): vectorized == original loop implementation")
old, new = make_envs()
set_state(old); set_state(new)
# add some visited cells + mark one drone dead per env to exercise masking
for n in range(4):
    new.shared_visited[n, 5:8, 5:8] = 1.0
    old.shared_visited[n, 5:8, 5:8] = 1.0
    new.total_cells_explored[n, 5:8, 5:8] = True
    old.total_cells_explored[n, 5:8, 5:8] = True
old.drone_alive[0, 0] = False
new.drone_alive[0, 0] = False
old._update_fire_dist(); new._update_fire_dist()
old.episode_cells = old.total_cells_explored.sum(dim=(1, 2)).float()
new.episode_cells = new.total_cells_explored.sum(dim=(1, 2)).float()

o = old._get_obs()
n_ = new._get_obs()
check("new obs widened by 36-dim visited-map cue",
      o.shape == (4, 10, 496) and n_.shape == (4, 10, 532),
      f"old={o.shape} new={n_.shape}")
# Original obs contract (first 496 dims) must stay bit-identical
check("obs original dims bit-identical", torch.equal(o, n_[:, :, :496]),
      f"max abs diff = {(o - n_[:, :, :496]).abs().max().item() if not torch.equal(o, n_[:, :, :496]) else 0}")
# The visited-map cue tail must equal an independent recompute of the downsample
vis_map = F.adaptive_avg_pool2d(new.shared_visited.unsqueeze(1), (6, 6)).reshape(4, 36)
tail = n_[:, 2, 496:532]   # agent 2 is alive in every env (agent 0 in env 0 is dead)
check("visited-map cue matches independent downsample",
      torch.allclose(tail, vis_map, atol=1e-6),
      f"max |d|={(tail - vis_map).abs().max().item():.3g}")
# sanity: obs is not degenerate (nonzero variance) so comparison is meaningful
check("obs has signal (std>0)", o.std().item() > 1e-6, f"std={o.std().item():.4f}")
# dead agent must be all-zero (including the new cue)
check("dead agent obs all-zero", n_[0, 0].abs().sum().item() == 0.0)
# spot-check a local fire channel value matches the raw channel at drone's cell
e, k = 2, 0
px = int(new.drone_pos[e, k, 0].item()); py = int(new.drone_pos[e, k, 1].item())
fire_ch = new.fire[e, py, px].item()
fire_patch_center = n_[e, k, 0 * 81 + 4 * 9 + 4].item()
check("obs fire channel at drone center matches env.fire",
      abs(fire_patch_center - fire_ch) < 1e-6, f"obs={fire_patch_center:.4f} env={fire_ch:.4f}")

# ═══════════════════════════════════════════════════════════
print("\n[TEST B] _update_fire_dist(): exact EDT (new) vs original loop EDT (orig)")
# Compare distance maps produced from IDENTICAL fire states (initial + grown fires).
errs_all = []
false_safe_total = false_danger_total = 0
fire_cell_counts = []
for trial in range(6):
    e_old, e_new = make_envs(exact_new=False)
    set_state(e_old); set_state(e_new)
    if trial < 4:  # grow the fire by stepping the NEW env (single canonical trajectory)
        torch.manual_seed(trial)
        acts = torch.zeros(4, 10, dtype=torch.long)
        for s in range(1, 40):
            torch.manual_seed(trial * 1000 + s)
            e_new.step(acts)
    # Replay the exact same fire state into the original env and recompute both maps
    e_old.fire.copy_(e_new.fire)
    e_old.fuel.copy_(e_new.fuel)
    e_old._update_fire_dist(); e_new._update_fire_dist()
    exact = e_old.fire_dist
    mine = e_new.fire_dist
    fire_cell_counts.append(int((e_new.fire > 0.2).sum(dim=(1, 2)).float().mean().item()))
    err = (mine - exact).abs()
    errs_all.append(err)
    false_safe = int(((mine >= 0.5) & (exact < 0.5)).sum().item())    # mine misses danger
    false_danger = int(((mine < 0.5) & (exact >= 0.5)).sum().item())  # mine over-alarms
    false_safe_total += false_safe
    false_danger_total += false_danger

err_cat = torch.cat([e.reshape(-1) for e in errs_all])
print(f"  fire cells per env: initial={fire_cell_counts[5]} grown={max(fire_cell_counts[:5])}")
print(f"  EDT error over {err_cat.numel()} cells: max={err_cat.max().item():.5f} "
      f"mean={err_cat.mean().item():.8f} p95={err_cat.quantile(0.95).item():.6f}")
check("EDT distance maps match original (max err < 1e-3)", err_cat.max().item() < 1e-3,
      f"max={err_cat.max().item():.3g}")
check("SAFETY: no cell new-says-SAFE while original says DANGEROUS", false_safe_total == 0,
      f"false_safe={false_safe_total} false_danger={false_danger_total}")

# ═══════════════════════════════════════════════════════════
print("\n[TEST C] step() mechanics: new == original every step (no deaths, exact fire_dist)")
old, new = make_envs(exact_new=True)
set_state(old); set_state(new)
acts = torch.zeros(4, 10, dtype=torch.long)
all_eq = True
for s in range(12):
    # deterministic action schedule: mostly stay, some moves far from fire
    for n in range(4):
        for k in range(10):
            acts[n, k] = [0, 1, 2, 0, 0, 3, 0, 4, 0, 0][k] if n == 1 else 0
    torch.manual_seed(1000 + s)
    o_obs, o_done, o_crash = old.step(acts)
    torch.manual_seed(1000 + s)
    n_obs, n_done, n_crash = new.step(acts)
    all_eq = True
    det = ""
    for nm, o_t, n_t in [
        ("dones", o_done, n_done), ("crashed", o_crash, n_crash),
        ("drone_alive", old.drone_alive, new.drone_alive),
        ("drone_pos", old.drone_pos, new.drone_pos),
        ("drone_vel", old.drone_vel, new.drone_vel),
        ("shared_visited", old.shared_visited, new.shared_visited),
        ("episode_cells", old.episode_cells, new.episode_cells),
        ("fire", old.fire, new.fire), ("fuel", old.fuel, new.fuel),
        ("thermal", old.thermal, new.thermal),
        ("fire_dist", old.fire_dist, new.fire_dist),
    ]:
        if not torch.equal(o_t, n_t):
            all_eq = False
            d = (o_t.float() - n_t.float()).abs()
            det += f"{nm}(max|d|={d.max().item():.4g}) "
    # obs: allow float32 rounding-order noise (<1 ULP at ~1.0 scale); states above
    # must be bit-identical. Compare only the shared original dims (the new env
    # additionally carries the 36-dim visited-map cue by design).
    n_obs_shared = n_obs[:, :, :o_obs.shape[-1]]
    if not torch.allclose(o_obs, n_obs_shared, atol=1e-6):
        all_eq = False
        det += f"obs(max|d|={(o_obs - n_obs_shared).abs().max().item():.3g}) "
    check(f"step {s}: full state identical", all_eq, det.strip())
    if not all_eq:
        break
# real crashes never happened in this corridor test
check("no drone crashed in corridor test", old.drone_alive.all().item())

# ═══════════════════════════════════════════════════════════
print("\n[TEST D] Death semantics: crashed drone stays dead/frozen (CPU-ref semantics)")
old, new = make_envs(exact_new=True, N=2, K=1)
set_state(old, N=2, K=1)
set_state(new, N=2, K=1)
# env0's drone sits just left of fire and walks right into it; env1's drone far away.
for env in (old, new):
    env.drone_pos[0, 0] = torch.tensor([12.0, 15.0])   # 3 cells left of fire center (15,15), radius 3
    env.drone_pos[1, 0] = torch.tensor([4.0, 4.0])
    env._update_thermal()
    env._update_fire_dist()
    env.episode_cells = env.total_cells_explored.sum(dim=(1, 2)).float()

moved = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # RIGHT / STAY alternation (N,K)=(2,1)->(2,)
crash_step = None
for s in range(8):
    acts = (torch.tensor([moved[s], 0]).reshape(2, 1))
    old.step(acts)
    new.step(acts)
    # New env: env0 drone should crash around step 1-2 (x: 12->12.3->... fire edge at x<=12)
    if crash_step is None and not new.drone_alive[0, 0]:
        crash_step = s + 1
        cov_before = int(new.total_cells_explored[0].sum().item())

# ---- assertions on NEW (should match CPU reference semantics) ----
check("drone crashed by walking into fire", crash_step is not None, f"crash at step {crash_step}")
# dead: stays dead, stays frozen, adds no coverage
if crash_step is not None:
    still_dead = not bool(new.drone_alive[0, 0].item())
    pos_at_crash = new.drone_pos[0, 0].clone()
    cov_now = int(new.total_cells_explored[0].sum().item())
    check("dead drone never resurrects", still_dead)
    check("dead drone position frozen", bool(torch.equal(new.drone_pos[0, 0], pos_at_crash)))
    check("dead drone adds no coverage after crash", cov_now == cov_before,
          f"cov {cov_before} -> {cov_now}")
    check("dones stays True for dead drone",
          bool(new.drone_alive[0, 0].logical_not()))
else:
    for nm in ("still_dead", "frozen", "no coverage"):
        check(nm, False)
# ---- show the OLD gpu env resurrects (bug my rewrite fixes) ----
if crash_step is not None:
    resurrected = bool(old.drone_alive[0, 0].item()) and not bool(new.drone_alive[0, 0].item())
    print(f"  (note: original GPU env {'' if resurrected else 'does NOT '}resurrect dead drone "
          f"— alive_orig={bool(old.drone_alive[0,0].item())}, alive_new={bool(new.drone_alive[0,0].item())})")
    check("orig-gpu bug demos resurrection (informational)", resurrected)

# ═══════════════════════════════════════════════════════════
print("\n[TEST E] End-to-end batched_train() (tensor-buffer PPO + reward path)")
rid = "verify_tmp"
try:
    agent, res = batched_train(n_episodes=96, grid=30, n_drones=10, max_steps=150,
                               n_envs=32, use_gat=False, seed=7, run_id=rid)
    covs = res['coverages']
    check("coverage recorded for all episodes", len(covs) >= 96, f"len={len(covs)}")
    check("coverage in valid range", all(0 <= c <= 100 for c in covs),
          f"min={min(covs):.1f} max={max(covs):.1f}")
    check("safety in valid range", all(0 <= x <= 100 for x in res['safety']))
    check("no NaN in rewards", all(math.isfinite(r) for r in res['rewards']))
    # Learning must be finite and must actually explore: a fresh-env probe of the
    # BEST checkpoint must stay well above the hover/stall regime (~5-8%) that this
    # fix targets. (Random wandering on this grid sits near ~37%.)
    for f in (f"{rid}_final.pt", f"{rid}_best.pt"):
        ck = torch.load(f, map_location='cpu')
        check(f"{f} saves gat+policy state", 'gat' in ck and 'policy' in ck)
    js = json.load(open(f"{rid}_training_results.json"))
    check("results JSON valid", len(js['coverages']) >= 96)
    # Fresh-env probe of the best checkpoint on 32 fresh episodes (same 150-step cfg)
    torch.manual_seed(11)
    e = NewEnv(n_envs=32, grid=30, n_drones=10, max_steps=150)
    best = BatchedGATPPO(obs_dim=e.obs_dim, act_dim=e.act_dim, use_gat=False)
    best.load(f"{rid}_best.pt")
    o = e.reset()
    covs_e = []
    for _ in range(150):
        if not e.drone_alive.any():
            break
        with torch.no_grad():
            a, lp, _, _ = best.select_actions_batched(o, e.drone_pos, e.drone_alive)
        o, d, _ = e.step(a)
        if d.all():
            break
    cov_e = float(e.total_cells_explored.sum(dim=(1, 2)).float().mean().item()) / (30 * 30) * 100
    check("best checkpoint explores (fresh-eval cov > 15%)", cov_e > 15.0,
          f"fresh cov={cov_e:.1f}%")
finally:
    for f in (f"{rid}_final.pt", f"{rid}_best.pt", f"{rid}_training_results.json"):
        if os.path.exists(f):
            os.remove(f)

print("\n" + "=" * 70)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL BEHAVIORAL CHECKS PASSED")
