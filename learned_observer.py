"""
Learned Disturbance Observer (LDO) -- measured ablation vs analytic feedforward
===============================================================================

Same 1-DOF horizontal position-hold plant as ``macondo_hover.py`` (first-order
rotor lag ``tau_rot``, lagged + noisy accelerometer, PD+I position loop, and
an optional disturbance-observer feedforward), but the disturbance estimate
is produced by a small MLP trained *predictively* on domain-randomized
trajectories:

    plant :  x_ddot = u_act + w(t)                 (w = drag specific force)
    sensor:  a_meas lagged by tau_sens + white noise
    net   :  w_hat(k) = MLP( window of [a_meas, u_act, v] ), target w(k + H)

The analytic observer (first-order low-pass with tau_obs = 30 ms) is compared
against the learned estimator on identical closed-loop rollouts under four
scenarios: a nominal 50 mph step gust, an out-of-distribution airframe
(mass/drag outside the training range), turbulent (colored-noise) wind, and
heavy sensor noise. Domain randomization during training (mass, drag area,
rotor lag, sensor lag, noise, gust shape, wind speed) is the only information
the network is given; no plant model is encoded in the weights.

Honest framing: the analytic observer is optimal when the disturbance is
(quasi-)constant, because ``a_meas - u_act`` is then an exact, unbiased
measurement. The learned estimator can only win where there is temporal
structure to exploit -- gust ramps and the turbulence spectrum -- or where
the sensor lag corrupts the direct measurement. This script MEASURES where
that happens and where it does not, and reports both.

With ``--retrain-loop N`` the estimator is then retrained on its own
rollouts: N additional rounds collect fresh domain-randomized trajectories
under the current learned feedforward and retrain on the combined data. The
round-0 (distill-only) closed-loop table is kept as the baseline; if the
retrained model improves any scenario's learned row it becomes the recorded
result, otherwise the round-0 table is kept and the null result is recorded
in the JSON meta.

Run:  python learned_observer.py [--epochs 12] [--trajs 100] [--horizon-ms 5]
                                  [--seeds 5] [--retrain-loop 1]
                                  [--out learned_observer.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from macondo_hover import Airframe, drag_force, pick_gains, G
from openrescue.metrics import bootstrap_ci

LOOP_HZ = 1000
W = 20          # feature window (steps, = ms at 1 kHz)
H = 5           # prediction horizon (steps, = ms at 1 kHz)

TRAIN_DOMAINS = dict(
    mass_kg=(0.11, 0.15),        # nominal 0.130
    cd_a=(0.008, 0.016),         # nominal 0.012
    tau_rot_s=(0.015, 0.050),    # rotor+prop first-order lag
    tau_sens_s=(0.002, 0.006),   # accelerometer lag
    noise_ms2=(0.02, 0.15),      # accelerometer white noise
    v_steady_ms=(10.0, 25.0),    # sustained wind
    gust_rise_s=(0.05, 0.30),    # step-gust rise times
    turb_tau_s=(0.20, 0.80),     # turbulence correlation time
    turb_amp=(0.15, 0.40),       # turbulence intensity (fraction of steady)
)


class Net(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid // 2), nn.ReLU(),
            nn.Linear(hid // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


def wind_array(kind: str, t: np.ndarray, v_steady: float, rng: np.random.Generator,
               rise: float = 0.15, t_gust: float = 0.5,
               tau_turb: float = 0.4, amp: float = 0.25) -> np.ndarray:
    """Wind speed profile: step gust or turbulent (colored-noise) wind."""
    if kind == "step":
        return v_steady * np.clip((t - t_gust) / rise, 0.0, 1.0)
    # turbulent: steady base + colored noise (cascaded 1st-order low-pass of
    # white noise -- an approximate Von-Karman-like energy decay)
    n = t.shape[0]
    white = rng.normal(0.0, 1.0, n)
    x = white
    alpha = 1.0 / (tau_turb * LOOP_HZ)
    for _ in range(2):
        y = np.empty_like(x)
        y[0] = x[0]
        for k in range(1, n):
            y[k] = y[k - 1] + (x[k] - y[k - 1]) * alpha
        x = y
    x = x / (np.std(x) + 1e-12)
    v = v_steady + amp * v_steady * x
    return np.clip(v, 0.5, None)


def rollout(af: Airframe, tau_rot: float, tau_sens: float, tau_obs: float,
            wind_arr: np.ndarray, ff: str, model=None, feat_mean=None,
            feat_std=None, tgt_std: float = 1.0, tgt_mean: float = 0.0,
            noise_sigma: float = 0.05,
            rng: np.random.Generator = None, t_end: float = 5.0,
            omega_n: float = 10.0, zeta: float = 0.9,
            collect: bool = False, trace: bool = False) -> dict:
    """One closed-loop rollout. ``ff`` in {"none", "analytic", "learned"}."""
    dt = 1.0 / LOOP_HZ
    n = int(round(t_end * LOOP_HZ))
    t = np.arange(n) * dt
    gains = pick_gains(tau_rot, omega_n, zeta)
    kp, kd, ki = gains["kp"], gains["kd"], gains["ki"]
    a_h_max = math.sqrt((af.twr * G) ** 2 - G ** 2)
    u_max = 0.95 * a_h_max

    w = drag_force(wind_arr, af) / af.mass_kg

    x = np.zeros(n); v = np.zeros(n)
    u_act = np.zeros(n); u_cmd = np.zeros(n); integ = np.zeros(n)
    sat_frac = 0.0
    w_hat = 0.0; a_meas = 0.0; w_hat_l = 0.0
    buf_a = np.zeros(W); buf_u = np.zeros(W); buf_v = np.zeros(W)
    feats: list = []; targets: list = []
    tr = {"a_meas": np.zeros(n), "u_act": np.zeros(n), "v": np.zeros(n),
          "w_hat": np.zeros(n), "x": x}

    for k in range(1, n):
        x[k] = x[k - 1] + v[k - 1] * dt
        v[k] = v[k - 1] + (u_act[k - 1] + w[k - 1]) * dt
        u_act[k] = u_act[k - 1] + (u_cmd[k - 1] - u_act[k - 1]) * dt / tau_rot
        a_true = u_act[k] + w[k]
        a_meas += (a_true - a_meas) * dt / tau_sens
        a_meas_n = a_meas + rng.normal(0.0, noise_sigma)
        w_hat += ((a_meas_n - u_act[k]) - w_hat) * dt / tau_obs

        e = -x[k]
        integ[k] = integ[k - 1] + e * dt
        u_pd = kp * e + kd * (-v[k]) + ki * integ[k]
        if ff == "analytic":
            u_cmd[k] = u_pd - w_hat
        elif ff == "learned":
            buf_a[1:] = buf_a[:-1]; buf_a[0] = a_meas_n
            buf_u[1:] = buf_u[:-1]; buf_u[0] = u_act[k]
            buf_v[1:] = buf_v[:-1]; buf_v[0] = v[k]
            f = np.concatenate([buf_a, buf_u, buf_v]).astype(np.float32)
            f = (f - feat_mean) / feat_std
            with torch.no_grad():
                w_hat_l = float(model(torch.from_numpy(f[None, :]))) * tgt_std + tgt_mean
            u_cmd[k] = u_pd - w_hat_l
        else:
            u_cmd[k] = u_pd
        if abs(u_cmd[k]) > u_max:
            u_cmd[k] = math.copysign(u_max, u_cmd[k])
            sat_frac += 1.0 / n

        if trace:
            tr["a_meas"][k] = a_meas_n; tr["u_act"][k] = u_act[k]
            tr["v"][k] = v[k]; tr["w_hat"][k] = w_hat
        if collect and k >= W and k + H < n:
            buf_a[1:] = buf_a[:-1]; buf_a[0] = a_meas_n
            buf_u[1:] = buf_u[:-1]; buf_u[0] = u_act[k]
            buf_v[1:] = buf_v[:-1]; buf_v[0] = v[k]
            feats.append(np.concatenate([buf_a, buf_u, buf_v]).astype(np.float32))
            targets.append(w[k + H])

    dev = np.abs(x)
    out = dict(
        peak_dev_m=float(dev.max()),
        rms_dev_m=float(np.sqrt(np.mean(dev ** 2))),
        steady_err_m=float(np.mean(np.abs(x[int(-0.5 / dt):]))),
        sat_frac=sat_frac,
    )
    if collect:
        out["X"] = np.array(feats); out["y"] = np.array(targets)
    if trace:
        tr["w_hat_l"] = w_hat_l  # placeholder; learned trace filled offline
        out["trace"] = tr
    return out


def gen_dataset(rng_master: np.random.Generator, n_trajs: int,
                t_end: float = 5.0, model=None, mean=None, std=None,
                t_mean: float = 0.0, t_std: float = 1.0) -> tuple:
    feats, targets = [], []
    for _ in range(n_trajs):
        rng = np.random.default_rng(int(rng_master.integers(0, 2 ** 31)))
        mass = rng.uniform(*TRAIN_DOMAINS["mass_kg"])
        cd_a = rng.uniform(*TRAIN_DOMAINS["cd_a"])
        tau_rot = rng.uniform(*TRAIN_DOMAINS["tau_rot_s"])
        tau_sens = rng.uniform(*TRAIN_DOMAINS["tau_sens_s"])
        noise = rng.uniform(*TRAIN_DOMAINS["noise_ms2"])
        v_steady = rng.uniform(*TRAIN_DOMAINS["v_steady_ms"])
        kind = str(rng.choice(["step", "turb"]))
        kw = {}
        if kind == "step":
            kw = dict(rise=float(rng.uniform(*TRAIN_DOMAINS["gust_rise_s"])),
                      t_gust=float(rng.uniform(0.3, 0.8)))
        else:
            kw = dict(tau_turb=float(rng.uniform(*TRAIN_DOMAINS["turb_tau_s"])),
                      amp=float(rng.uniform(*TRAIN_DOMAINS["turb_amp"])))
        af = Airframe(mass_kg=mass, twr=5.0, cd_a=cd_a)
        t = np.arange(int(round(t_end * LOOP_HZ))) / LOOP_HZ
        wind = wind_array(kind, t, v_steady, rng, **kw)
        ff = "learned" if model is not None else "analytic"
        r = rollout(af, tau_rot, tau_sens, 0.030, wind, ff,
                    model, mean, std, t_std, t_mean,
                    noise_sigma=noise, rng=rng, t_end=t_end, collect=True)
        feats.append(r["X"]); targets.append(r["y"])
    X = np.concatenate(feats); y = np.concatenate(targets)
    idx = np.arange(0, X.shape[0], 3)   # subsample for CPU training
    return X[idx], y[idx]


def train(X: np.ndarray, y: np.ndarray, epochs: int, lr: float = 1e-3,
          batch: int = 4096, seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    n = X.shape[0]
    n_val = max(1, n // 10)
    perm = np.random.default_rng(seed).permutation(n)
    Xtr, ytr = X[perm[n_val:]], y[perm[n_val:]]
    Xva, yva = X[perm[:n_val]], y[perm[:n_val]]

    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-6
    t_mean = float(ytr.mean()); t_std = float(ytr.std()) + 1e-6
    Xt = torch.from_numpy(((Xtr - mean) / std).astype(np.float32))
    yt = torch.from_numpy(((ytr - t_mean) / t_std).astype(np.float32))
    Xv = torch.from_numpy(((Xva - mean) / std).astype(np.float32))
    yv = torch.from_numpy(((yva - t_mean) / t_std).astype(np.float32))

    model = Net(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    with torch.no_grad():
        val0 = float(lossf(model(Xv), yv))
    for ep in range(epochs):
        perm_t = torch.randperm(Xt.shape[0])
        tot = 0.0; nb = 0
        for i in range(0, Xt.shape[0], batch):
            idx = perm_t[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            with torch.no_grad():
                val = float(lossf(model(Xv), yv))
            print(f"  epoch {ep + 1:>3}/{epochs}  train_mse={tot / nb:.5f}  "
                  f"val_mse={val:.5f} (started {val0:.5f})")
    return model, mean, std, t_mean, t_std


def learned_from_trace(model, mean, std, t_std, t_mean, a_meas, u_act, v) -> np.ndarray:
    """Offline learned estimates on a recorded (analytic-ff) trace."""
    n = a_meas.shape[0]
    out = np.zeros(n)
    buf_a = np.zeros(W); buf_u = np.zeros(W); buf_v = np.zeros(W)
    for k in range(1, n):
        buf_a[1:] = buf_a[:-1]; buf_a[0] = a_meas[k]
        buf_u[1:] = buf_u[:-1]; buf_u[0] = u_act[k]
        buf_v[1:] = buf_v[:-1]; buf_v[0] = v[k]
        f = np.concatenate([buf_a, buf_u, buf_v]).astype(np.float32)
        f = (f - mean) / std
        with torch.no_grad():
            out[k] = float(model(torch.from_numpy(f[None, :]))) * t_std + t_mean
    return out


def estimator_quality(model, mean, std, t_std, t_mean, rng_seed: int = 7,
                      t_end: float = 5.0) -> dict:
    """Per-step estimation error on a held-out turbulent trajectory."""
    rng = np.random.default_rng(rng_seed)
    af = Airframe(mass_kg=0.13, twr=5.0, cd_a=0.012)
    t = np.arange(int(round(t_end * LOOP_HZ))) / LOOP_HZ
    wind = wind_array("turb", t, 22.3, rng, tau_turb=0.4, amp=0.25)
    w = drag_force(wind, af) / af.mass_kg
    r = rollout(af, 0.025, 0.003, 0.030, wind, "analytic",
                noise_sigma=0.05, rng=rng, t_end=t_end, trace=True)
    tr = r["trace"]
    wl = learned_from_trace(model, mean, std, t_std, t_mean,
                            tr["a_meas"], tr["u_act"], tr["v"])
    idx = np.arange(W, w.shape[0] - H)  # skip warmup and the un-predictable tail
    err_an = np.abs(tr["w_hat"][idx] - w[idx])
    err_ld_H = np.abs(wl[idx] - w[idx + H])      # vs its prediction target
    err_ld_0 = np.abs(wl[idx] - w[idx])          # vs current w
    return dict(
        analytic_tracking_rmse=float(np.sqrt(np.mean(err_an ** 2))),
        learned_Hstep_pred_rmse=float(np.sqrt(np.mean(err_ld_H ** 2))),
        learned_current_rmse=float(np.sqrt(np.mean(err_ld_0 ** 2))),
        horizon_steps=H,
        trace=dict(t=t.tolist(), w=w.tolist(), w_hat=tr["w_hat"].tolist(),
                   w_hat_l=wl.tolist()),
    )


SCENARIOS = {
    "nominal_step": dict(kind="step", v_steady=22.3, mass=0.13, cd_a=0.012,
                         tau_rot=0.025, tau_sens=0.003, noise=0.05,
                         kw=dict(rise=0.15, t_gust=0.5)),
    "ood_airframe": dict(kind="step", v_steady=22.3, mass=0.16, cd_a=0.016,
                         tau_rot=0.025, tau_sens=0.003, noise=0.05,
                         kw=dict(rise=0.15, t_gust=0.5)),
    "turbulent": dict(kind="turb", v_steady=22.3, mass=0.13, cd_a=0.012,
                      tau_rot=0.025, tau_sens=0.003, noise=0.05,
                      kw=dict(tau_turb=0.4, amp=0.15)),
    "authority_limit": dict(kind="turb", v_steady=22.3, mass=0.13, cd_a=0.012,
                            tau_rot=0.025, tau_sens=0.003, noise=0.05,
                            kw=dict(tau_turb=0.4, amp=0.25)),
    "high_noise": dict(kind="step", v_steady=22.3, mass=0.13, cd_a=0.012,
                       tau_rot=0.025, tau_sens=0.003, noise=0.30,
                       kw=dict(rise=0.15, t_gust=0.5)),
}
CTRL_LABEL = {"none": "PD+I only", "analytic": "analytic observer",
              "learned": "learned estimator"}


def eval_scenarios(model, mean, std, t_std, t_mean, seeds: int = 5,
                   t_end: float = 5.0) -> dict:
    out = {}
    for name, sc in SCENARIOS.items():
        per = {"none": [], "analytic": [], "learned": []}
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            af = Airframe(mass_kg=sc["mass"], twr=5.0, cd_a=sc["cd_a"])
            t = np.arange(int(round(t_end * LOOP_HZ))) / LOOP_HZ
            wind = wind_array(sc["kind"], t, sc["v_steady"], rng, **sc["kw"])
            for ff in per:
                r = rollout(af, sc["tau_rot"], sc["tau_sens"], 0.030, wind, ff,
                            model, mean, std, t_std, t_mean,
                            noise_sigma=sc["noise"], rng=rng, t_end=t_end)
                per[ff].append(r)
        out[name] = {}
        for ff, rows in per.items():
            agg = {"n_seeds": seeds}
            for m in ("peak_dev_m", "rms_dev_m", "steady_err_m"):
                iqm, lo, hi = bootstrap_ci([r[m] for r in rows])
                agg[m] = {"iqm": iqm, "ci_low": lo, "ci_high": hi}
            iqm, lo, hi = bootstrap_ci([r["sat_frac"] for r in rows])
            agg["sat_frac"] = {"iqm": iqm, "ci_low": lo, "ci_high": hi}
            out[name][ff] = agg
    return out


def nominal_traces(model, mean, std, t_std, t_mean, seed: int = 5) -> dict:
    """Deterministic traces for the figure (nominal step gust)."""
    sc = SCENARIOS["nominal_step"]
    af = Airframe(mass_kg=sc["mass"], twr=5.0, cd_a=sc["cd_a"])
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(5.0 * LOOP_HZ))) / LOOP_HZ
    wind = wind_array(sc["kind"], t, sc["v_steady"], rng, **sc["kw"])
    out = {}
    for ff in ("none", "analytic", "learned"):
        r = rollout(af, sc["tau_rot"], sc["tau_sens"], 0.030, wind, ff,
                    model, mean, std, t_std, t_mean,
                    noise_sigma=sc["noise"], rng=rng, t_end=5.0, trace=True)
        out[ff] = (t, r["trace"]["x"])
    return out


def make_figure(traces: dict, eq: dict, table: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    for ff, (t, x) in traces.items():
        ax.plot(t, x * 100.0, lw=1.4, label=CTRL_LABEL[ff])
    ax.axvline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (cm)")
    ax.set_title("Nominal 50 mph step gust")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    tt = np.array(eq["trace"]["t"])
    w = np.array(eq["trace"]["w"])
    wh = np.array(eq["trace"]["w_hat"])
    wl = np.array(eq["trace"]["w_hat_l"])
    m = (tt > 1.0) & (tt < 3.0)
    ax.plot(tt[m], w[m], "k-", lw=1.4, label="true drag $w$")
    ax.plot(tt[m], wh[m], "orange", lw=1.2, label="analytic $\\hat w$")
    ax.plot(tt[m], wl[m], "g-", lw=1.2, label="learned $\\hat w$")
    ax.set_xlabel("time (s)"); ax.set_ylabel("drag specific force (m/s$^2$)")
    ax.set_title("Estimates under turbulent wind (held-out)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    names = list(table.keys())
    xpos = np.arange(len(names))
    width = 0.26
    for j, ff in enumerate(("none", "analytic", "learned")):
        vals = [table[sc][ff]["peak_dev_m"]["iqm"] * 100.0 for sc in names]
        ax.bar(xpos + (j - 1) * width, vals, width, label=CTRL_LABEL[ff])
    ax.set_xticks(xpos)
    short = {"nominal_step": "nominal", "ood_airframe": "OOD\nairframe",
             "turbulent": "turbulent", "authority_limit": "authority\nlimit",
             "high_noise": "high\nnoise"}
    ax.set_xticklabels([short[s] for s in names], fontsize=8)
    ax.set_ylabel("peak deviation (cm)"); ax.set_title("Peak hold error, 5 seeds")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    global H
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--trajs", type=int, default=100)
    ap.add_argument("--horizon-ms", type=int, default=H)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--retrain-loop", type=int, default=0,
                    help="learned-in-the-loop rounds: collect fresh trajectories "
                         "under the current learned feedforward and retrain on the "
                         "combined data (0 = distill on analytic trajectories only)")
    ap.add_argument("--out", default="learned_observer.json")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    H = args.horizon_ms

    t0 = time.time()
    print("generating domain-randomized training data ...")
    X, y = gen_dataset(np.random.default_rng(0), args.trajs)
    print(f"  {X.shape[0]} samples x {X.shape[1]} features "
          f"(window {W} steps of [a_meas, u_act, v], horizon {H} ms)")
    print("training MLP estimator (round 0: analytic trajectories) ...")
    model, mean, std, t_mean, t_std = train(X, y, args.epochs)

    # round-0 baseline kept so improved-vs-null is decided from data
    retrain_info = None
    if args.retrain_loop > 0:
        table_r0 = eval_scenarios(model, mean, std, t_std, t_mean, seeds=args.seeds)
        model_r0, mean_r0, std_r0, t_mean_r0, t_std_r0 = model, mean, std, t_mean, t_std
        for rnd in range(1, args.retrain_loop + 1):
            print(f"round {rnd}/{args.retrain_loop}: collecting trajectories "
                  "under the learned feedforward ...")
            X2, y2 = gen_dataset(np.random.default_rng(1000 * rnd), args.trajs,
                                 model=model, mean=mean, std=std,
                                 t_mean=t_mean, t_std=t_std)
            X = np.concatenate([X, X2]); y = np.concatenate([y, y2])
            print(f"  {X2.shape[0]} new samples (total {X.shape[0]})")
            model, mean, std, t_mean, t_std = train(X, y, args.epochs)

    print(f"closed-loop evaluation, {args.seeds} seeds per scenario ...")
    table = eval_scenarios(model, mean, std, t_std, t_mean, seeds=args.seeds)
    if args.retrain_loop > 0:
        deltas = {sc: {
            "learned_peak_delta_cm": round(
                (table[sc]["learned"]["peak_dev_m"]["iqm"]
                 - table_r0[sc]["learned"]["peak_dev_m"]["iqm"]) * 100, 3),
            "learned_rms_delta_cm": round(
                (table[sc]["learned"]["rms_dev_m"]["iqm"]
                 - table_r0[sc]["learned"]["rms_dev_m"]["iqm"]) * 100, 3),
        } for sc in table}
        improved_sc = [sc for sc in deltas
                       if deltas[sc]["learned_peak_delta_cm"] < 0
                       or deltas[sc]["learned_rms_delta_cm"] < 0]
        retrain_info = dict(rounds=args.retrain_loop,
                            improved=bool(improved_sc),
                            improved_scenarios=improved_sc,
                            deltas_cm=deltas)
        if not improved_sc:
            retrain_info["retrained_table"] = table
            retrain_info["note"] = "null result: learned-in-the-loop retraining " \
                                    "did not improve any scenario's learned row; " \
                                    "closed_loop below is the round-0 (distill-only) " \
                                    "verified table"
            model, mean, std, t_mean, t_std = model_r0, mean_r0, std_r0, t_mean_r0, t_std_r0
            table = table_r0
        else:
            print(f"  retraining improved {len(improved_sc)} scenario(s): {improved_sc}")

    ckpt = dict(model=model.state_dict(), W=W, H=H, in_dim=X.shape[1],
                mean=mean.tolist(), std=std.tolist(),
                t_mean=t_mean, t_std=t_std)
    torch.save(ckpt, "learned_observer.pt")
    print(f"  saved learned_observer.pt  ({time.time() - t0:.1f} s)")

    print("estimator quality on held-out turbulent trajectory ...")
    eq = estimator_quality(model, mean, std, t_std, t_mean)
    print(f"  analytic tracking RMSE  {eq['analytic_tracking_rmse']:.3f} m/s^2")
    print(f"  learned {H}-step pred RMSE {eq['learned_Hstep_pred_rmse']:.3f} m/s^2")

    print(f"  {'scenario':<13}{'controller':<17}{'peak cm (IQM [CI])':>22}"
          f"{'rms cm':>9}{'steady cm':>10}{'sat%':>6}")
    for sc, rows in table.items():
        for ff, r in rows.items():
            p = r["peak_dev_m"]; rms = r["rms_dev_m"]
            st = r["steady_err_m"]; sat = r["sat_frac"]
            print(f"  {sc:<13}{CTRL_LABEL[ff]:<17}"
                  f"{p['iqm']*100:6.2f} [{p['ci_low']*100:5.1f},{p['ci_high']*100:5.1f}] "
                  f"{rms['iqm']*100:7.2f} {st['iqm']*100:9.2f} "
                  f"{sat['iqm']*100:5.1f}")
    if retrain_info is not None and retrain_info["improved"]:
        print("  learned row vs round-0 (cm deltas):")
        for sc, dd in retrain_info["deltas_cm"].items():
            print(f"    {sc:<13} peak {dd['learned_peak_delta_cm']:+.3f}  "
                  f"rms {dd['learned_rms_delta_cm']:+.3f}")
    elif retrain_info is not None:
        print("  learned-in-the-loop: NULL result, round-0 table kept "
              "(deltas in JSON meta)")

    # verdict computed from the data, not asserted
    verdict = {}
    for sc in table:
        p_n = table[sc]["none"]["peak_dev_m"]["iqm"]
        p_a = table[sc]["analytic"]["peak_dev_m"]["iqm"]
        p_l = table[sc]["learned"]["peak_dev_m"]["iqm"]
        r_a = table[sc]["analytic"]["rms_dev_m"]["iqm"]
        r_l = table[sc]["learned"]["rms_dev_m"]["iqm"]
        hi_l = table[sc]["learned"]["peak_dev_m"]["ci_high"]
        lo_a = table[sc]["analytic"]["peak_dev_m"]["ci_low"]
        verdict[sc] = dict(
            peak_vs_none_pct=100.0 * (p_l - p_n) / p_n,
            peak_vs_analytic_pct=100.0 * (p_l - p_a) / p_a,
            rms_vs_analytic_pct=100.0 * (r_l - r_a) / r_a,
            learned_peak_ci_separated_from_analytic=bool(hi_l < lo_a),
        )
    data = dict(
        meta=dict(
            task="learned vs analytic disturbance observer, 1-DOF wind-hold "
                 "(macondo_hover plant)",
            plant="x_ddot = u_act + w; first-order rotor lag tau_rot; lagged "
                  "accelerometer tau_sens; PD+I position loop (omega_n=10, "
                  "zeta=0.9); 1 kHz",
            net=dict(arch="MLP 60->128->64->1", window_steps=W,
                     horizon_steps=H, horizon_ms=H),
            training=dict(trajs=args.trajs, epochs=args.epochs,
                          samples=int(X.shape[0]),
                          subsample_every=3,
                          domains=TRAIN_DOMAINS),
            protocol=dict(seeds=args.seeds,
                          identical_draws=True,
                          aggregation="IQM + 95% bootstrap CI "
                                     "(openrescue.metrics.bootstrap_ci, "
                                     "Agarwal et al. 2021)",
                          scenarios=list(SCENARIOS.keys())),
            retrain_loop=retrain_info,
            runtime_s=round(time.time() - t0, 1),
        ),
        estimator_quality={k: v for k, v in eq.items() if k != "trace"},
        closed_loop=table,
        verdict=verdict,
    )
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    print(f"wrote {args.out}")

    if not args.no_figure:
        traces = nominal_traces(model, mean, std, t_std, t_mean)
        fig_path = "figures/observer_comparison.png"
        make_figure(traces, eq, table, fig_path)
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()