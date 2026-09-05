"""
Autonomy Stress Mission — COVER + MOVE + WITHSTAND + FOOTAGE
============================================================

One integrated, headless mission that exercises the full stack autonomously:

  * COVER     — real OpenRescueEnv grid exploration (coverage %, POI discovery)
  * MOVE      — the Resilient policy's R-gated actions (explore/cluster/relay)
  * WITHSTAND — macondo_hover (validated) wind physics gates the mission:
                wind raises sensor noise / GPS denial / packet loss, so R drop
                and the policy switches mode itself; when wind exceeds the
                airframe's max holdable speed the drones are grounded
                ('wind_incapable'), and high wind imposes extra energy drain
                (battery strain). The gust-response model reports the peak
                position deviation the controller must fight.
  * FOOTAGE   — renders a mission MP4 (via matplotlib FFmpegWriter) + JSON
                telemetry of the whole flight.

Wind scenario (storm): calm -> 50 mph gust -> drop -> second gust -> calm.

Run:
    python autonomy_stress.py [--policy resilient] [--steps 160]
                              [--seed 0] [--out figures]

Writes: <out>/autonomy_stress_summary.json, <out>/autonomy_stress.png,
        <out>/autonomy_footage.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openrescue.environment import OpenRescueEnv
from openrescue.policies import make_policy
from openrescue.failures import FailureConfig
import macondo_hover as wind

DT = 0.25          # seconds per env step (200 steps = 50 s mission)
WIND_50MPH = 22.3  # m/s


def wind_at(t: float, seed: int = 0) -> float:
    """Piecewise-linear storm profile: calm -> 50 mph -> drop -> gust -> calm."""
    rng = np.random.default_rng(seed)
    noise = 0.6 + 0.4 * rng.random()
    profile = [
        (0.0, 0.0), (5.0, 1.5),
        (8.0, WIND_50MPH), (14.0, WIND_50MPH),
        (16.0, 9.0), (20.0, 15.0), (24.0, 12.0),
        (30.0, 4.0), (40.0, 1.0),
    ]
    # linear interpolation between vertices
    for (t0, v0), (t1, v1) in zip(profile, profile[1:]):
        if t0 <= t <= t1:
            frac = (t - t0) / max(1e-9, t1 - t0)
            return v0 + (v1 - v0) * frac + noise * math.sin(2.2 * t) * 0.4
    return profile[-1][1]


def wind_effects(v: float, base: FailureConfig) -> FailureConfig:
    """Map wind speed to degraded sensing/comm — the physical->R link."""
    w = min(1.5, v / WIND_50MPH)
    def clamp(p): return float(min(1.0, p))
    return FailureConfig(
        level=base.level,
        sensor_noise_std=base.sensor_noise_std + 0.55 * w,
        gps_denial_prob=clamp(base.gps_denial_prob + 0.30 * w),
        gps_drift_std=base.gps_drift_std + 0.5 * w,
        packet_loss_prob=clamp(base.packet_loss_prob + 0.38 * w),
        comm_range_scale=max(0.30, base.comm_range_scale * (1.0 - 0.30 * w)),
        map_corrupt_prob=clamp(base.map_corrupt_prob + 0.10 * w),
    )


def render_frame(fig, env, tele, t, v, hold_ok):
    """Draw one mission frame: grid + swarm colored by mode + telemetry."""
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(env.obstacles, cmap="Greys", alpha=0.45, origin="lower")
    pois = np.array([p for p in env.pois]) if env.pois else np.zeros((0, 2))
    if len(pois):
        found = env.poi_found
        if found.any():
            ax.scatter(pois[found, 0], pois[found, 1], marker="*", s=120,
                       color="lime", edgecolor="k", label="POI found")
        if (~found).any():
            ax.scatter(pois[~found, 0], pois[~found, 1], marker="*", s=120,
                       color="gold", edgecolor="k", label="POI unfound")
    for d in env.drones:
        mode_color = {"explore": "#2980b9", "cluster": "#e67e22", "relay": "#c0392b"}
        c = mode_color.get(d["mode"], "0.5")
        ax.scatter(*d["pos"], s=60, color=c,
                   marker="x" if d["grounded"] else "o",
                   edgecolor="k", linewidth=0.6)
    ax.scatter([], [], color=("red" if not hold_ok else "green"), marker="o", s=8,
               label=("GROUNDED (wind)" if not hold_ok else "hold OK"))
    ax.set_xlim(-0.5, env.grid - 0.5); ax.set_ylim(-0.5, env.grid - 0.5)
    ax.set_title(f"t={t:5.1f}s   wind={v:4.1f} m/s ({v*2.23694:4.0f} mph)  "
                 f"hold={'OK' if hold_ok else 'OVER'}")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ts = [r["t"] for r in tele]
    ax2.plot(ts, [r["wind"] for r in tele], "b-", label="wind m/s", lw=1)
    ax2.set_ylabel("wind (m/s)", color="b"); ax2.tick_params(axis="y", colors="b")
    ax2r = ax2.twinx()
    ax2r.plot(ts, [r["mean_r"] for r in tele], "g-", label="mean R", lw=1.2)
    ax2r.plot(ts, [r["coverage"] for r in tele], "orange", ls="--", label="coverage", lw=1)
    ax2r.set_ylabel("mean R / coverage"); ax2r.set_ylim(0, 1.0)
    ax2.set_title("telemetry"); ax2.set_xlabel("time (s)")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower left")
    fig.tight_layout()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="resilient",
                    choices=["random", "frontier", "resilient", "ig"])
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wind-scale", type=float, default=1.0,
                    help="multiply the storm wind profile (1.0 = 50 mph peak; "
                         "raise to exercise the airframe hold-limit grounding)")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--no-video", action="store_true",
                    help="skip the mp4 (just JSON + summary figure)")
    args = ap.parse_args()

    if args.steps < 20:
        raise SystemExit("--steps must be >= 20")
    if args.wind_scale <= 0:
        raise SystemExit("--wind-scale must be > 0")
    duration = args.steps * DT
    af = wind.Airframe()
    env = OpenRescueEnv(grid=24, n_drones=6, n_pois=10, max_steps=args.steps,
                        failure_level=1, seed=args.seed)
    policy = make_policy(args.policy)
    rng = np.random.default_rng(args.seed)
    obs, info = env.reset(seed=args.seed)
    if hasattr(policy, "reset"):
        policy.reset()
    base_failure = env.failure

    tele = []
    travel_m = 0.0
    wind_energy_j = 0.0
    mode_counts = {"explore": 0, "cluster": 0, "relay": 0}
    hold_violations = 0
    peak_dev_cm = 0.0
    prev_pos = [np.array(a["pos"]) for a in info["agents"]]
    ground_causes = {}

    os.makedirs(args.out, exist_ok=True)
    fig = plt.figure(figsize=(11, 5.4))
    writer = FFMpegWriter(fps=4, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    video_path = os.path.join(args.out, "autonomy_footage.mp4")
    frame_skip = max(1, args.steps // 40)   # keep the movie short (~40 frames)

    vid_ctx = nullcontext() if args.no_video else writer.saving(fig, video_path, dpi=110)
    with vid_ctx:
        for step in range(args.steps):
            t = step * DT
            v = wind_at(t, seed=args.seed) * args.wind_scale
            static = wind.static_analysis(af, v)
            hold_ok = v <= static["max_holdable_wind_ms"]
            if not hold_ok:
                hold_violations += 1
            # controller deviation is only meaningful while the airframe can
            # hold; beyond the limit the drones are grounded (wind_incapable)
            if hold_ok:
                dev = wind.simulate_gust(af, 0.025, 0.003, 0.03, 1000,
                                         True, v)["peak_dev_m"] * 100.0
                peak_dev_cm = max(peak_dev_cm, dev)

            # physical->R coupling: wind degrades sensing/comm
            env.failure = wind_effects(v, base_failure)

            acts = policy.act(obs, info, rng)
            if not hold_ok:
                # airframe cannot hold: spool down / hold position
                acts = np.zeros(env.n_drones, dtype=np.int32)
                for d in env.drones:
                    if not d["grounded"] and d["battery"] > 0:
                        d["grounded"] = True
                        d["cause"] = "wind_incapable"

            obs, _, terminated, truncated, info = env.step(acts)

            # extra energy drain from fighting high wind (battery strain)
            if v > 12.0:
                strain = 0.35 * (v / WIND_50MPH)
                for d in env.drones:
                    if not d["grounded"] and d["battery"] > 0:
                        d["battery"] = max(0.0, d["battery"] - strain)
                        if d["battery"] == 0:
                            d["grounded"] = True
                            d["cause"] = "battery"
                wind_energy_j += strain * env.energy_to_joules * env.n_drones

            for i, d in enumerate(env.drones):
                travel_m += float(np.linalg.norm(d["pos"] - prev_pos[i]))
                prev_pos[i] = np.array(d["pos"])
                mode_counts[d["mode"]] = mode_counts.get(d["mode"], 0) + 1
                if d["grounded"] and d["cause"] and i not in ground_causes:
                    ground_causes[i] = d["cause"]

            tele.append({
                "t": round(t, 2),
                "wind": round(v, 2),
                "hold_ok": bool(hold_ok),
                "mean_r": round(float(np.mean([d["r_index"] for d in env.drones])), 4),
                "coverage": round(float(env.coverage), 4),
                "pois": int(env.poi_found.sum()),
                "alive": int(sum(1 for d in env.drones if not d["grounded"])),
                "battery": round(float(np.mean([d["battery"] for d in env.drones])), 2),
            })

            if step % frame_skip == 0:
                render_frame(fig, env, tele, t, v, hold_ok)
                if not args.no_video:
                    writer.grab_frame()

            if bool(np.all(terminated)) or bool(np.all(truncated)):
                break

    summary = env.episode_summary()
    holdable_mph = wind.static_analysis(af, WIND_50MPH)["max_holdable_wind_mph"]
    summary["mission_meta"] = {
        "policy": args.policy,
        "seed": args.seed,
        "wind_scale": args.wind_scale,
        "steps_run": step + 1,
        "duration_s": round(duration, 1),
        "dt_s": DT,
        "airframe": {"mass_g": af.mass_kg * 1000, "twr": af.twr,
                      "cd_a": af.cd_a},
        "max_holdable_wind_mph": round(holdable_mph, 1),
    }
    summary["wind_stress"] = {
        "peak_wind_ms": round(max(r["wind"] for r in tele), 2),
        "peak_wind_mph": round(max(r["wind"] for r in tele) * 2.23694, 1),
        "seconds_above_holdable": round(hold_violations * DT, 1),
        "hold_ok_fraction": round(1.0 - hold_violations / max(1, len(tele)), 3),
        "peak_controller_deviation_cm": round(peak_dev_cm, 2),
        "extra_wind_energy_j": round(wind_energy_j, 2),
        "travel_distance_m": round(travel_m, 2),
        "mode_seconds": {k: round(v * DT, 1) for k, v in mode_counts.items()},
        "ground_causes": ground_causes,
        "coverage": round(float(env.coverage), 4),
        "pois_found": int(env.poi_found.sum()),
        "agents_lost": int(sum(1 for d in env.drones if d["grounded"])),
        "lost_reasons": {k: int(sum(1 for d in env.drones if d["cause"] == k))
                         for k in set(d["cause"] for d in env.drones if d["cause"])},
        "telemetry": tele,
    }

    json_path = os.path.join(args.out, "autonomy_stress_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    # summary figure (telemetry timeline)
    fig2, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ts = [r["t"] for r in tele]
    axs[0].plot(ts, [r["wind"] * 2.23694 for r in tele], "b-", lw=1.2)
    axs[0].axhline(holdable_mph, color="r", ls="--", lw=1,
                   label=f"max holdable wind ({holdable_mph:.1f} mph)")
    axs[0].set_ylabel("wind (mph)"); axs[0].legend(fontsize=8)
    axs[1].plot(ts, [r["mean_r"] for r in tele], "g-", lw=1.2, label="mean R")
    axs[1].plot(ts, [r["coverage"] for r in tele], "orange", ls="--", lw=1.2, label="coverage")
    axs[1].set_ylabel("mean R / coverage"); axs[1].set_ylim(0, 1); axs[1].legend(fontsize=8)
    axs[2].plot(ts, [r["alive"] for r in tele], "k-", lw=1.2, label="alive")
    axs[2].plot(ts, [r["pois"] for r in tele], "m-", lw=1.2, label="POIs")
    axs[2].set_xlabel("time (s)"); axs[2].set_ylabel("count"); axs[2].legend(fontsize=8)
    fig2.suptitle(f"Autonomy Stress Mission — policy={args.policy}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out, "autonomy_stress.png"), dpi=130)
    plt.close(fig2)

    ws = summary["wind_stress"]
    print(f"Autonomy stress mission complete  policy={args.policy} seed={args.seed}")
    print(f"  duration {summary['mission_meta']['duration_s']}s  "
          f"peak wind {ws['peak_wind_mph']:.0f} mph  hold-OK fraction {ws['hold_ok_fraction']:.2f}")
    if ws['peak_controller_deviation_cm'] > 0:
        print(f"  controller peak deviation {ws['peak_controller_deviation_cm']:.1f} cm "
              f"(estimates at each gust)")
    else:
        print("  controller peak deviation N/A (never below the hold limit)")
    print(f"  coverage {ws['coverage']:.2f}  POIs {ws['pois_found']}/10  "
          f"agents lost {ws['agents_lost']}  reasons {ws['lost_reasons']}")
    print(f"  travel {ws['travel_distance_m']:.0f} m  extra wind energy {ws['extra_wind_energy_j']:.0f} J")
    print(f"  mode seconds {ws['mode_seconds']}")
    print(f"wrote {json_path}")
    print(f"wrote {os.path.join(args.out, 'autonomy_stress.png')}")
    if not args.no_video:
        print(f"wrote {video_path}")


if __name__ == "__main__":
    main()