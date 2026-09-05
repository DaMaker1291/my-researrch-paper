"""
Macondo — Wind-Hold Feasibility & Tuning Model
=============================================

Translates the OpenRescue resilience framework down one level: from *which
behavior to run* (grid-world R_{i,t}) to *can the airframe physically hold
position* in a given wind. This is the "sim-to-real" data bridge for the
Phase-3 Macondo sub-250g testbed.

Two layers of physics, both pure NumPy:

1. STATIC TRIM  — at steady wind speed V, how far must the quad tilt and how
   much thrust does it need vs. its maximum (TWR)?
       F_d = 0.5 * rho * V^2 * Cd*A          (drag on the airframe)
       tilt = atan2(F_d, m*g)                (tilt from vertical)
       T_req = sqrt((m*g)^2 + F_d^2)         (total thrust required)
       V_max = sqrt( F_d_max / (0.5*rho*Cd*A) ),  F_d_max = m*g*sqrt(TWR^2 - 1)
   Outputs: required tilt, TWR load, climb margin, max holdable wind, and the
   gust headroom (extra m/s of wind before the throttle saturates).

2. DYNAMIC GUST RESPONSE — 1-DOF horizontal position-hold simulation of a
   discrete control loop with a first-order rotor/prop actuator lag tau_rot
   (the 15-50 ms mechanical bottleneck), a delayed accelerometer, and an
   optional disturbance-observer feedforward:
       x_plant_ddot = u_act + w(t)          (w = drag specific force)
       u_act_dot = (u_cmd - u_act) / tau_rot
       w_hat_dot = ((a_meas - u_act) - w_hat) / tau_obs
       u_cmd = Kp*e + Kd*edot + Ki*int(e) - w_hat     (FF on)
   The loop rate is 1 kHz; going faster (8 kHz) changes nothing because the
   actuator lag dominates — the sim makes that visible in data: peak position
   deviation scales with tau_rot, not with loop rate.

Writes ``macondo_wind_data.json`` (static table, gust table, tuning targets)
and ``figures/macondo_hold.png``.

Run:  python macondo_hover.py [--mass-gram 130] [--twr 5.0] [--cd-a 0.012]
                              [--wind-ms 22.3] [--out figures]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81
RHO = 1.225

# Actuator / loop parameters (Macondo spec defaults).
DEFAULTS = dict(
    mass_gram=130.0,      # AUW of the 3" toothpick (sub-250g class)
    twr=5.0,              # max thrust / weight at full throttle
    cd_a=0.012,           # frontal drag area * Cd (tune after anemometer test)
    wind_ms=22.3,         # 50 mph = 22.352 m/s
    tau_rot_ms=25.0,      # first-order rotor+prop thrust lag
    tau_sens_ms=3.0,      # accel measurement lag (filter)
    tau_obs_ms=30.0,      # disturbance-observer time constant
    loop_hz=1000,         # control loop rate (8 kHz inner loop not needed here)
)


@dataclass
class Airframe:
    mass_kg: float = 0.130
    twr: float = 5.0
    cd_a: float = 0.012
    rho: float = RHO

    @property
    def weight(self) -> float:
        return self.mass_kg * G

    @property
    def thrust_max(self) -> float:
        return self.twr * self.weight

    @property
    def max_horiz_force(self) -> float:
        # at max tilt:  F_h = sqrt(T_max^2 - W^2)
        return self.weight * math.sqrt(self.twr ** 2 - 1.0)


def drag_force(V: float, af: Airframe) -> float:
    return 0.5 * af.rho * V ** 2 * af.cd_a


def static_analysis(af: Airframe, V: float) -> dict:
    """Steady-state trim + margins at wind speed V."""
    W = af.weight
    Fd = drag_force(V, af)
    tilt = math.atan2(Fd, W)
    T_req = math.hypot(W, Fd)
    twr_req = T_req / W
    fh_max = af.max_horiz_force
    if af.cd_a <= 0:
        v_max = math.inf
    else:
        v_max = math.sqrt(fh_max / (0.5 * af.rho * af.cd_a))
    # headroom: exact extra m/s of wind before the horizontal channel
    # saturates (drag grows as V^2, so it saturates exactly at v_max)
    headroom = math.inf if not math.isfinite(v_max) else max(0.0, v_max - V)
    # climb authority once tilted
    climb_margin = (af.thrust_max * math.cos(tilt) - W) / W
    return {
        "wind_ms": V,
        "wind_mph": V * 2.23694,
        "drag_N": Fd,
        "drag_g_equiv": Fd / G * 1000.0,
        "tilt_deg": math.degrees(tilt),
        "twr_required": twr_req,
        "twr_margin_pct": 100.0 * (1.0 - twr_req / af.twr),
        "climb_margin_pct": 100.0 * climb_margin,
        "max_holdable_wind_ms": v_max,
        "max_holdable_wind_mph": v_max * 2.23694,
        "gust_headroom_ms": headroom,
        "gust_headroom_mph": headroom * 2.23694,
    }


def pick_gains(tau_rot: float, omega_n: float = 10.0, zeta: float = 0.9) -> dict:
    """Position-loop gains designed against the actuator lag (PD+I)."""
    kp = omega_n ** 2
    kd = 2.0 * zeta * omega_n
    ki = omega_n / 2.5            # integral against steady wind (still slow)
    return {"kp": kp, "kd": kd, "ki": ki,
            "omega_n": omega_n, "tau_rot": tau_rot}


def gust_wind(t: np.ndarray, v_steady: float, t_gust: float = 0.5) -> np.ndarray:
    """A 0.15 s ramp gust arriving at t_gust seconds, then sustained."""
    rise = 0.15
    ramp = (t - t_gust) / rise
    return v_steady * np.clip(ramp, 0.0, 1.0)


def simulate_gust(af: Airframe, tau_rot: float, tau_sens: float, tau_obs: float,
                  loop_hz: int, feedforward: bool, v_steady: float,
                  t_end: float = 5.0, omega_n: float = 10.0, zeta: float = 0.9,
                  seed_meas_noise: float = 0.01) -> dict:
    """1-DOF horizontal position-hold response to a gust step."""
    dt = 1.0 / loop_hz
    n = int(round(t_end / dt))
    t = np.arange(n) * dt
    gains = pick_gains(tau_rot, omega_n, zeta)
    kp, kd, ki = gains["kp"], gains["kd"], gains["ki"]

    # horizontal authority before throttle saturation
    a_h_max = math.sqrt((af.twr * G) ** 2 - G ** 2)
    u_max = 0.95 * a_h_max

    v_wind = gust_wind(t, v_steady, t_gust=0.5)
    # drag specific force on the horizontal plant (tilt coupling ignored: 1-D)
    w = drag_force(v_wind, af) / af.mass_kg

    x = np.zeros(n)
    v = np.zeros(n)
    u_act = np.zeros(n)          # actual specific horizontal force (lags)
    u_cmd = np.zeros(n)
    integ = np.zeros(n)
    w_hat = 0.0
    a_meas = 0.0
    sat_frac = 0.0
    rng = np.random.default_rng(0)

    for k in range(1, n):
        # plant
        x[k] = x[k - 1] + v[k - 1] * dt
        v[k] = v[k - 1] + (u_act[k - 1] + w[k - 1]) * dt
        # actuator first-order lag (the 15-50 ms bottleneck)
        u_act[k] = u_act[k - 1] + (u_cmd[k - 1] - u_act[k - 1]) * dt / tau_rot
        # sensor: lagged + tiny-noise accel measurement
        a_true = u_act[k] + w[k]
        a_meas += (a_true - a_meas) * dt / tau_sens
        a_meas_n = a_meas + rng.normal(0.0, seed_meas_noise)
        # disturbance observer
        w_hat += ((a_meas_n - u_act[k]) - w_hat) * dt / tau_obs

        e = -x[k]
        integ[k] = integ[k - 1] + e * dt
        u_pd = kp * e + kd * (-v[k]) + ki * integ[k]
        u_cmd[k] = u_pd - (w_hat if feedforward else 0.0)
        # saturation
        if abs(u_cmd[k]) > u_max:
            u_cmd[k] = math.copysign(u_max, u_cmd[k])
            sat_frac += 1.0 / n

    dev = np.abs(x)
    peak = float(dev.max())
    rms = float(np.sqrt(np.mean(dev ** 2)))
    t_peak = float(t[int(np.argmax(dev))])
    # steady-state offset after settling (last 0.5 s) -- the downwind droop
    # a pure PD+I leaves under constant wind because the integral is slow
    ss = float(np.mean(np.abs(x[int(-0.5 / dt):])))
    return {
        "feedforward": feedforward,
        "tau_rot_ms": tau_rot * 1e3,
        "peak_dev_m": peak,
        # how much of the peak is gust *transient* vs sustained-drag droop
        "transient_dev_m": max(0.0, peak - ss),
        "rms_dev_m": rms,
        "t_peak_s": t_peak,
        "steady_err_m": ss,
        "sat_frac": sat_frac,
        # tilt demanded by the commanded horizontal specific force
        "max_tilt_deg": math.degrees(math.atan(np.abs(u_act).max() / G)),
        "gains": gains,
        "trace": {"t": t, "x": x, "u_cmd": u_cmd, "u_act": u_act,
                  "v_wind": v_wind, "w_hat": w_hat},
    }


def _gt(name: str, lo: float):
    """argparse type validator: finite float strictly greater than ``lo``."""
    def _check(v: str) -> float:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"{name} must be a number")
        if not math.isfinite(fv) or fv <= lo:
            raise argparse.ArgumentTypeError(f"{name} must be a finite number > {lo:g}")
        return fv
    return _check


def _ge(name: str, lo: float):
    """argparse type validator: finite float >= ``lo``."""
    def _check(v: str) -> float:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"{name} must be a number")
        if not math.isfinite(fv) or fv < lo:
            raise argparse.ArgumentTypeError(f"{name} must be a finite number >= {lo:g}")
        return fv
    return _check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mass-gram", type=_gt("mass", 0.0), default=DEFAULTS["mass_gram"])
    ap.add_argument("--twr", type=_gt("twr", 1.0), default=DEFAULTS["twr"],
                    help="thrust-to-weight ratio (must exceed 1 to hold altitude)")
    ap.add_argument("--cd-a", type=_gt("cd-a", 0.0), default=DEFAULTS["cd_a"])
    ap.add_argument("--wind-ms", type=_ge("wind-ms", 0.0), default=DEFAULTS["wind_ms"])
    ap.add_argument("--tau-rot-ms", type=_gt("tau-rot", 0.0), default=DEFAULTS["tau_rot_ms"])
    ap.add_argument("--tau-sens-ms", type=_gt("tau-sens", 0.0), default=DEFAULTS["tau_sens_ms"])
    ap.add_argument("--tau-obs-ms", type=_gt("tau-obs", 0.0), default=DEFAULTS["tau_obs_ms"])
    ap.add_argument("--loop-hz", type=_gt("loop-hz", 0.0), default=DEFAULTS["loop_hz"])
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    af = Airframe(mass_kg=args.mass_gram / 1000.0, twr=args.twr,
                  cd_a=args.cd_a, rho=RHO)
    V = args.wind_ms

    # 1) static trim
    static = static_analysis(af, V)
    curve = [static_analysis(af, v) for v in np.linspace(0.5, 30.0, 60)]

    # 2) gust response: tau sweep, PD+I vs PD+I+FF
    taus = [15.0, 25.0, 35.0, 50.0]
    rows_ff = [simulate_gust(af, t / 1e3, args.tau_sens_ms / 1e3,
                             args.tau_obs_ms / 1e3, args.loop_hz,
                             True, V, seed_meas_noise=0.01) for t in taus]
    rows_pd = [simulate_gust(af, t / 1e3, args.tau_sens_ms / 1e3,
                             args.tau_obs_ms / 1e3, args.loop_hz,
                             False, V, seed_meas_noise=0.01) for t in taus]
    gust_default = simulate_gust(af, args.tau_rot_ms / 1e3, args.tau_sens_ms / 1e3,
                                 args.tau_obs_ms / 1e3, args.loop_hz,
                                 True, V, seed_meas_noise=0.01)
    gust_default_pd = simulate_gust(af, args.tau_rot_ms / 1e3, args.tau_sens_ms / 1e3,
                                    args.tau_obs_ms / 1e3, args.loop_hz,
                                    False, V, seed_meas_noise=0.01)

    data = {
        "meta": {
            "model": "Macondo wind-hold feasibility (static trim + 1-DOF gust)",
            "spec": asdict(af),
            "wind_ms": V,
            "loop_hz": args.loop_hz,
            "note": "tau_rot (rotor inertia) dominates peak deviation; "
                    "loop rate 1 kHz is already far past the actuator limit.",
        },
        "static_trim_at_wind": static,
        "static_curve": curve,
        "gust_table": {
            "feedforward_on": [
                {k: r[k] for k in ("tau_rot_ms", "peak_dev_m", "transient_dev_m",
                                   "steady_err_m", "sat_frac", "max_tilt_deg")}
                for r in rows_ff],
            "feedforward_off": [
                {k: r[k] for k in ("tau_rot_ms", "peak_dev_m", "transient_dev_m",
                                   "steady_err_m", "sat_frac", "max_tilt_deg")}
                for r in rows_pd],
        },
        "tuning_targets": {
            "position_loop_hz": args.loop_hz,
            "omega_n_rad_s": gust_default["gains"]["omega_n"],
            "kp": gust_default["gains"]["kp"],
            "kd": gust_default["gains"]["kd"],
            "ki": gust_default["gains"]["ki"],
            "tau_obs_ms": args.tau_obs_ms,
            "observer_gain_hz": 1.0 / (2 * math.pi * args.tau_obs_ms / 1e3),
            "esc_protocol": "DShot600 (1.0-1.5 ms frame) is sufficient; "
                            "bidirectional DShot for RPM filtering",
            "inner_loop_hz": 8000,
            "note": "Enable accel feedforward / disturbance observer; "
                    "PID itself lags because w(t) must first move the drone. "
                    "Rotor time constant is the lever: reduce prop inertia + "
                    "raise TWR to shrink peak deviation.",
        },
        "sim_to_real_map": {
            "50mph_sustained": "expect R_sensor class terms to fall into the "
                               "Cluster/Relay band (<0.75) at sustained 50 mph; "
                               "validate OpenRescue at L3-L5 and add the "
                               "hysteresis dead-band before flying.",
        },
    }

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "macondo_wind_data.json")
    with open(json_path, "w") as f:
        # allow_nan=False: a NaN/Infinity (e.g. from a 0/0 slip) must fail
        # loudly rather than silently write invalid JSON for the paper
        json.dump(data, f, indent=2, allow_nan=False)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Macondo: Wind-Hold Feasibility (sub-250 g, TWR %.1f)" % af.twr)

    ax = axes[0]
    vs = [c["wind_ms"] for c in curve]
    twr_req = [c["twr_required"] for c in curve]
    tilt = [c["tilt_deg"] for c in curve]
    ax.plot(vs, twr_req, "b-", label="Required TWR")
    ax.axhline(af.twr, color="k", ls="--", lw=1.2, label="Airframe TWR %.1f" % af.twr)
    ax.axvline(V, color="r", ls=":", lw=1.4)
    ax.text(V + 0.3, 0.7 * af.twr, "50 mph\n%.0f%% of TWR, %.0f deg tilt"
            % (100 * static["twr_margin_pct"] / 100, static["tilt_deg"]),
            fontsize=9, color="r")
    ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Required TWR")
    ax.set_title("Static trim vs. wind")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    g = gust_default
    gpd = gust_default_pd
    ax.plot(g["trace"]["t"], g["trace"]["x"] * 100.0, "g-", lw=1.6,
            label="PD+I + accel FF (%.1f cm peak)" % (g["peak_dev_m"] * 100))
    ax.plot(gpd["trace"]["t"], gpd["trace"]["x"] * 100.0, "orange", lw=1.4, alpha=0.85,
            label="PD+I only (%.1f cm drift, ~%.0f cm steady droop)"
            % (gpd["peak_dev_m"] * 100, gpd["steady_err_m"] * 100))
    ax.axvline(0.5, color="0.6", ls=":", lw=1)
    ax.annotate("gust hits", (0.5, 0), (0.62, -1.0),
                fontsize=8, color="0.4")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position error (cm)")
    ax.set_title("Gust response at tau_rot=%.0f ms" % (args.tau_rot_ms))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(args.out, "macondo_hold.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    # ---------------- console table ----------------
    print(f"Macondo wind-hold data @ {V:.1f} m/s ({V*2.23694:.0f} mph), "
          f"mass={af.mass_kg*1000:.0f} g, TWR={af.twr}, Cd*A={af.cd_a:.3f} m^2")
    print(f"  static: tilt={static['tilt_deg']:.1f} deg, "
          f"TWR needed={static['twr_required']:.2f} "
          f"(margin {static['twr_margin_pct']:.0f}%), "
          f"climb margin {static['climb_margin_pct']:.0f}%")
    print(f"  max holdable wind: {static['max_holdable_wind_ms']:.1f} m/s "
          f"({static['max_holdable_wind_mph']:.0f} mph); "
          f"gust headroom at 50 mph: {static['gust_headroom_mph']:+.0f} mph")
    print(f"  gust response to a 0.15 s ramp from calm to {static['wind_mph']:.0f} mph:")
    print(f"    {'tau_rot':>8} {'peak':>7} {'transient':>10} {'steady drift':>12} {'sat':>4}")
    for r in data["gust_table"]["feedforward_on"]:
        print(f"    FF on : {r['tau_rot_ms']:>5.0f} ms {r['peak_dev_m']*100:6.1f} cm "
              f"{r['transient_dev_m']*100:9.1f} cm {r['steady_err_m']*100:11.2f} cm "
              f"{r['sat_frac']*100:3.0f}%")
    for r in data["gust_table"]["feedforward_off"]:
        print(f"    FF off: {r['tau_rot_ms']:>5.0f} ms {r['peak_dev_m']*100:6.1f} cm "
              f"{r['transient_dev_m']*100:9.1f} cm {r['steady_err_m']*100:11.2f} cm "
              f"{r['sat_frac']*100:3.0f}%")
    print(f"wrote {json_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()