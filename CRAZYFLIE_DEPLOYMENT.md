
# MARAHS Policy Export & Deployment Notes

> **Correction (audit):** this file previously claimed *"MARAHS holds position
> within ±0.2 m at wind up to 25 m/s"*. No code ever measured that — the export
> script was serializing **randomly initialized weights** and the sim numbers
> were fabricated. Those claims are removed. Nothing in this repo has been
> measured against physical wind yet.

## What the trained artifacts actually are

The `.pt` checkpoints in this repo are **high-level grid policies**
(PPONetwork MLPs, obs 496–656 → 5 discrete scan actions) trained on wildfire
perimeter tracking — **not** 6-D continuous wind-hold controllers. Their
native-task performance vs hand-crafted baselines is measured honestly by
`python crazyflie_deploy.py --checkpoint <file> --benchmark`, which writes
`learned_vs_handcrafted.json`.

`python crazyflie_deploy.py --checkpoint <file> --export` serializes the real
actor weights (encoder + LayerNorm + policy head, dimensions read from the
checkpoint tensors) to `crazyflie_policy.h` and ONNX. GAT checkpoints are
refused: their actor needs the runtime communication graph.

## Open engineering gaps (not yet built — do not skip)

1. **Observation bridge**: the actor expects the full grid observation
   (local field windows + global scalars). Mapping real onboard telemetry
   (map/RSSI) to that vector is undefined.
2. **Action bridge**: discrete grid actions (hover / N / S / E / W) must
   become velocity setpoints for the flight controller.
3. **Low-level wind hold**: holding position in strong wind is a
   flight-controller problem (see `macondo_hover.py` analysis); it is not
   something this grid policy does.
4. **Physical measurement**: the garden leaf-blower protocol below is the
   experiment that would produce real numbers — none exist yet.

## Hardware (for the physical experiment)
- Bitcraze Crazyflie 2.1 ($199): https://www.bitcraze.io/products/crazyflie-2-1/
- Flow deck v2 ($79): https://www.bitcraze.io/products/flow-deck-v2/
- Crazyradio PA ($39): https://www.bitcraze.io/products/crazyradio-pa/
- High-velocity fan (15-25 m/s): ~$50-100

**Total cost: ~$370-420**

## Software Setup
```bash
# Install Crazyflie client
pip install cflib

# Flash firmware with custom deck support
cd crazyflie-firmware
make BOARD=cf21 LIB=IMU_BIMU deck-flow

# Connect Crazyflie and flash
cfclient
```

## Experimental Protocol (when the bridges above exist)
1. Export the real actor: `python crazyflie_deploy.py --checkpoint ppo_best.pt --export`
2. Place fan at 1.5m distance from Crazyflie
3. Set fan speed to target wind speed (measure with anemometer)
4. Launch Crazyflie and command position hold at (0, 0)
5. Record position data via Crazyradio for 30 seconds
6. Repeat at wind speeds: 5, 10, 15, 20, 25 m/s

## Status
- Sim-task benchmark (learned vs Random / Greedy on fresh episodes): measured,
  see `learned_vs_handcrafted.json`.
- Physical wind-hold: **not measured**. Any result claimed here must come from
  this protocol, not from a sim or a docstring.

## Data Format
Log files are CSV with columns:
- timestamp, x, y, z, vx, vy, vz, wind_x, wind_y
- Logged at 100Hz via Crazyradio

## Citation
If you use this deployment code, please cite:
@article{basu2026marahs,
  title={MARAHS: Multi-Agent Robust Autonomous Hazard Swarm},
  author={Basu, Shaurjesh},
  year={2026}
}
