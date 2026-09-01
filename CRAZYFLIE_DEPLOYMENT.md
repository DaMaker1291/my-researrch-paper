
# MARAHS Crazyflie Deployment Guide

## Hardware Required
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

## Policy Deployment
1. Export policy: `python crazyflie_deploy.py --export`
2. Copy `crazyflie_policy.h` to firmware/src/decks/
3. Rebuild and flash firmware
4. Policy runs at 100Hz on Crazyflie's STM32F4 MCU

## Experimental Protocol
1. Place fan at 1.5m distance from Crazyflie
2. Set fan speed to target wind speed (measure with anemometer)
3. Launch Crazyflie and command position hold at (0, 0)
4. Record position data via Crazyradio for 30 seconds
5. Repeat at wind speeds: 5, 10, 15, 20, 25 m/s
6. Compare PID vs MARAHS position error

## Expected Results
- PID: drifts >0.5m at wind >15 m/s, crashes at >20 m/s
- MARAHS: holds position within ±0.2m at wind up to 25 m/s

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
