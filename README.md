# Multi-Agent Reinforcement Learning Car Simulation

A 2D driving simulation where 8 cars learn to drive from start (A) to finish (B) using **tabular Q-learning implemented from scratch**, with no RL libraries. The agents share a single Q-table and learn to follow a curved road, stop at red lights and avoid obstacles.

## Features

- **Q-learning from scratch** with NumPy: epsilon-greedy exploration, a shared Q-table for 8 parallel agents, and persistent save/load
- **Procedural road:** a smooth track generated from control points with Catmull-Rom splines
- **Traffic lights:** 3 lights with offset red/green cycles
- **Obstacles:** pothole and accident types, placed along the track (configurable)
- **Live HUD:** frame count, epsilon, number of visited states, and per-agent episodes, best reward and steps
- **Best-agent highlight:** the top-scoring car is shown in gold with a crown and a fading trail
- **Adjustable simulation speed** (1×–8×) for faster training

## How It Works

### State (discretized)

| Feature | Bins |
|---|---|
| Track progress | 30 |
| Lateral offset from the road center | 5 |
| Heading error relative to the road direction | 5 |
| Speed | 4 |
| Traffic light ahead (none / green / red) | 3 |
| Obstacle proximity (none / near / very near) | 3 |

This gives up to 27,000 discrete states.

### Actions

| # | Action |
|---|---|
| 0 | Steer left |
| 1 | Steer right |
| 2 | Accelerate |
| 3 | Brake |
| 4 | Coast |

### Reward

- **+** progress made along the track
- **−** a small time penalty every step
- **−** distance from the road center
- **−** driving through a red light above a speed threshold
- **−** getting close to an obstacle
- **−12** and episode end when leaving the road
- **−8** and episode end when hitting an obstacle
- **+100** and episode end when reaching the finish

Episodes are also capped at 2,500 steps.

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (α) | 0.10 |
| Discount factor (γ) | 0.95 |
| Epsilon | 1.0 → 0.05, decay 0.99999 per frame |

All values are constants at the top of the script and are easy to tweak.

## Getting Started

```bash
git clone https://github.com/Zanyirin/rl-car-simulation.git
cd rl-car-simulation
pip install -r requirements.txt
python rl_car_simulation.py
```

If `q_table.pkl` exists, training continues from the saved Q-table.

## Controls

| Key | Action |
|---|---|
| `+` / `-` | Increase / decrease simulation speed |
| `S` | Save the Q-table to `q_table.pkl` |
| `SPACE` | Reset all agents |
| `ESC` | Quit |

## Tech Stack

Python · NumPy · Pygame
