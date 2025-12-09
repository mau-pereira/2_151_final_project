# UR5e LQR Control with Force Feedback

This project implements Linear Quadratic Regulator (LQR) controllers for a UR5e robotic arm using analytical equations of motion derived from Lagrangian mechanics.

## Overview

The project contains implementations of LQR controllers for a 3-DOF planar UR5e arm:

- **LQR with Analytical EOM**: Complete LQR controller using derived equations of motion
- **Force-Augmented LQR**: Extended controller that regulates both joint configuration and contact force

## Files

- `lqr_eom_complete_version2.py`: LQR controller using analytical equations of motion
- `lqr_force_version4.py`: Force-augmented LQR controller with normal contact force regulation
- `solve_ik.py`: Inverse kinematics solver
- `ur5e_t.xml`: MuJoCo model file for the UR5e robot
- `scene.xml`: MuJoCo scene configuration
- `meshes/`: 3D mesh files for robot visualization
- `obj_meshes/`: Additional mesh files in OBJ format

## Requirements

- Python 3.x
- NumPy
- SciPy
- SymPy
- MuJoCo
- mujoco-viewer

## Usage

### LQR Controller (Position Control)

```bash
python lqr_eom_complete_version2.py
```

Modify `q_desired` in the script to set the desired joint configuration.

### Force-Augmented LQR Controller

```bash
python lqr_force_version4.py
```

Set both `q_desired` and `F_N_DES` (desired normal force) in the script.

## Features

- Analytical derivation of equations of motion using Lagrangian mechanics
- Symbolic computation of A and B matrices for LQR
- Real-time simulation using MuJoCo
- Force feedback integration for contact tasks

## License

[Add your license here]

