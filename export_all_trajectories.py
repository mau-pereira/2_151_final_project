"""
Batch-export every test trajectory to JSON for the portfolio MuJoCo lab.

Generates the full menu matrix:
    position    in {low, medium, high}              (scene.xml keyframes)
    force       in {0, 1, 10} N                     (desired normal wall force)
    disturbance in {0, 1, 10} N                     (external pull magnitude)
    direction   in {down, upleft}                   (only when magnitude > 0)

The expensive symbolic EOM derivation runs once, and the LQI controller is
solved once per position (reused across every force/disturbance combination).

Run from this directory (where scene.xml lives):
    .venv/Scripts/python.exe export_all_trajectories.py

Outputs: trajectories/{pos}_f{force}_d{mag}_{dir}.json
"""
from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np
import scipy.linalg

from export_trajectory import (
    WALL_REF_POS,
    _load_symbolic_math,
    load_q_desired_from_keyframe,
)

# =============================================================================
# TEST MATRIX — every combination below is exported to its own JSON file
# =============================================================================

POSITIONS = ["low", "medium", "high"]
FORCES_N = [0.0, 1.0, 10.0]
DISTURBANCES_N = [0.0, 1.0, 10.0]

# World-frame directions (wall sits at +x, so -x points away from the wall).
DIRECTIONS = {
    "down": np.array([0.0, 0.0, -1.0]),       # straight down
    "upleft": np.array([-1.0, 0.0, 1.0]),     # up and away from the wall
}

SIM_TIME_SECONDS = 15.0
DISTURBANCE_START_TIME = 5.0
DISTURBANCE_DURATION_SECONDS = 1.0
RECORD_HZ = 50
OUTPUT_DIR = "trajectories"

# LQI / contact model tuning (matches export_trajectory.py)
CONTACT_STIFFNESS_N_PER_M = 1500.0
Q_POSITION_WEIGHT = 500.0
Q_VELOCITY_WEIGHT = 50.0
Q_FORCE_INTEGRAL_WEIGHT = 50.0
R_CONTROL_WEIGHT = 0.5


def build_controller(config: str, sym: dict) -> dict:
    """Solve the augmented LQI controller for a single wall configuration."""
    compute_linearization_from_eom = sym["compute_linearization_from_eom"]
    ee_jacobian = sym["ee_jacobian"]
    param_values = sym["param_values"]

    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    joint_names = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint"]
    actuator_names = ["torq_j2", "torq_j3", "torq_j4"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    qpos_addrs = [model.joint(jid).qposadr[0] for jid in joint_ids]
    qvel_addrs = [model.joint(jid).dofadr[0] for jid in joint_ids]
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_2_link")
    ee_force_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force")
    ee_force_adr = model.sensor_adr[ee_force_sid]
    ee_force_dim = model.sensor_dim[ee_force_sid]
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config)

    q_desired = load_q_desired_from_keyframe(model, data, config, qpos_addrs)
    qdot_desired = np.zeros(3)
    x_desired_joint = np.zeros(6)
    x_desired_joint[:3] = q_desired

    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] = q_desired[i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    data.ctrl[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    u_ff = data.qfrc_inverse[qvel_addrs].copy()

    A, B = compute_linearization_from_eom(q_desired, qdot_desired, u_ff, param_values)

    x_ee_eq = data.site_xpos[ee_site_id].copy()
    x_wall_eq = WALL_REF_POS[config].copy()
    n_hat = np.array([1.0, 0.0, 0.0])
    if x_ee_eq[0] > x_wall_eq[0]:
        n_hat = -n_hat
    n_hat /= np.linalg.norm(n_hat)

    J_eq = ee_jacobian(model, data, ee_site_id, qvel_addrs)
    C_F = np.zeros(6)
    C_F[:3] = CONTACT_STIFFNESS_N_PER_M * (n_hat @ J_eq)

    A_aug = np.zeros((7, 7))
    A_aug[:6, :6] = A
    A_aug[6, :6] = C_F
    B_aug = np.zeros((7, 3))
    B_aug[:6, :] = B

    Q_aug = np.zeros((7, 7))
    Q_aug[:3, :3] = np.eye(3) * Q_POSITION_WEIGHT
    Q_aug[3:6, 3:6] = np.eye(3) * Q_VELOCITY_WEIGHT
    Q_aug[6, 6] = Q_FORCE_INTEGRAL_WEIGHT
    R = np.eye(3) * R_CONTROL_WEIGHT

    P = scipy.linalg.solve_continuous_are(A_aug, B_aug, Q_aug, R)
    K_aug = np.linalg.solve(R, B_aug.T @ P)

    print(f"  Controller ready for config='{config}', q_desired={q_desired}")

    return {
        "config": config,
        "model": model,
        "data": data,
        "key_id": key_id,
        "qpos_addrs": qpos_addrs,
        "qvel_addrs": qvel_addrs,
        "actuator_ids": actuator_ids,
        "ee_site_id": ee_site_id,
        "ee_body_id": ee_body_id,
        "ee_force_adr": ee_force_adr,
        "ee_force_dim": ee_force_dim,
        "q_desired": q_desired,
        "x_desired_joint": x_desired_joint,
        "ee_pos_desired": x_ee_eq,
        "n_hat": n_hat,
        "u_ff": u_ff,
        "K_aug": K_aug,
    }


def simulate(ctrl: dict, force_target: float, dist_vector: np.ndarray) -> dict:
    """Run one closed-loop episode and return the recorded JSON payload."""
    model = ctrl["model"]
    data = ctrl["data"]
    qpos_addrs = ctrl["qpos_addrs"]
    qvel_addrs = ctrl["qvel_addrs"]
    actuator_ids = ctrl["actuator_ids"]
    ee_site_id = ctrl["ee_site_id"]
    ee_force_adr = ctrl["ee_force_adr"]
    ee_force_dim = ctrl["ee_force_dim"]
    n_hat = ctrl["n_hat"]
    x_desired_joint = ctrl["x_desired_joint"]
    ee_pos_desired = ctrl["ee_pos_desired"]
    u_ff = ctrl["u_ff"]
    K_aug = ctrl["K_aug"]

    dt = model.opt.timestep
    n_steps = int(round(SIM_TIME_SECONDS / dt))
    sample_every = max(1, int(round(1.0 / (RECORD_HZ * dt))))
    dist_end_time = DISTURBANCE_START_TIME + DISTURBANCE_DURATION_SECONDS
    dist_mag = float(np.linalg.norm(dist_vector))
    has_disturbance = dist_mag > 1e-9

    # Clean, identical starting state for every episode.
    mujoco.mj_resetDataKeyframe(model, data, ctrl["key_id"])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    z_f = 0.0
    times: list[float] = []
    qpos_series: list[list[float]] = []
    disturbance_n: list[float] = []
    force_on_wall_err_n: list[float] = []
    ee_pos_err_m: list[float] = []

    for step in range(n_steps):
        t_now = step * dt

        x_joint = np.zeros(6)
        x_joint[:3] = [data.qpos[addr] for addr in qpos_addrs]
        x_joint[3:] = [data.qvel[addr] for addr in qvel_addrs]

        F_site = data.sensordata[ee_force_adr : ee_force_adr + ee_force_dim].copy()
        R_site = data.site_xmat[ee_site_id].reshape(3, 3)
        F_world = R_site @ F_site
        F_n_meas = float(np.dot(F_world, n_hat))

        z_f += (F_n_meas - force_target) * dt

        x_err_aug = np.zeros(7)
        x_err_aug[:6] = x_joint - x_desired_joint
        x_err_aug[-1] = z_f
        u = u_ff - K_aug @ x_err_aug
        for i, act_id in enumerate(actuator_ids):
            data.ctrl[act_id] = u[i]

        data.qfrc_applied[:] = 0.0
        dist_active = has_disturbance and (DISTURBANCE_START_TIME <= t_now < dist_end_time)
        if dist_active:
            ee_pos = data.site_xpos[ee_site_id].copy()
            qfrc_target = np.zeros((model.nv, 1))
            mujoco.mj_applyFT(
                model, data, dist_vector.reshape(3, 1), np.zeros((3, 1)),
                ee_pos.reshape(3, 1), ctrl["ee_body_id"], qfrc_target,
            )
            data.qfrc_applied[:] = qfrc_target.flatten()

        mujoco.mj_step(model, data)

        if step % sample_every == 0:
            mujoco.mj_kinematics(model, data)
            ee_pos_now = data.site_xpos[ee_site_id]
            ee_pos_err = float(np.linalg.norm(ee_pos_now - ee_pos_desired))

            times.append(round(t_now, 4))
            qpos_series.append([round(float(data.qpos[addr]), 6) for addr in qpos_addrs])
            disturbance_n.append(round(dist_mag if dist_active else 0.0, 4))
            force_on_wall_err_n.append(round(F_n_meas - force_target, 4))
            ee_pos_err_m.append(round(ee_pos_err, 6))

    return {
        "time": times,
        "qpos": qpos_series,
        "disturbanceN": disturbance_n,
        "forceOnWallErrN": force_on_wall_err_n,
        "eePosErrM": ee_pos_err_m,
    }


def run_all() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    sym = _load_symbolic_math()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dist_end_time = DISTURBANCE_START_TIME + DISTURBANCE_DURATION_SECONDS
    written = 0

    for config in POSITIONS:
        print(f"\n=== Position: {config} ===")
        ctrl = build_controller(config, sym)

        for force in FORCES_N:
            for mag in DISTURBANCES_N:
                if mag == 0.0:
                    combos = [("none", np.zeros(3))]
                else:
                    combos = list(DIRECTIONS.items())

                for dir_name, unit in combos:
                    if mag == 0.0:
                        dist_vector = np.zeros(3)
                        dist_unit = np.zeros(3)
                    else:
                        dist_unit = unit / np.linalg.norm(unit)
                        dist_vector = mag * dist_unit

                    series = simulate(ctrl, force, dist_vector)

                    payload = {
                        "schema": 1,
                        "config": config,
                        "duration": SIM_TIME_SECONDS,
                        "recordHz": RECORD_HZ,
                        "targetNormalForce": force,
                        "disturbance": {
                            "magnitude": mag,
                            "direction": dir_name,
                            "tStart": DISTURBANCE_START_TIME,
                            "tEnd": dist_end_time if mag > 0 else 0.0,
                            "vector": [round(float(v), 6) for v in dist_vector],
                            "unit": [round(float(v), 6) for v in dist_unit],
                        },
                        **series,
                    }

                    fname = f"{config}_f{int(force)}_d{int(mag)}_{dir_name}.json"
                    out_path = os.path.join(OUTPUT_DIR, fname)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, separators=(",", ":"))

                    written += 1
                    print(
                        f"  wrote {fname} "
                        f"({len(series['time'])} samples, {os.path.getsize(out_path) / 1024:.1f} KiB)"
                    )

    print(f"\nDone. Wrote {written} trajectory files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    try:
        run_all()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
