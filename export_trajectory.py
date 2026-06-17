"""
Export a MuJoCo simulation trajectory + plot series to JSON for portfolio playback.

Run from this directory (where scene.xml lives):
    .venv/Scripts/python.exe export_trajectory.py

Output: trajectories/robot_sim_{config}.json
"""

from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np
import scipy.linalg
import sympy as sp

# =============================================================================
# USER CONFIG — edit these, then run the script
# =============================================================================

SIM_TIME_SECONDS = 15.0          # total simulation length [s]
ROBOT_CONFIG = "medium"          # "high" | "medium" | "low" (scene.xml keyframes)
FORCE_ON_WALL_N = 1.0           # desired normal contact force [N]
DISTURBANCE_MAGNITUDE_N = 1.0    # external wrench magnitude on end effector [N]
DISTURBANCE_DIRECTION = np.array([0.0, 0.0, -1.0])  # world-frame direction (normalized internally)

DISTURBANCE_START_TIME = 5.0     # when the disturbance begins [s]
DISTURBANCE_DURATION_SECONDS = 1.0  # how long the disturbance lasts [s]
RECORD_HZ = 50                   # playback / plot sample rate [Hz]
OUTPUT_DIR = "trajectories"

# =============================================================================
# Model helpers
# =============================================================================
WALL_REF_POS = {
    "low": np.array([0.7, 0.135, 0.05]),
    "medium": np.array([0.7, 0.135, 0.5]),
    "high": np.array([0.7, 0.135, 0.8]),
}

VALID_CONFIGS = frozenset(WALL_REF_POS)

_SYM_MATH: dict | None = None


def _load_symbolic_math() -> dict:
    """Reuse EOM + linearization from the proven headless graph script (lines 1–238)."""
    global _SYM_MATH
    if _SYM_MATH is not None:
        return _SYM_MATH

    headless_path = os.path.join(os.path.dirname(__file__), "simulation_wtihout_viewer_for_graphs.py")
    with open(headless_path, encoding="utf-8") as f:
        source = "".join(f.readlines()[:238])

    ns: dict = {"sp": sp, "np": np, "scipy": scipy, "mujoco": mujoco}
    exec(compile(source, headless_path, "exec"), ns)  # noqa: S102
    _SYM_MATH = ns
    return ns


def load_q_desired_from_keyframe(model, data, config: str, qpos_addrs) -> np.ndarray:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config)
    if key_id < 0:
        raise ValueError(f"Unknown keyframe '{config}' in scene.xml")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    return np.array([data.qpos[addr] for addr in qpos_addrs], dtype=float)


def run_export() -> str:
    config = ROBOT_CONFIG.strip().lower()
    if config not in VALID_CONFIGS:
        raise ValueError(f"ROBOT_CONFIG must be one of {sorted(VALID_CONFIGS)}, got '{ROBOT_CONFIG}'")

    if SIM_TIME_SECONDS <= DISTURBANCE_START_TIME:
        raise ValueError("SIM_TIME_SECONDS must be greater than DISTURBANCE_START_TIME")
    if DISTURBANCE_DURATION_SECONDS <= 0:
        raise ValueError("DISTURBANCE_DURATION_SECONDS must be positive")
    if DISTURBANCE_START_TIME + DISTURBANCE_DURATION_SECONDS > SIM_TIME_SECONDS:
        raise ValueError(
            "Disturbance window exceeds simulation: "
            f"start({DISTURBANCE_START_TIME}) + duration({DISTURBANCE_DURATION_SECONDS}) "
            f"> SIM_TIME_SECONDS({SIM_TIME_SECONDS})"
        )

    dir_norm = np.linalg.norm(DISTURBANCE_DIRECTION)
    if dir_norm < 1e-9:
        raise ValueError("DISTURBANCE_DIRECTION must be non-zero")
    dist_unit = DISTURBANCE_DIRECTION / dir_norm
    dist_vector = DISTURBANCE_MAGNITUDE_N * dist_unit
    dist_end_time = DISTURBANCE_START_TIME + DISTURBANCE_DURATION_SECONDS

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    sym = _load_symbolic_math()
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

    q_desired = load_q_desired_from_keyframe(model, data, config, qpos_addrs)
    qdot_desired = np.zeros(3)
    n_states_joint = 6
    n_states_aug = 7
    n_controls = 3
    x_desired_joint = np.zeros(n_states_joint)
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
    k_c_eff = 1500.0
    C_F = np.zeros(n_states_joint)
    C_F[:3] = k_c_eff * (n_hat @ J_eq)

    A_aug = np.zeros((n_states_aug, n_states_aug))
    A_aug[:n_states_joint, :n_states_joint] = A
    A_aug[n_states_joint, :n_states_joint] = C_F
    B_aug = np.zeros((n_states_aug, n_controls))
    B_aug[:n_states_joint, :] = B

    Q_joint = np.eye(n_states_joint)
    Q_joint[:3, :3] *= 500.0
    Q_joint[3:, 3:] *= 50.0
    Q_aug = np.zeros((n_states_aug, n_states_aug))
    Q_aug[:n_states_joint, :n_states_joint] = Q_joint
    Q_aug[n_states_joint, n_states_joint] = 50.0
    R = np.eye(n_controls) * 0.5

    P = scipy.linalg.solve_continuous_are(A_aug, B_aug, Q_aug, R)
    K_aug = np.linalg.solve(R, B_aug.T @ P)
    print(f"LQI gains ready for config='{config}', F_N_des={FORCE_ON_WALL_N} N")

    dt = model.opt.timestep
    n_steps = int(round(SIM_TIME_SECONDS / dt))
    sample_every = max(1, int(round(1.0 / (RECORD_HZ * dt))))

    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] = q_desired[i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Desired end-effector position is the EE site at the target configuration,
    # captured above as x_ee_eq (site_xpos while qpos == q_desired).
    ee_pos_desired = x_ee_eq

    z_f = 0.0
    times: list[float] = []
    qpos_series: list[list[float]] = []
    disturbance_n: list[float] = []
    force_on_wall_err_n: list[float] = []
    ee_pos_err_m: list[float] = []

    print(
        f"Simulating {SIM_TIME_SECONDS}s @ dt={dt:.4f}, "
        f"disturbance {DISTURBANCE_START_TIME}s->{dist_end_time}s, "
        f"|F_dist|={DISTURBANCE_MAGNITUDE_N} N..."
    )

    for step in range(n_steps):
        t_now = step * dt

        x_joint = np.zeros(n_states_joint)
        x_joint[:3] = [data.qpos[addr] for addr in qpos_addrs]
        x_joint[3:] = [data.qvel[addr] for addr in qvel_addrs]

        F_site = data.sensordata[ee_force_adr : ee_force_adr + ee_force_dim].copy()
        R_site = data.site_xmat[ee_site_id].reshape(3, 3)
        F_world = R_site @ F_site
        F_n_meas = float(np.dot(F_world, n_hat))

        z_f += (F_n_meas - FORCE_ON_WALL_N) * dt

        x_err_aug = np.zeros(n_states_aug)
        x_err_aug[:n_states_joint] = x_joint - x_desired_joint
        x_err_aug[-1] = z_f
        u = u_ff - K_aug @ x_err_aug
        for i, act_id in enumerate(actuator_ids):
            data.ctrl[act_id] = u[i]

        data.qfrc_applied[:] = 0.0
        dist_active = DISTURBANCE_START_TIME <= t_now < dist_end_time
        if dist_active:
            ee_pos = data.site_xpos[ee_site_id].copy()
            force_world = dist_vector.reshape(3, 1)
            torque_world = np.zeros((3, 1))
            qfrc_target = np.zeros((model.nv, 1))
            mujoco.mj_applyFT(
                model, data, force_world, torque_world,
                ee_pos.reshape(3, 1), ee_body_id, qfrc_target,
            )
            data.qfrc_applied[:] = qfrc_target.flatten()

        mujoco.mj_step(model, data)

        if step % sample_every == 0:
            # Refresh forward kinematics so site_xpos matches the freshly stepped
            # qpos, then measure how far the end effector is from its target pose.
            mujoco.mj_kinematics(model, data)
            ee_pos_now = data.site_xpos[ee_site_id]
            ee_pos_err = float(np.linalg.norm(ee_pos_now - ee_pos_desired))

            times.append(round(t_now, 4))
            qpos_series.append([round(float(data.qpos[addr]), 6) for addr in qpos_addrs])
            disturbance_n.append(round(DISTURBANCE_MAGNITUDE_N if dist_active else 0.0, 4))
            force_on_wall_err_n.append(round(F_n_meas - FORCE_ON_WALL_N, 4))
            ee_pos_err_m.append(round(ee_pos_err, 6))

    payload = {
        "schema": 1,
        "config": config,
        "duration": SIM_TIME_SECONDS,
        "recordHz": RECORD_HZ,
        "targetNormalForce": FORCE_ON_WALL_N,
        "qDesired": [round(float(q), 6) for q in q_desired],
        "disturbance": {
            "tStart": DISTURBANCE_START_TIME,
            "tEnd": dist_end_time,
            "duration": DISTURBANCE_DURATION_SECONDS,
            "magnitude": DISTURBANCE_MAGNITUDE_N,
            "direction": [round(float(v), 6) for v in dist_unit],
            "vector": [round(float(v), 6) for v in dist_vector],
        },
        "time": times,
        "qpos": qpos_series,
        "disturbanceN": disturbance_n,
        "forceOnWallErrN": force_on_wall_err_n,
        "eePosErrM": ee_pos_err_m,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"robot_sim_{config}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"Wrote {out_path} ({len(times)} samples, {os.path.getsize(out_path) / 1024:.1f} KiB)")
    return out_path


if __name__ == "__main__":
    try:
        run_export()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
