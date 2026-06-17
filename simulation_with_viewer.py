"""
Force-augmented LQR controller using analytical EOM for 3-link arm.

Interactive MuJoCo viewer for the UR5e wall-contact test. Edit USER CONFIG at
the top, then run from this directory:

    .venv/Scripts/python.exe simulation_with_viewer.py

The viewer runs until you close the window. For JSON export (portfolio playback),
use export_trajectory.py with the same USER CONFIG fields.
"""
from __future__ import annotations

import sys

import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
import sympy as sp

# =============================================================================
# USER CONFIG — edit these, then run the script
# =============================================================================

ROBOT_CONFIG = "medium"          # "high" | "medium" | "low" (scene.xml keyframes)
FORCE_ON_WALL_N = 10.0           # desired normal contact force [N]

DISTURBANCE_MAGNITUDE_N = 1.0   # external wrench magnitude on end effector [N]
DISTURBANCE_DIRECTION = np.array([1.0, 0.0, 1.0])  # world-frame direction (normalized internally)
DISTURBANCE_START_TIME = 5.0     # when the disturbance begins [s]
DISTURBANCE_DURATION_SECONDS = 1.0  # how long the disturbance lasts [s]

# Minimum simulation horizon used to validate the disturbance window. The viewer
# itself runs until you close the window (not capped at this value).
SIM_TIME_SECONDS = 15.0

# LQI / contact model tuning
CONTACT_STIFFNESS_N_PER_M = 1500.0
Q_POSITION_WEIGHT = 500.0
Q_VELOCITY_WEIGHT = 50.0
Q_FORCE_INTEGRAL_WEIGHT = 50.0
R_CONTROL_WEIGHT = 0.5

# MuJoCo passive viewer camera (MuJoCo world frame: x, y, z)
VIEWER_CAMERA_LOOKAT = (0.35, 0.1, 0.3)
VIEWER_CAMERA_DISTANCE = 2.0
VIEWER_CAMERA_AZIMUTH = 135
VIEWER_CAMERA_ELEVATION = -15

LOG_EVERY_N_STEPS = 50

# =============================================================================
# Model helpers
# =============================================================================

WALL_REF_POS = {
    "low": np.array([0.7, 0.135, 0.05]),
    "medium": np.array([0.7, 0.135, 0.5]),
    "high": np.array([0.7, 0.135, 0.8]),
}

VALID_CONFIGS = frozenset(WALL_REF_POS)


def validate_user_config() -> None:
    config = ROBOT_CONFIG.strip().lower()
    if config not in VALID_CONFIGS:
        raise ValueError(f"ROBOT_CONFIG must be one of {sorted(VALID_CONFIGS)}, got '{ROBOT_CONFIG}'")
    if SIM_TIME_SECONDS <= DISTURBANCE_START_TIME:
        raise ValueError("SIM_TIME_SECONDS must be greater than DISTURBANCE_START_TIME")
    if DISTURBANCE_DURATION_SECONDS <= 0:
        raise ValueError("DISTURBANCE_DURATION_SECONDS must be positive")
    if DISTURBANCE_START_TIME + DISTURBANCE_DURATION_SECONDS > SIM_TIME_SECONDS:
        raise ValueError(
            "Disturbance window exceeds SIM_TIME_SECONDS: "
            f"start({DISTURBANCE_START_TIME}) + duration({DISTURBANCE_DURATION_SECONDS}) "
            f"> SIM_TIME_SECONDS({SIM_TIME_SECONDS})"
        )
    if np.linalg.norm(DISTURBANCE_DIRECTION) < 1e-9:
        raise ValueError("DISTURBANCE_DIRECTION must be non-zero")


def load_q_desired_from_keyframe(model, data, config: str, qpos_addrs) -> np.ndarray:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config)
    if key_id < 0:
        raise ValueError(f"Unknown keyframe '{config}' in scene.xml")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    return np.array([data.qpos[addr] for addr in qpos_addrs], dtype=float)


def normalized_disturbance_vector() -> tuple[np.ndarray, np.ndarray, float]:
    dist_unit = DISTURBANCE_DIRECTION / np.linalg.norm(DISTURBANCE_DIRECTION)
    dist_vector = DISTURBANCE_MAGNITUDE_N * dist_unit
    dist_end_time = DISTURBANCE_START_TIME + DISTURBANCE_DURATION_SECONDS
    return dist_unit, dist_vector, dist_end_time

# Time variable
t = sp.symbols('t')
# Symbols
# Generalized coordinates q_i(t)
q1, q2, q3 = sp.symbols('q1 q2 q3', cls=sp.Function)
q = [q1(t), q2(t), q3(t)]
# Velocities
q1dot, q2dot, q3dot = [sp.diff(q1(t), t), sp.diff(q2(t), t), sp.diff(q3(t), t)]
qdot = [q1dot, q2dot, q3dot]
# Accelerations
q1ddot = sp.diff(q1dot, t)
q2ddot = sp.diff(q2dot, t)
q3ddot = sp.diff(q3dot, t)

# Parameters (symbolic)
I1, I2, I3 = sp.symbols('I1 I2 I3', real=True, positive=True)  # Inertias (kg·m²)
m1, m2, m3 = sp.symbols('m1 m2 m3', real=True, positive=True)  # Masses (kg)
L1, L2 = sp.symbols('L1 L2', real=True, positive=True)  # Link lengths (m)
L3com = sp.symbols('L3com', real=True, positive=True)  # COM offset of link 3
g = sp.symbols('g', real=True, positive=True)  # Gravitational acceleration (m/s²)
# Known numeric values for physical parameters (UR5e)
param_values = {
    'I1': 0.133886,
    'I2': 0.0311796,
    'I3': 0.011752,
    'm1': 8.393,
    'm2': 2.275,
    'm3': 2.629,
    'L1': 0.425,
    'L2': 0.392,
    'L3com': 0.053595,
    'g': 9.81,
}


# ============================================================================
# PART 1: DERIVE EQUATIONS OF MOTION
# ============================================================================
print("="*70)
print("PART 1: Deriving Equations of Motion")
print("="*70)
# Define COM positions
x1 = (L1/2)*sp.cos(q1(t))
y1 = 0.163 + (L1/2)*sp.sin(q1(t))

x2 = L1*sp.cos(q1(t)) + (L2/2)*sp.cos(q1(t)+q2(t))
y2 = 0.163 + L1*sp.sin(q1(t)) + (L2/2)*sp.sin(q1(t)+q2(t))

x3 = L1*sp.cos(q1(t)) + L2*sp.cos(q1(t)+q2(t)) + L3com*sp.cos(q1(t)+q2(t)+q3(t))
y3 = 0.163 + L1*sp.sin(q1(t)) + L2*sp.sin(q1(t)+q2(t)) + L3com*sp.sin(q1(t)+q2(t)+q3(t))

# Potential energy
V = m1*g*y1 + m2*g*y2 + m3*g*y3

# Kinetic energy
# Velocities (chain rule automatically applied)
dx1 = sp.diff(x1, q1(t))*q1dot
dy1 = sp.diff(y1, q1(t))*q1dot

dx2 = sp.diff(x2, q1(t))*q1dot + sp.diff(x2, q2(t))*q2dot
dy2 = sp.diff(y2, q1(t))*q1dot + sp.diff(y2, q2(t))*q2dot

dx3 = (sp.diff(x3, q1(t))*q1dot + sp.diff(x3, q2(t))*q2dot + sp.diff(x3, q3(t))*q3dot)
dy3 = (sp.diff(y3, q1(t))*q1dot + sp.diff(y3, q2(t))*q2dot + sp.diff(y3, q3(t))*q3dot)

# Kinetic energy
T = (0.5*m1*(dx1**2 + dy1**2) + 0.5*I1*q1dot**2
   + 0.5*m2*(dx2**2 + dy2**2) + 0.5*I2*(q1dot+q2dot)**2
   + 0.5*m3*(dx3**2 + dy3**2) + 0.5*I3*(q1dot+q2dot+q3dot)**2)

# Lagrangian
L = T - V

# Equations of Motion: d/dt(∂L/∂q̇) - ∂L/∂q = τ
tau1, tau2, tau3 = sp.symbols('tau1 tau2 tau3', cls=sp.Function)
tau = [tau1(t), tau2(t), tau3(t)]

# Compute partial derivatives
dL_dq1dot = sp.diff(L, q1dot)
dL_dq2dot = sp.diff(L, q2dot)
dL_dq3dot = sp.diff(L, q3dot)

dL_dq1 = sp.diff(L, q1(t))
dL_dq2 = sp.diff(L, q2(t))
dL_dq3 = sp.diff(L, q3(t))

# Compute time derivatives of ∂L/∂q̇
ddt_dL_dq1dot = sp.diff(dL_dq1dot, t)
ddt_dL_dq2dot = sp.diff(dL_dq2dot, t)
ddt_dL_dq3dot = sp.diff(dL_dq3dot, t)

# Equations of motion: d/dt(∂L/∂q̇) - ∂L/∂q = τ
eom1 = ddt_dL_dq1dot - dL_dq1 - tau1(t)
eom2 = ddt_dL_dq2dot - dL_dq2 - tau2(t)
eom3 = ddt_dL_dq3dot - dL_dq3 - tau3(t)

eom = [eom1, eom2, eom3]

# Simplify the equations
print("Simplifying equations of motion...")
eom_simplified = [sp.simplify(eq) for eq in eom]
print("  EOM simplified")


def ee_jacobian(model, data, site_id, qvel_addrs):
    """
    3x3 translational Jacobian of the EE site with respect to the selected
    3 DOFs (shoulder_lift, elbow, wrist_1).

    Used to approximate how joint deviations change EE position (and thus normal
    contact force) around the equilibrium.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J_full = jacp
    J = J_full[:, qvel_addrs]
    return J

###############################################################################
# PART 2: LINEARIZATION DIRECTLY FROM EOM (NO MANUAL M, C, G EXTRACTION)
###############################################################################

def compute_linearization_from_eom(q_eq, qdot_eq, tau_eq, param_values):
    """
    Compute A and B matrices directly from the EOM using symbolic Jacobians.
    This keeps everything derived from the EOM without manually extracting M, C, G.
    """
    print("\nComputing linearization matrices directly from EOM...")

    # 1) Simple symbols (no Derivative objects)
    q1s, q2s, q3s = sp.symbols('q1s q2s q3s', real=True)
    v1s, v2s, v3s = sp.symbols('v1s v2s v3s', real=True)   # velocities
    a1s, a2s, a3s = sp.symbols('a1s a2s a3s', real=True)   # accelerations
    tau1s, tau2s, tau3s = sp.symbols('tau1s tau2s tau3s', real=True)

    # Map function-of-time variables and derivatives to these simple symbols
    subs_dyn = {
        q1(t): q1s,
        q2(t): q2s,
        q3(t): q3s,
        q1dot: v1s,
        q2dot: v2s,
        q3dot: v3s,
        q1ddot: a1s,
        q2ddot: a2s,
        q3ddot: a3s,
        tau1(t): tau1s,
        tau2(t): tau2s,
        tau3(t): tau3s,
    }

    # EOM rewritten in terms of simple symbols (no Derivative objects)
    eom_sym = [eq.subs(subs_dyn) for eq in eom_simplified]

    # 2) Linear system in accelerations: M_sym * a = h_sym
    M_sym, h_sym = sp.linear_eq_to_matrix(eom_sym, (a1s, a2s, a3s))
    # EOM: M_sym * a - h_sym = 0  =>  M_sym * a = h_sym

    # Solve symbolically for accelerations a = f(q, v, tau)
    a_vec = M_sym.LUsolve(h_sym)  # shape (3, 1)

    # 3) Build state-space dynamics: xdot = f(x, u)
    #    x = [q1, q2, q3, v1, v2, v3], u = [tau1, tau2, tau3]
    x_sym = [q1s, q2s, q3s, v1s, v2s, v3s]
    u_sym = [tau1s, tau2s, tau3s]

    f_sym = sp.Matrix([
        v1s,
        v2s,
        v3s,
        a_vec[0],
        a_vec[1],
        a_vec[2],
    ])

    # 4) Symbolic Jacobians: A_sym = df/dx, B_sym = df/du
    print("  Computing symbolic Jacobians df/dx and df/du ...")
    A_sym = f_sym.jacobian(x_sym)
    B_sym = f_sym.jacobian(u_sym)

    # 5) Substitute physical parameters and equilibrium point
    print("  Substituting parameters and equilibrium values...")

    # Parameter substitution dictionary
    param_subs = {
        I1: param_values['I1'],
        I2: param_values['I2'],
        I3: param_values['I3'],
        m1: param_values['m1'],
        m2: param_values['m2'],
        m3: param_values['m3'],
        L1: param_values['L1'],
        L2: param_values['L2'],
        L3com: param_values['L3com'],
        g:  param_values['g'],
    }

    # Equilibrium state and input
    eq_subs = {
        q1s: q_eq[0],
        q2s: q_eq[1],
        q3s: q_eq[2],
        v1s: qdot_eq[0],
        v2s: qdot_eq[1],
        v3s: qdot_eq[2],
        tau1s: tau_eq[0],
        tau2s: tau_eq[1],
        tau3s: tau_eq[2],
    }

    A_num = np.zeros((6, 6))
    B_num = np.zeros((6, 3))

    def eval_num(expr):
        expr_sub = expr.subs(param_subs).subs(eq_subs)
        return float(expr_sub.evalf())

    for i in range(6):
        for j in range(6):
            A_num[i, j] = eval_num(A_sym[i, j])
        for j in range(3):
            B_num[i, j] = eval_num(B_sym[i, j])

    print("  Linearization matrices A and B computed from EOM.")
    return A_num, B_num



def run_simulation() -> None:
    validate_user_config()
    config = ROBOT_CONFIG.strip().lower()
    dist_unit, dist_vector, dist_end_time = normalized_disturbance_vector()

    n_states_joint = 6
    n_states_aug = 7
    n_controls = 3

    print("\n" + "=" * 70)
    print("PART 2: Computing Linearization Matrices (Joint States Only)")
    print("=" * 70)

    print("\nComputing feedforward torques (gravity compensation)...")
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    joint_names = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    qpos_addrs = [model.joint(joint_id).qposadr[0] for joint_id in joint_ids]
    qvel_addrs = [model.joint(joint_id).dofadr[0] for joint_id in joint_ids]
    actuator_names = ["torq_j2", "torq_j3", "torq_j4"]
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_2_link")
    ee_force_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force")
    ee_force_adr = model.sensor_adr[ee_force_sid]
    ee_force_dim = model.sensor_dim[ee_force_sid]

    q_desired = load_q_desired_from_keyframe(model, data, config, qpos_addrs)
    qdot_desired = np.zeros(3)

    x_desired_joint = np.zeros(n_states_joint)
    x_desired_joint[:3] = q_desired
    x_desired_joint[3:] = qdot_desired

    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] = q_desired[i]
    for i, addr in enumerate(qvel_addrs):
        data.qvel[addr] = qdot_desired[i]

    mujoco.mj_forward(model, data)

    data.ctrl[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    u_ff = data.qfrc_inverse[qvel_addrs].copy()

    print(f"  Config: {config!r}, q_desired: {q_desired}")
    print(f"  Feedforward torques: {u_ff}")

    A, B = compute_linearization_from_eom(q_desired, qdot_desired, u_ff, param_values)

    print(f"\nA matrix shape: {A.shape}")
    print(f"B matrix shape: {B.shape}")

    print("\n" + "=" * 70)
    print("PART 3: Building Augmented LQI Model (Joint + Force-Integral)")
    print("=" * 70)

    J_eq = ee_jacobian(model, data, ee_site_id, qvel_addrs)
    x_ee_eq = data.site_xpos[ee_site_id].copy()
    x_wall_eq = WALL_REF_POS[config].copy()

    n_hat = np.array([1.0, 0.0, 0.0])
    if x_ee_eq[0] > x_wall_eq[0]:
        n_hat = -n_hat
    n_hat /= np.linalg.norm(n_hat)

    print(f"EE equilibrium position: {x_ee_eq}")
    print(f"Wall reference position ({config}): {x_wall_eq}")
    print(f"Wall normal n_hat: {n_hat}")
    print(f"Desired normal force: {FORCE_ON_WALL_N} N")

    C_q = CONTACT_STIFFNESS_N_PER_M * (n_hat @ J_eq)
    C_F = np.zeros(n_states_joint)
    C_F[:3] = C_q

    print(f"Approximate C_F (dF_n/dx) row: {C_F}")

    A_aug = np.zeros((n_states_aug, n_states_aug))
    A_aug[:n_states_joint, :n_states_joint] = A
    A_aug[n_states_aug - 1, :n_states_joint] = C_F

    B_aug = np.zeros((n_states_aug, n_controls))
    B_aug[:n_states_joint, :] = B

    Q_joint = np.eye(n_states_joint)
    Q_joint[:3, :3] *= Q_POSITION_WEIGHT
    Q_joint[3:, 3:] *= Q_VELOCITY_WEIGHT

    Q_aug = np.zeros((n_states_aug, n_states_aug))
    Q_aug[:n_states_joint, :n_states_joint] = Q_joint
    Q_aug[n_states_aug - 1, n_states_aug - 1] = Q_FORCE_INTEGRAL_WEIGHT

    R = np.eye(n_controls) * R_CONTROL_WEIGHT

    print("Solving Riccati equation (augmented joint + force-integral states)...")
    P = scipy.linalg.solve_continuous_are(A_aug, B_aug, Q_aug, R)
    K_aug = np.linalg.solve(R, B_aug.T @ P)

    print("Augmented gain matrix K_aug (including force-integral state) computed.")
    print(f"K_aug shape: {K_aug.shape}")
    print(f"K_aug:\n{K_aug}")

    print("\n" + "=" * 70)
    print("PART 4: Starting Simulation with Force-Integral-Augmented Feedback")
    print("=" * 70)
    print(
        f"Disturbance: |F|={DISTURBANCE_MAGNITUDE_N} N, "
        f"dir={dist_unit}, "
        f"t={DISTURBANCE_START_TIME}s->{dist_end_time}s"
    )
    print("Close the viewer window to exit.")

    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] = q_desired[i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    z_f = 0.0
    step = 0

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = VIEWER_CAMERA_LOOKAT
        viewer.cam.distance = VIEWER_CAMERA_DISTANCE
        viewer.cam.azimuth = VIEWER_CAMERA_AZIMUTH
        viewer.cam.elevation = VIEWER_CAMERA_ELEVATION

        while viewer.is_running():
            x_joint = np.zeros(n_states_joint)
            x_joint[:3] = [data.qpos[addr] for addr in qpos_addrs]
            x_joint[3:] = [data.qvel[addr] for addr in qvel_addrs]

            F_site = data.sensordata[ee_force_adr : ee_force_adr + ee_force_dim].copy()
            R_site = data.site_xmat[ee_site_id].reshape(3, 3)
            F_world = R_site @ F_site
            F_n_meas = float(np.dot(F_world, n_hat))

            z_f += (F_n_meas - FORCE_ON_WALL_N) * model.opt.timestep

            x_err_aug = np.zeros(n_states_aug)
            x_err_aug[:n_states_joint] = x_joint - x_desired_joint
            x_err_aug[-1] = z_f
            u = u_ff - K_aug @ x_err_aug

            for i, act_id in enumerate(actuator_ids):
                data.ctrl[act_id] = u[i]

            data.qfrc_applied[:] = 0.0
            dist_active = DISTURBANCE_START_TIME <= data.time < dist_end_time
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

            if step % LOG_EVERY_N_STEPS == 0:
                pos_err = x_joint[:3] - x_desired_joint[:3]
                perturb_magnitude = np.linalg.norm(data.xfrc_applied[ee_body_id, :3])
                print(
                    f"t={data.time:6.3f}  "
                    f"F_n_meas={F_n_meas:7.3f} N  "
                    f"F_err={F_n_meas - FORCE_ON_WALL_N:7.3f} N  "
                    f"z_f={z_f:7.4f}  "
                    f"disturbance={'YES' if dist_active else 'NO':>3}  "
                    f"mouse_force={perturb_magnitude:.2f} N  "
                    f"pos_err_norm={np.linalg.norm(pos_err):.6f}"
                )
            step += 1
            viewer.sync()

    print("\nSimulation complete.")


if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
