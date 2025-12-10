"""
Force-augmented LQR controller using analytical EOM for 3-link arm.

It:
1. Derives equations of motion using Lagrangian mechanics (3-DOF planar UR5e)
2. Computes A and B matrices analytically using Jacobians
3. Computes joint-space LQR gains for the 6 joint states
4. Augments the feedback with a 7th state: normal force error at the
   end-effector, measured from MuJoCo
5. Stabilizes the robot around a desired joint state while also regulating
   the normal force to a target value.
6. Can produce a disturbance by editing its parameters below.

Usage:
    - Set the desired joint state by modifying q_desired and WALL_REF_SITE_NAME 
    (low, medium, or high) below.
    - Set the desired normal force F_N_DES.
    - Set the disturbance parameters below.
    - Run

"""
import sympy as sp
import numpy as np
import scipy.linalg
import mujoco
import mujoco.viewer

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



# ============================================================================
# CONFIGURATION: Desired state vector (joint + force-integral)
#   x_joint = [q1, q2, q3, q1dot, q2dot, q3dot]      (for EOM / base LQR)
#   x_aug   = [q1, q2, q3, q1dot, q2dot, q3dot, z_f] (for LQI feedback)
# ============================================================================

# 6 joint states used for analytic linearization and Riccati solve
n_states_joint = 6
n_controls = 3

# Desired joint configuration
q_desired = np.array([1.97253242, 1.26623755, 0.01733902])
qdot_desired = np.array([0.0, 0.0, 0.0])

x_desired_joint = np.zeros(n_states_joint)
x_desired_joint[:3] = q_desired
x_desired_joint[3:] = qdot_desired

# Augmented state (adds 7th entry for integral of normal force error z_f)
n_states_aug = 7

# Desired normal force (env-on-robot, positive when pushing into wall)
F_N_DES = 10.0  # [N]

# ============================================================================
# PART 3: SETUP AND COMPUTATION
# ============================================================================
print("\n" + "="*70)
print("PART 2: Computing Linearization Matrices (Joint States Only)")
print("="*70)

# Compute feedforward torques using MuJoCo at the joint equilibrium
print("\nComputing feedforward torques (gravity compensation)...")
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

joint_names = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint"]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
qpos_addrs = [model.joint(joint_id).qposadr[0] for joint_id in joint_ids]
qvel_addrs = [model.joint(joint_id).dofadr[0] for joint_id in joint_ids]
actuator_names = ["torq_j2", "torq_j3", "torq_j4"]
actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]

# Force sensor and wall / EE sites (for measuring F_n in simulation)
EE_SITE_NAME = "end_effector_site"
WALL_REF_SITE_NAME = "medium"
EE_FORCE_SENSOR_NAME = "ee_force"

ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
wall_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, WALL_REF_SITE_NAME)
ee_force_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, EE_FORCE_SENSOR_NAME)
ee_force_adr = model.sensor_adr[ee_force_sid]
ee_force_dim = model.sensor_dim[ee_force_sid]  # expected 3 for <force>

# Get body ID for end-effector (wrist_2_link contains the end_effector_site)
EE_BODY_NAME = "wrist_2_link"
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

# Set equilibrium state (joint space only)
for i, addr in enumerate(qpos_addrs):
    data.qpos[addr] = q_desired[i]
for i, addr in enumerate(qvel_addrs):
    data.qvel[addr] = qdot_desired[i]

mujoco.mj_forward(model, data)

# Compute feedforward torque at the equilibrium
data.ctrl[:] = 0.0
data.qacc[:] = 0.0
mujoco.mj_inverse(model, data)
u_ff = data.qfrc_inverse[qvel_addrs].copy()

print(f"  Feedforward torques: {u_ff}")

# ----------------------------------------------------------------------------
# Compute A and B matrices directly from the EOM (6x6, 6x3)
# ----------------------------------------------------------------------------
A, B = compute_linearization_from_eom(q_desired, qdot_desired, u_ff, param_values)

print(f"\nA matrix shape: {A.shape}")
print(f"B matrix shape: {B.shape}")

# ----------------------------------------------------------------------------
# Build augmented (LQI) model with integral of normal force error z_f:
#
#   x_joint_dot = A x_joint + B u
#   z_f_dot     = (F_n - F_N_DES) ≈ C_F x_joint   (around equilibrium)
#
# where C_F is a 1x6 row vector approximating how joint-state deviations
# change normal force, using a simple contact-stiffness model:
#
#   F_n ≈ k_c_eff * (n_hat^T J_eq) (q - q_eq)
#
# We ignore dependence on velocities in C_F (quasi-static contact).
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print("PART 3: Building Augmented LQI Model (Joint + Force-Integral)")
print("="*70)

# Compute EE Jacobian and wall normal at the equilibrium
J_eq = ee_jacobian(model, data, ee_site_id, qvel_addrs)  # 3x3
x_ee_eq = data.site_xpos[ee_site_id].copy()
x_wall_eq = data.site_xpos[wall_site_id].copy()

# Wall normal n_hat
n_hat = np.array([1.0, 0.0, 0.0])
if x_ee_eq[0] > x_wall_eq[0]:
    n_hat = -n_hat
n_hat = n_hat / np.linalg.norm(n_hat)

print(f"EE equilibrium position: {x_ee_eq}")
print(f"Wall reference position (site 'medium'): {x_wall_eq}")
print(f"Wall normal n_hat: {n_hat}")
print(f"Desired normal force F_N_DES: {F_N_DES} N")

# Effective contact stiffness in the normal direction
k_c_eff = 1500.0  # [N/m]

# C_q: row mapping joint angle deviations to normal force deviation
# F_n ≈ k_c_eff * (n_hat^T J_eq) (q - q_eq)
C_q = k_c_eff * (n_hat @ J_eq)  # shape (3,)

# Build C_F (1x6) acting on [q1, q2, q3, q1dot, q2dot, q3dot]
C_F = np.zeros(n_states_joint)
C_F[:3] = C_q

print(f"Approximate C_F (dF_n/dx) row: {C_F}")

# Augmented A and B for [x_joint; z_f]
A_aug = np.zeros((n_states_aug, n_states_aug))
A_aug[:n_states_joint, :n_states_joint] = A
A_aug[n_states_joint, :n_states_joint] = C_F  # z_f_dot = C_F * x_joint

B_aug = np.zeros((n_states_aug, n_controls))
B_aug[:n_states_joint, :] = B  # z_f has no direct input

# Q and R matrices for the augmented (joint + integral-of-force-error) state
Q_joint = np.eye(n_states_joint)
Q_joint[:3, :3] *= 500.0  # Position error weight
Q_joint[3:, 3:] *= 50.0   # Velocity error weight

Q_aug = np.zeros((n_states_aug, n_states_aug))
Q_aug[:n_states_joint, :n_states_joint] = Q_joint

# Weight on integral of force error z_f
Q_aug[n_states_joint, n_states_joint] = 50.0 

R = np.eye(n_controls) * 0.5 # Control effort weight

print("Solving Riccati equation (augmented joint + force-integral states)...")
P = scipy.linalg.solve_continuous_are(A_aug, B_aug, Q_aug, R)

# Augmented LQI gain: K_aug (3x7)
K_aug = np.linalg.solve(R, B_aug.T @ P)

print("Augmented gain matrix K_aug (including force-integral state) computed.")
print(f"K_aug shape: {K_aug.shape}")
print(f"K_aug:\n{K_aug}")

# ============================================================================
# PART 5: SIMULATION
# ============================================================================
print("\n" + "="*70)
print("PART 4: Starting Simulation with Force-Integral-Augmented Feedback")
print("="*70)
print("Close the viewer window to exit.")


# Initialize to desired joint positions
for i, addr in enumerate(qpos_addrs):
    data.qpos[addr] = q_desired[i]
for i, addr in enumerate(qvel_addrs):
    data.qvel[addr] = 0.0

mujoco.mj_forward(model, data)

# Integral of force error state z_f
z_f = 0.0

# Baseline force measurement (to subtract internal forces)
baseline_force_magnitude = None
baseline_settled = False
baseline_settle_time = 2.0  # Wait 2 seconds to establish baseline

# ============================================================================
# DISTURBANCE PARAMETERS
# ============================================================================
DISTURBANCE_START_TIME = 5.0  # [s]
DISTURBANCE_DURATION = 1.0     # [s]
DISTURBANCE_FORCE = 10.0        # [N]
DISTURBANCE_DIRECTION = np.array([1.0, 0.0, 1.0]) 
# ============================================================================

# Set up viewer
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    viewer.cam.lookat[:] = [0.35, 0.1, 0.3]
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -15

    step = 0
    while viewer.is_running():
        # Get current joint state
        x_joint = np.zeros(n_states_joint)
        x_joint[:3] = np.array([data.qpos[addr] for addr in qpos_addrs])
        x_joint[3:] = np.array([data.qvel[addr] for addr in qvel_addrs])

        # Measure contact force at end-effector (env-on-robot)
        F_site = data.sensordata[ee_force_adr:ee_force_adr + ee_force_dim].copy()
        R_site = data.site_xmat[ee_site_id].reshape(3, 3)
        F_world = R_site @ F_site
        F_n_meas = float(np.dot(F_world, n_hat))

        # Update integral of force error: z_f_dot = F_n - F_N_DES
        z_f += (F_n_meas - F_N_DES) * model.opt.timestep

        # Build 7D augmented error state: [x_joint - x_desired_joint, z_f]
        x_err_aug = np.zeros(n_states_aug)
        x_err_aug[:n_states_joint] = x_joint - x_desired_joint
        x_err_aug[6] = z_f

        # Compute control: u = u_ff - K_aug * x_err_aug
        u = u_ff - K_aug @ x_err_aug

        # Apply control
        for i, act_id in enumerate(actuator_ids):
            data.ctrl[act_id] = u[i]

        # ====================================================================
        # DISTURBANCE APPLICATION
        # ====================================================================
        # Clear qfrc_applied before applying new forces (prevent accumulation)
        data.qfrc_applied[:] = 0.0
        
        # Apply disturbance force at end-effector if in disturbance window
        if DISTURBANCE_START_TIME <= data.time < DISTURBANCE_START_TIME + DISTURBANCE_DURATION:
            # Get current end-effector position in world frame
            # This is the position of the "end_effector_site" (0.139m along z-axis of wrist_2_link)
            ee_pos = data.site_xpos[ee_site_id].copy()
            # Force and torque in world frame (upward force, no torque)
            force_world = (DISTURBANCE_FORCE * DISTURBANCE_DIRECTION).reshape(3, 1)
            torque_world = np.zeros((3, 1))
            # qfrc_target: where to store the resulting generalized forces
            qfrc_target = np.zeros((model.nv, 1))
            # Apply force at end-effector site position (world coordinates)
            mujoco.mj_applyFT(model, data, force_world, torque_world, ee_pos.reshape(3, 1), ee_body_id, qfrc_target)
            # Set the generalized forces (not add, to prevent accumulation)
            data.qfrc_applied[:] = qfrc_target.flatten()
        # ====================================================================

        # Step simulation
        mujoco.mj_step(model, data)

        # Periodic logging
        if step % 50 == 0:
            pos_err = x_joint[:3] - x_desired_joint[:3]
            # === DISTURBANCE STATUS ===
            disturbance_active = "YES" if (DISTURBANCE_START_TIME <= data.time < DISTURBANCE_START_TIME + DISTURBANCE_DURATION) else "NO"
            # ==========================
            
            # Also check for mouse drag forces
            perturb_force = data.xfrc_applied[ee_body_id, :3]
            perturb_magnitude = np.linalg.norm(perturb_force)
            
            print(
                f"t={data.time:6.3f}  "
                f"F_n_meas={F_n_meas:7.3f} N  "
                f"F_err={F_n_meas - F_N_DES:7.3f} N  "
                f"z_f={z_f:7.4f}  "
                f"disturbance={disturbance_active}  "  # === DISTURBANCE ===
                f"mouse_force={perturb_magnitude:.2f} N  "
                f"pos_err_norm={np.linalg.norm(pos_err):.6f}"
            )
        step += 1

        # Sync viewer
        viewer.sync()

print("\nSimulation complete.")
