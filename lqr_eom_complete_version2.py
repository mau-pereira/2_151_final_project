"""
Complete LQR controller using analytical EOM for 3-link arm.

This script:
1. Derives equations of motion using Lagrangian mechanics
2. Computes A and B matrices analytically using Jacobians
3. Computes LQR gains
4. Stabilizes the robot around a desired state

Usage:
    Set the desired state by modifying q_desired below (around line 270).
    The desired velocities are set to zero for static equilibrium.
    
    Run: python lqr_eom_complete.py
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
# CONFIGURATION: Desired state vector: x = [q1, q2, q3, q1dot, q2dot, q3dot]
# ============================================================================
n_states = 6
n_controls = 3
q_desired = np.array([2.43748575e+00 , 1.27675648e-14 ,-2.70200529e-14])  # Desired joint positions
qdot_desired = np.array([0.0, 0.0, 0.0])  # Desired velocities (zero for static equilibrium)
x_desired = np.zeros(n_states)
x_desired[:3] = q_desired
x_desired[3:] = qdot_desired
# ============================================================================
# PART 3: SETUP AND COMPUTATION
# ============================================================================
print("\n" + "="*70)
print("PART 2: Computing Linearization Matrices")
print("="*70)

# Compute feedforward torques using MuJoCo
print("\nComputing feedforward torques (gravity compensation)...")
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

joint_names = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint"]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
qpos_addrs = [model.joint(joint_id).qposadr[0] for joint_id in joint_ids]
qvel_addrs = [model.joint(joint_id).dofadr[0] for joint_id in joint_ids]
actuator_names = ["torq_j2", "torq_j3", "torq_j4"]
actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]

# Set equilibrium state
for i, addr in enumerate(qpos_addrs):
    data.qpos[addr] = q_desired[i]
for i, addr in enumerate(qvel_addrs):
    data.qvel[addr] = qdot_desired[i]

# Compute feedforward torque
data.ctrl[:] = 0.0
data.qacc[:] = 0.0
mujoco.mj_inverse(model, data)
u_ff = data.qfrc_inverse[qvel_addrs].copy()

print(f"  Feedforward torques: {u_ff}")

# Compute A and B matrices directly from the EOM
A, B = compute_linearization_from_eom(q_desired, qdot_desired, u_ff, param_values)

print(f"\nA matrix shape: {A.shape}")
print(f"B matrix shape: {B.shape}")

# ============================================================================
# PART 4: COMPUTE LQR GAINS
# ============================================================================
print("\n" + "="*70)
print("PART 3: Computing LQR Gains")
print("="*70)



# Q and R matrices
Q = np.eye(n_states)
Q[:3, :3] *= 100.0  # Position error weight
Q[3:, 3:] *= 10.0   # Velocity error weight

R = np.eye(n_controls) * 0.1  # Control effort weight

# Solve continuous-time Riccati equation
print("Solving Riccati equation...")
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# Compute LQR gain: K = R^-1 B^T P
K = np.linalg.solve(R, B.T @ P)

print("LQR gain matrix computed.")
print(f"K shape: {K.shape}")
print(f"K:\n{K}")

# ============================================================================
# PART 5: SIMULATION
# ============================================================================
print("\n" + "="*70)
print("PART 4: Starting Simulation")
print("="*70)
print("Close the viewer window to exit.")



# Initialize to desired joint positions
for i, addr in enumerate(qpos_addrs):
    data.qpos[addr] = q_desired[i]
for i, addr in enumerate(qvel_addrs):
    data.qvel[addr] = 0.0

mujoco.mj_forward(model, data)

# Set up viewer
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    viewer.cam.lookat[:] = [0.35, 0.1, 0.3]
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -15
    
    while viewer.is_running():
        # Get current state
        x = np.zeros(n_states)
        x[:3] = np.array([data.qpos[addr] for addr in qpos_addrs])
        x[3:] = np.array([data.qvel[addr] for addr in qvel_addrs])
        
        # Compute control: u = u_ff - K * (x - x_desired)
        error = x - x_desired
        u = u_ff - K @ error
        
        # Apply control
        for i, act_id in enumerate(actuator_ids):
            data.ctrl[act_id] = u[i]
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Sync viewer
        viewer.sync()

print("\nSimulation complete.")

