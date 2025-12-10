"""
Analyze stability differences between low, medium, and high configurations.
Computes linearized dynamics, Jacobians, manipulability, and other factors.
"""
import numpy as np
import sympy as sp
import scipy.linalg
import mujoco

# Import the linearization function from the main script
import sys
sys.path.insert(0, '/Users/Mauricio/code/Project/2_151_final_project')

# We'll need to import or duplicate the linearization code
# For now, let's read the main file structure

# Physical parameters (same as in main script)
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

# Height configurations
HEIGHT_CONFIGS = [
    {
        "name": "low",
        "q_desired": np.array([2.55461019, 1.60952476, -0.92886277]),
    },
    {
        "name": "medium",
        "q_desired": np.array([1.97253242, 1.26623755, 0.01733902]),
    },
    {
        "name": "high",
        "q_desired": np.array([2.18718072, 0.36788401, 0.076558322]),
    },
]

def ee_jacobian(model, data, site_id, qvel_addrs):
    """3x3 translational Jacobian of the EE site."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J_full = jacp
    J = J_full[:, qvel_addrs]
    return J

def compute_gravity_torques(model, data, q_desired, qpos_addrs, qvel_addrs):
    """Compute gravity compensation torques at equilibrium."""
    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] = q_desired[i]
    for addr in qvel_addrs:
        data.qvel[addr] = 0.0
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    return data.qfrc_inverse[qvel_addrs].copy()

def analyze_configuration(q_desired, name):
    """Analyze a single configuration."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {name.upper()}")
    print(f"{'='*70}")
    print(f"Joint angles: q = {q_desired}")
    print(f"  q1 (shoulder): {q_desired[0]:.4f} rad ({np.degrees(q_desired[0]):.2f}°)")
    print(f"  q2 (elbow):    {q_desired[1]:.4f} rad ({np.degrees(q_desired[1]):.2f}°)")
    print(f"  q3 (wrist):    {q_desired[2]:.4f} rad ({np.degrees(q_desired[2]):.2f}°)")
    
    # Load model
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    joint_names = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    qpos_addrs = [model.joint(joint_id).qposadr[0] for joint_id in joint_ids]
    qvel_addrs = [model.joint(joint_id).dofadr[0] for joint_id in joint_ids]
    
    EE_SITE_NAME = "end_effector_site"
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    
    # Set configuration
    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] = q_desired[i]
    for addr in qvel_addrs:
        data.qvel[addr] = 0.0
    mujoco.mj_forward(model, data)
    
    # 1. End-effector position
    x_ee = data.site_xpos[ee_site_id].copy()
    print(f"\n1. END-EFFECTOR POSITION:")
    print(f"   Position: [{x_ee[0]:.4f}, {x_ee[1]:.4f}, {x_ee[2]:.4f}]")
    print(f"   Height (z): {x_ee[2]:.4f} m")
    
    # 2. Gravity compensation torques
    u_ff = compute_gravity_torques(model, data, q_desired, qpos_addrs, qvel_addrs)
    print(f"\n2. GRAVITY COMPENSATION TORQUES:")
    print(f"   u_ff = [{u_ff[0]:.4f}, {u_ff[1]:.4f}, {u_ff[2]:.4f}] N⋅m")
    print(f"   ||u_ff|| = {np.linalg.norm(u_ff):.4f} N⋅m")
    
    # 3. End-effector Jacobian
    J = ee_jacobian(model, data, ee_site_id, qvel_addrs)
    print(f"\n3. END-EFFECTOR JACOBIAN (3x3):")
    print(f"   J = \n{J}")
    
    # Manipulability (measure of how well-conditioned the Jacobian is)
    manipulability = np.sqrt(np.linalg.det(J @ J.T))
    print(f"\n4. MANIPULABILITY:")
    print(f"   μ = √det(JJ^T) = {manipulability:.6f}")
    print(f"   (Higher = better dexterity, more isotropic workspace)")
    
    # Condition number of Jacobian
    _, s, _ = np.linalg.svd(J)
    cond_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
    print(f"   Condition number = {cond_number:.4f}")
    print(f"   (Lower = better, 1.0 = isotropic)")
    
    # 4. Contact normal direction and C_F
    # Assume wall is at x = 0.75, normal is [-1, 0, 0]
    n_hat = np.array([-1.0, 0.0, 0.0])  # Wall normal (points from wall to robot)
    C_q = 1500.0 * (n_hat @ J)  # Force sensitivity row
    print(f"\n5. FORCE CONTROL SENSITIVITY:")
    print(f"   C_F[:3] = {C_q}")
    print(f"   ||C_F|| = {np.linalg.norm(C_q):.4f}")
    print(f"   (How joint errors map to force errors)")
    
    # 5. Analyze link angles and geometry
    print(f"\n6. ARM GEOMETRY ANALYSIS:")
    # Approximate link directions
    # Link 1: from base, angle q1
    # Link 2: angle q1 + q2  
    # Link 3: angle q1 + q2 + q3
    
    link1_angle = q_desired[0]
    link2_angle = q_desired[0] + q_desired[1]
    link3_angle = q_desired[0] + q_desired[1] + q_desired[2]
    
    print(f"   Link 1 angle: {np.degrees(link1_angle):.2f}°")
    print(f"   Link 2 angle: {np.degrees(link2_angle):.2f}°")
    print(f"   Link 3 angle: {np.degrees(link3_angle):.2f}°")
    
    # Check for near-singular configurations
    # Elbow singularity: q2 ≈ 0 or q2 ≈ π
    elbow_angle = q_desired[1]
    print(f"\n   ELBOW CONFIGURATION:")
    if abs(elbow_angle) < 0.2:
        print(f"     ⚠️  Near extended singularity (q2 ≈ 0)")
    elif abs(elbow_angle - np.pi) < 0.2 or abs(elbow_angle + np.pi) < 0.2:
        print(f"     ⚠️  Near folded singularity (q2 ≈ ±π)")
    else:
        print(f"     ✓  Well away from singularities")
    
    # Check wrist configuration
    wrist_angle = q_desired[2]
    print(f"   WRIST CONFIGURATION:")
    if abs(wrist_angle) < 0.1:
        print(f"     → Wrist aligned with arm (q3 ≈ 0)")
    elif abs(wrist_angle) > 1.5:
        print(f"     ⚠️  Large wrist bend")
    else:
        print(f"     → Moderate wrist angle")
    
    # 6. Potential energy (approximate)
    # V = m1*g*y1 + m2*g*y2 + m3*g*y3
    # Approximate COM positions
    L1, L2, L3com = 0.425, 0.392, 0.053595
    y_base = 0.163
    
    y1_com = y_base + (L1/2) * np.sin(q_desired[0])
    y2_com = y_base + L1 * np.sin(q_desired[0]) + (L2/2) * np.sin(q_desired[0] + q_desired[1])
    y3_com = y_base + L1 * np.sin(q_desired[0]) + L2 * np.sin(q_desired[0] + q_desired[1]) + L3com * np.sin(link3_angle)
    
    m1, m2, m3 = param_values['m1'], param_values['m2'], param_values['m3']
    g = param_values['g']
    V_total = m1*g*y1_com + m2*g*y2_com + m3*g*y3_com
    
    print(f"\n7. POTENTIAL ENERGY:")
    print(f"   COM heights: y1={y1_com:.4f}, y2={y2_com:.4f}, y3={y3_com:.4f}")
    print(f"   Total V = {V_total:.4f} J")
    
    return {
        'name': name,
        'q': q_desired,
        'x_ee': x_ee,
        'u_ff': u_ff,
        'J': J,
        'manipulability': manipulability,
        'cond_number': cond_number,
        'C_q': C_q,
        'V': V_total,
    }

# Main analysis
print("STABILITY ANALYSIS BY HEIGHT CONFIGURATION")
print("="*70)

results = []
for cfg in HEIGHT_CONFIGS:
    result = analyze_configuration(cfg['q_desired'], cfg['name'])
    results.append(result)

# Comparative analysis
print(f"\n\n{'='*70}")
print("COMPARATIVE SUMMARY")
print(f"{'='*70}")

print("\n1. MANIPULABILITY (higher = better):")
for r in results:
    print(f"   {r['name']:8s}: {r['manipulability']:.6f}")

print("\n2. CONDITION NUMBER (lower = better):")
for r in results:
    print(f"   {r['name']:8s}: {r['cond_number']:.4f}")

print("\n3. GRAVITY TORQUE MAGNITUDE:")
for r in results:
    print(f"   {r['name']:8s}: ||u_ff|| = {np.linalg.norm(r['u_ff']):.4f} N⋅m")

print("\n4. FORCE SENSITIVITY MAGNITUDE:")
for r in results:
    print(f"   {r['name']:8s}: ||C_F|| = {np.linalg.norm(r['C_q']):.4f}")

print("\n5. END-EFFECTOR HEIGHT:")
for r in results:
    print(f"   {r['name']:8s}: z = {r['x_ee'][2]:.4f} m")

print("\n6. POTENTIAL ENERGY:")
for r in results:
    print(f"   {r['name']:8s}: V = {r['V']:.4f} J")

print("\n" + "="*70)
print("STABILITY HYPOTHESES:")
print("="*70)
print("""
Based on the analysis, potential reasons for stability differences:

1. MANIPULABILITY: Higher manipulability means better dexterity and more
   uniform force/torque transmission from joints to end-effector.

2. CONDITION NUMBER: Lower condition number means the Jacobian is better
   conditioned, leading to more stable force control.

3. GRAVITY TORQUES: Larger gravity compensation torques mean the arm is
   fighting more against gravity, potentially reducing stability margin.

4. FORCE SENSITIVITY (C_F): Different C_F magnitudes affect how joint
   position errors translate to force errors, affecting closed-loop dynamics.

5. GEOMETRY: Near-singular configurations (extended/folded elbow) reduce
   controllability and stability.
""")

