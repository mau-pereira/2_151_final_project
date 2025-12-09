#!/usr/bin/env python3
"""Solve IK to position end effector near target_1 sphere."""

import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Target position (from target_1 sphere)
target_pos = np.array([0.7, 0.135, 0.05])

# Get end effector site ID
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")

# Get joint IDs
joint_names = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint"]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
qpos_addrs = [model.joint(joint_id).qposadr[0] for joint_id in joint_ids]

# Set initial joint positions
# initial_qpos = np.array([-0.6, 1.6, -2.3])
initial_qpos = np.array([2.5, 1.67, -0.942])
for i, addr in enumerate(qpos_addrs):
    data.qpos[addr] = initial_qpos[i]

# IK solver parameters
damping = 0.01
max_iterations = 100
tolerance = 0.01  # 5 cm

# Iterative IK solving
for iteration in range(max_iterations):
    mujoco.mj_forward(model, data)
    
    # Current end effector position
    current_pos = data.site(site_id).xpos.copy()
    
    # Error
    error = target_pos - current_pos
    error_norm = np.linalg.norm(error)
    
    if error_norm < tolerance:
        break
    
    # Compute Jacobian (3x3 for position only)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    
    # Extract Jacobian for our joints
    jac = jacp[:, qpos_addrs]
    
    # Damped least squares: delta_q = J^T (J J^T + lambda I)^-1 delta_x
    jjt = jac @ jac.T
    jjt_reg = jjt + damping * np.eye(3)
    delta_q = jac.T @ np.linalg.solve(jjt_reg, error)
    
    # Update joint positions
    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] += delta_q[i]
        
        # Apply joint limits
        joint = model.joint(joint_ids[i])
        if joint.range[0] < joint.range[1]:
            data.qpos[addr] = np.clip(data.qpos[addr], joint.range[0], joint.range[1])

# Final forward kinematics
mujoco.mj_forward(model, data)

# Print joint positions
print("Joint positions (qpos):")
print(data.qpos)
for name, addr in zip(joint_names, qpos_addrs):
    print(f"  {name}: {data.qpos[addr]:.6f}")

