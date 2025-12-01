"""
Detailed analysis of walking motion - check if IK is actually producing leg movement
"""
import mujoco
import numpy as np
import os
import re
from pathlib import Path

def load_model_with_assets():
    xml_path = Path(__file__).parent / "legged_robot_ik.xml"
    
    with open(xml_path, "r") as f:
        xml_content = f.read()
    
    asset_dir = Path(__file__).parent / "Legged_robot" / "meshes"
    
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

def compute_ik_solution(model, data, com_target, foot1_target, foot2_target, max_iterations=20, tolerance=0.005):
    try:
        com_site_id = model.site(name='com_site').id
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id
    except Exception as e:
        raise ValueError(f"Could not find required sites: {e}")
    
    qpos = data.qpos.copy()
    alpha = 0.05
    epsilon = 1e-6
    
    for iteration in range(max_iterations):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        com_pos = data.site_xpos[com_site_id].copy()
        foot1_pos = data.site_xpos[foot1_site_id].copy()
        foot2_pos = data.site_xpos[foot2_site_id].copy()
        
        com_error = com_target - com_pos
        foot1_error = foot1_target - foot1_pos
        foot2_error = foot2_target - foot2_pos
        
        total_error = np.linalg.norm(com_error) + np.linalg.norm(foot1_error) + np.linalg.norm(foot2_error)
        
        if total_error < tolerance:
            return qpos, True
        
        n_joints = model.nq
        jacobian = np.zeros((9, n_joints))
        
        for j in range(n_joints):
            qpos_plus = qpos.copy()
            qpos_plus[j] += epsilon
            
            data.qpos[:] = qpos_plus
            mujoco.mj_forward(model, data)
            
            pos_plus_com = data.site_xpos[com_site_id].copy()
            pos_plus_foot1 = data.site_xpos[foot1_site_id].copy()
            pos_plus_foot2 = data.site_xpos[foot2_site_id].copy()
            
            jacobian[0:3, j] = (pos_plus_com - com_pos) / epsilon
            jacobian[3:6, j] = (pos_plus_foot1 - foot1_pos) / epsilon
            jacobian[6:9, j] = (pos_plus_foot2 - foot2_pos) / epsilon
        
        try:
            jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
        except:
            return qpos, False
        
        errors = np.concatenate([com_error, foot1_error, foot2_error])
        dq = alpha * jacobian_pinv @ errors
        
        if np.any(~np.isfinite(dq)):
            return qpos, False
        
        dq = np.clip(dq, -0.1, 0.1)
        qpos = qpos + dq
        
        for i in range(6, model.nq):
            jnt_idx = i - 6
            if jnt_idx < model.jnt_range.shape[0]:
                qpos[i] = np.clip(qpos[i], model.jnt_range[jnt_idx, 0], model.jnt_range[jnt_idx, 1])
    
    return qpos, False

# Load trajectories
com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

model = load_model_with_assets()
data = mujoco.MjData(model)

print("=== WALKING ANALYSIS ===\n")

# Compute IK for selected frames
frames = [0, 50, 100, 150, 200, 250, 300, 350, 399]
qpos_solutions = []

print("Computing IK solutions for key frames...\n")

for frame_idx in frames:
    com_target = com_traj[frame_idx]
    foot1_target = foot1_traj[frame_idx]
    foot2_target = foot2_traj[frame_idx]
    
    qpos_solution, _ = compute_ik_solution(model, data, com_target, foot1_target, foot2_target)
    qpos_solutions.append(qpos_solution)
    
    # Apply and check result
    data.qpos[:] = qpos_solution
    mujoco.mj_forward(model, data)
    
    com_site_id = model.site(name='com_site').id
    foot1_site_id = model.site(name='foot1_site').id
    foot2_site_id = model.site(name='foot2_site').id
    
    actual_com = data.site_xpos[com_site_id]
    actual_foot1 = data.site_xpos[foot1_site_id]
    actual_foot2 = data.site_xpos[foot2_site_id]
    
    print(f"Frame {frame_idx}:")
    print(f"  Targets: COM={com_target}, Foot1={foot1_target}, Foot2={foot2_target}")
    print(f"  Actual:  COM={actual_com}, Foot1={actual_foot1}, Foot2={actual_foot2}")
    print(f"  qpos[6:12]: {qpos_solution[6:12]}")
    print()

# Check leg joint movement
print("=== LEG JOINT MOVEMENTS ===\n")
print("Frame | Hip1   | Knee1  | Foot1  | Hip2   | Knee2  | Foot2")
print("------|--------|--------|--------|--------|--------|--------")
for i, frame_idx in enumerate(frames):
    q = qpos_solutions[i]
    print(f"{frame_idx:5d} | {q[6]:+.3f} | {q[7]:+.3f} | {q[8]:+.3f} | {q[9]:+.3f} | {q[10]:+.3f} | {q[11]:+.3f}")

# Check COM trajectory
print("\n=== COM TRAJECTORY ANALYSIS ===\n")
print("Frame | X Position | Y Position | Z Position | Forward Motion")
print("------|------------|------------|------------|----------------")
for i, frame_idx in enumerate(frames):
    x, y, z = com_traj[frame_idx]
    if i == 0:
        dx = 0
    else:
        dx = com_traj[frame_idx][0] - com_traj[frames[i-1]][0]
    print(f"{frame_idx:5d} | {x:+.4f}     | {y:+.4f}     | {z:+.4f}     | {dx:+.6f}")

# Check if feet are actually moving
print("\n=== FOOT MOVEMENT ANALYSIS ===\n")
print("Foot 1 trajectory:")
print(f"  X: {foot1_traj[:, 0].min():.4f} to {foot1_traj[:, 0].max():.4f} (range: {foot1_traj[:, 0].max() - foot1_traj[:, 0].min():.4f})")
print(f"  Y: {foot1_traj[:, 1].min():.4f} to {foot1_traj[:, 1].max():.4f} (range: {foot1_traj[:, 1].max() - foot1_traj[:, 1].min():.4f})")
print(f"  Z: {foot1_traj[:, 2].min():.4f} to {foot1_traj[:, 2].max():.4f} (range: {foot1_traj[:, 2].max() - foot1_traj[:, 2].min():.4f})")

print("\nFoot 2 trajectory:")
print(f"  X: {foot2_traj[:, 0].min():.4f} to {foot2_traj[:, 0].max():.4f} (range: {foot2_traj[:, 0].max() - foot2_traj[:, 0].min():.4f})")
print(f"  Y: {foot2_traj[:, 1].min():.4f} to {foot2_traj[:, 1].max():.4f} (range: {foot2_traj[:, 1].max() - foot2_traj[:, 1].min():.4f})")
print(f"  Z: {foot2_traj[:, 2].min():.4f} to {foot2_traj[:, 2].max():.4f} (range: {foot2_traj[:, 2].max() - foot2_traj[:, 2].min():.4f})")

# Check if legs are actually changing
print("\n=== DIAGNOSIS ===\n")

# Check leg joint changes
knee1_changes = [abs(qpos_solutions[i][7] - qpos_solutions[i-1][7]) if i > 0 else 0 for i in range(len(qpos_solutions))]
knee2_changes = [abs(qpos_solutions[i][10] - qpos_solutions[i-1][10]) if i > 0 else 0 for i in range(len(qpos_solutions))]

print(f"Knee 1 total change: {sum(knee1_changes):.4f} radians")
print(f"Knee 2 total change: {sum(knee2_changes):.4f} radians")

if sum(knee1_changes) < 0.1 and sum(knee2_changes) < 0.1:
    print("\n❌ PROBLEM: Leg joints are barely moving! Robot is likely NOT walking.")
    print("   The trajectories may represent a sliding motion, not a gait.")
else:
    print("\n✓ Leg joints are moving significantly. Robot should be walking.")

# Check com motion
com_x_range = com_traj[:, 0].max() - com_traj[:, 0].min()
com_y_range = com_traj[:, 1].max() - com_traj[:, 1].min()
com_z_range = com_traj[:, 2].max() - com_traj[:, 2].min()

print(f"\nCOM motion: X={com_x_range:.4f}, Y={com_y_range:.4f}, Z={com_z_range:.4f}")

if com_x_range > 0.1 and com_z_range < 0.01:
    print("⚠️  Robot is moving forward (X) but COM height is constant (Z).")
    print("    This suggests SLIDING motion, not walking (which would have vertical COM oscillation).")
