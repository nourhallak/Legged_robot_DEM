"""
Verify that IK simulation actually produces motion by checking qpos evolution
"""
import numpy as np
import os
import mujoco
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

# Load trajectories
com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

model = load_model_with_assets()
data = mujoco.MjData(model)

print("=== IK SOLVER TEST ===\n")

# Simple IK solver (same as in ik_simulation.py)
def compute_ik_solution(model, data, com_target, foot1_target, foot2_target, max_iterations=5, tolerance=0.01):
    try:
        com_site_id = model.site(name='com_site').id
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id
    except Exception as e:
        raise ValueError(f"Could not find required sites: {e}")
    
    qpos_init = data.qpos.copy()
    qpos = qpos_init.copy()
    
    alpha = 0.1
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
        
        qpos = qpos + dq
        
        # Clamp to joint limits (only for actuated joints)
        for i in range(model.nq):
            jnt_idx = i - 6
            if 0 <= jnt_idx < model.jnt_range.shape[0]:
                qpos[i] = np.clip(qpos[i], model.jnt_range[jnt_idx, 0], model.jnt_range[jnt_idx, 1])
    
    return qpos, False

# Test on several frames
test_frames = [0, 50, 100, 200, 300, 399]
print("Testing IK solver on selected frames:")
print("-" * 80)

qpos_history = []

for frame_idx in test_frames:
    com_target = com_traj[frame_idx]
    foot1_target = foot1_traj[frame_idx]
    foot2_target = foot2_traj[frame_idx]
    
    # Reset to initial pose
    data.qpos[:] = np.zeros(model.nq)
    data.qpos[2] = 0.1
    data.qpos[3] = 1.0
    
    qpos_solution, success = compute_ik_solution(model, data, com_target, foot1_target, foot2_target)
    
    # Evaluate solution
    data.qpos[:] = qpos_solution
    mujoco.mj_forward(model, data)
    
    com_site_id = model.site(name='com_site').id
    foot1_site_id = model.site(name='foot1_site').id
    foot2_site_id = model.site(name='foot2_site').id
    
    com_actual = data.site_xpos[com_site_id]
    foot1_actual = data.site_xpos[foot1_site_id]
    foot2_actual = data.site_xpos[foot2_site_id]
    
    com_error = np.linalg.norm(com_target - com_actual)
    foot1_error = np.linalg.norm(foot1_target - foot1_actual)
    foot2_error = np.linalg.norm(foot2_target - foot2_actual)
    
    print(f"\nFrame {frame_idx}:")
    print(f"  Target COM: {com_target}, Error: {com_error:.6f}")
    print(f"  Target Foot1: {foot1_target}, Error: {foot1_error:.6f}")
    print(f"  Target Foot2: {foot2_target}, Error: {foot2_error:.6f}")
    print(f"  qpos: {np.round(qpos_solution[:7], 4)}")
    print(f"  qpos_change: {np.linalg.norm(qpos_solution - np.array([0, 0, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])):.6f}")
    
    qpos_history.append(qpos_solution.copy())

# Check if qpos actually changes across frames
print("\n" + "="*80)
print("Checking qpos changes across frames:")
for i in range(len(test_frames)-1):
    change = np.linalg.norm(qpos_history[i+1] - qpos_history[i])
    print(f"  Frames {test_frames[i]} -> {test_frames[i+1]}: qpos_change = {change:.6f}")
