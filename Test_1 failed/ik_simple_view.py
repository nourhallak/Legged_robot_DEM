"""
Create a simple offline version that saves frames to check if motion is visible
"""
import mujoco
import numpy as np
import os
import re
from pathlib import Path
import time

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

def compute_ik_solution(model, data, com_target, foot1_target, foot2_target, max_iterations=5, tolerance=0.01):
    try:
        com_site_id = model.site(name='com_site').id
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id
    except Exception as e:
        raise ValueError(f"Could not find required sites: {e}")
    
    qpos = data.qpos.copy()
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
        
        for i in range(model.nq):
            jnt_idx = i - 6
            if 0 <= jnt_idx < model.jnt_range.shape[0]:
                qpos[i] = np.clip(qpos[i], model.jnt_range[jnt_idx, 0], model.jnt_range[jnt_idx, 1])
    
    return qpos, False

# Load data
com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

model = load_model_with_assets()
data = mujoco.MjData(model)

# Initialize
print("Running trajectory simulation with viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set camera
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -25
    viewer.cam.distance = 0.5
    viewer.cam.lookat[:] = [0.05, 0.0, 0.05]
    
    com_site_id = model.site(name='com_site').id
    foot1_site_id = model.site(name='foot1_site').id
    foot2_site_id = model.site(name='foot2_site').id
    
    # Initial pose
    mujoco.mj_resetData(model, data)
    
    print(f"Initial qpos: {data.qpos}")
    mujoco.mj_forward(model, data)
    print(f"Initial site positions:")
    print(f"  COM: {data.site_xpos[com_site_id]}")
    print(f"  Foot1: {data.site_xpos[foot1_site_id]}")
    print(f"  Foot2: {data.site_xpos[foot2_site_id]}")
    
    num_steps = len(com_traj)
    
    for step_idx in range(num_steps):
        if not viewer.is_running():
            break
        
        step_start = time.time()
        
        # Get targets
        com_target = com_traj[step_idx]
        foot1_target = foot1_traj[step_idx]
        foot2_target = foot2_traj[step_idx]
        
        # Solve IK
        qpos_solution, _ = compute_ik_solution(
            model, data,
            com_target,
            foot1_target,
            foot2_target,
            max_iterations=5,
            tolerance=0.01
        )
        
        # Apply solution
        data.qpos[:] = qpos_solution
        mujoco.mj_forward(model, data)
        
        # Show info every 100 steps
        if step_idx % 100 == 0:
            print(f"Step {step_idx}: COM={data.site_xpos[com_site_id]}, Foot1={data.site_xpos[foot1_site_id]}, Foot2={data.site_xpos[foot2_site_id]}")
        
        # Sync viewer
        viewer.sync()
        
        # Sleep for 10x slower playback
        time_until_next = model.opt.timestep * 10 - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
    
    print(f"Finished {num_steps} steps. Keep viewer open...")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
