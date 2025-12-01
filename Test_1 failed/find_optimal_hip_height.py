#!/usr/bin/env python3
"""
Find optimal hip height for walking
- Tests different hip heights to find best IK convergence
- Measures foot reachability
"""
import numpy as np
import mujoco
import os
import re

def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")
    
    with open(mjcf_path, 'r') as f:
        mjcf_content = f.read()
    
    meshes_dir = os.path.join(script_dir, "Legged_robot", "meshes")
    pattern = r'file="([^"]+\.STL)"'
    mesh_files = set(re.findall(pattern, mjcf_content))
    
    assets = {}
    for mesh_file in mesh_files:
        abs_path = os.path.join(meshes_dir, mesh_file)
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                assets[mesh_file] = f.read()
    
    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

def get_feet_positions(data, model):
    foot1_id = model.site(name='foot1_site').id
    foot2_id = model.site(name='foot2_site').id
    return data.site_xpos[foot1_id].copy(), data.site_xpos[foot2_id].copy()

model = load_model()
data = mujoco.MjData(model)

print("="*70)
print("OPTIMAL HIP HEIGHT ANALYSIS")
print("="*70)
print()

print("Testing with feet at 0.210m (ground level)...")
print("Finding hip height that puts feet closest to target")
print()
print(f"{'Hip_Z':<10} {'F1_Z (actual)':<15} {'Distance to':<15} {'Reach':<10}")
print(f"{'(m)':<10} {'(m)':<15} {'ground (mm)':<15}")
print("-" * 70)

target_foot_z = 0.210

best_hip_z = None
best_distance = float('inf')

for hip_z in np.linspace(0.200, 0.260, 31):
    data.qpos[0:3] = [0, 0, hip_z]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = [0]*6  # All joints at zero
    
    mujoco.mj_kinematics(model, data)
    foot1_pos, foot2_pos = get_feet_positions(data, model)
    
    # Average foot height
    avg_foot_z = (foot1_pos[2] + foot2_pos[2]) / 2
    distance = abs(avg_foot_z - target_foot_z)
    
    if distance < best_distance:
        best_distance = distance
        best_hip_z = hip_z
    
    print(f"{hip_z:<10.3f} {avg_foot_z:<15.4f} {distance*1000:<15.1f} {'OK' if distance < 0.015 else 'REACH':<10}")

print()
print("="*70)
print(f"RECOMMENDATION:")
print(f"  Optimal hip height: {best_hip_z:.3f}m")
print(f"  This puts feet at approximately ground level (0.210m)")
print(f"  Distance error: {best_distance*1000:.1f}mm")
print()

# Now test IK reachability at different hip heights
print("="*70)
print("IK CONVERGENCE TEST")
print("="*70)
print()

target_foot1 = np.array([0.005, 0.0, 0.210])
target_foot2 = np.array([0.000, 0.0, 0.210])

print(f"{'Hip_Z':<10} {'Min Error':<15} {'Max Error':<15} {'Converged %':<15}")
print("-" * 70)

for hip_z in np.linspace(0.210, 0.260, 11):
    data.qpos[0:3] = [0, 0, hip_z]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = [0]*6
    
    # Try simple IK for a few iterations
    for iter in range(20):
        mujoco.mj_kinematics(model, data)
        foot1_pos, foot2_pos = get_feet_positions(data, model)
        
        err1 = np.linalg.norm(target_foot1 - foot1_pos)
        err2 = np.linalg.norm(target_foot2 - foot2_pos)
        total_err = err1 + err2
        
        if total_err < 0.001:  # Converged
            break
        
        # Simple gradient step
        J = np.zeros((6, 6))
        dq = 1e-6
        for j in range(6):
            data.qpos[7 + j] += dq
            mujoco.mj_kinematics(model, data)
            f1p, f2p = get_feet_positions(data, model)
            J[0:3, j] = (f1p - foot1_pos) / dq
            J[3:6, j] = (f2p - foot2_pos) / dq
            data.qpos[7 + j] -= dq
        
        try:
            JtJ = J.T @ J + 0.01 * np.eye(6)
            dq = np.linalg.solve(JtJ, J.T @ np.concatenate([target_foot1 - foot1_pos, target_foot2 - foot2_pos]))
            data.qpos[7:] = np.clip(data.qpos[7:] + 0.05 * dq, 
                                   [-1.57, -2.0944, -1.57, -1.57, -2.0944, -1.57],
                                   [1.57, 1.0472, 1.57, 1.57, 1.0472, 1.57])
        except:
            break
    
    min_err = min(err1, err2)
    max_err = max(err1, err2)
    converged = "Yes" if total_err < 0.005 else "No"
    
    print(f"{hip_z:<10.3f} {min_err*1000:<15.1f} {max_err*1000:<15.1f} {converged:<15}")

print()
print("="*70)
print("CONCLUSION:")
print("Choose hip height that balances:")
print("  1. Feet reaching ground (0.210m)")
print("  2. IK convergence reliability")
print("  3. Natural walking motion")
print("="*70)
