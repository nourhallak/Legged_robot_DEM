"""
Analyze robot geometry to find min/max foot heights relative to hip
"""
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

# Get body IDs
hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, 'hip')
foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, 'foot1')
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, 'foot2')

print("=" * 80)
print("ROBOT GEOMETRY ANALYSIS")
print("=" * 80)

# Test different hip heights and find foot reach
hip_heights = np.linspace(0.15, 0.35, 11)
results = []

for hip_z in hip_heights:
    # Set hip at this height
    data.qpos[2] = hip_z
    data.qpos[3:7] = [1, 0, 0, 0]
    
    # Set legs to various angles to find min/max reach
    min_foot_z = float('inf')
    max_foot_z = -float('inf')
    
    # Sweep joint angles to find extremes
    for j1 in np.linspace(-1.57, 1.57, 5):
        for j2 in np.linspace(-2.0944, 1.0472, 5):
            for j3 in np.linspace(-1.57, 1.57, 5):
                data.qpos[6] = j1  # Joint 1 (hip_link_2_1)
                data.qpos[7] = j2  # Joint 2 (link_2_1_link_1_1)
                data.qpos[8] = j3  # Joint 3 (link_1_1_foot_1)
                
                # Mirror to other leg
                data.qpos[9] = j1
                data.qpos[10] = j2
                data.qpos[11] = j3
                
                mujoco.mj_kinematics(model, data)
                
                foot1_pos = data.xpos[foot1_id].copy()
                foot_z = foot1_pos[2]
                
                min_foot_z = min(min_foot_z, foot_z)
                max_foot_z = max(max_foot_z, foot_z)
    
    hip_pos = data.xpos[hip_id].copy()
    vertical_reach = max_foot_z - min_foot_z
    floor_dist_min = min_foot_z - 0.21  # Floor is at z=0.21
    
    results.append({
        'hip_z': hip_z,
        'min_foot_z': min_foot_z,
        'max_foot_z': max_foot_z,
        'reach': vertical_reach,
        'floor_clearance_min': floor_dist_min,
        'floor_clearance_max': max_foot_z - 0.21
    })
    
    print(f"\nHip Z: {hip_z:.4f}m")
    print(f"  Foot Z range: {min_foot_z:.4f}m to {max_foot_z:.4f}m")
    print(f"  Vertical reach: {vertical_reach:.4f}m")
    print(f"  Min clearance from floor (z=0.21): {floor_dist_min:.4f}m")
    print(f"  Max clearance from floor: {max_foot_z - 0.21:.4f}m")

print("\n" + "=" * 80)
print("RECOMMENDATION FOR TRAJECTORY:")
print("=" * 80)

# Find optimal hip height
best_idx = 0
best_margin = float('inf')
for i, r in enumerate(results):
    # We want hip high enough to avoid floor contact
    if r['floor_clearance_min'] > 0.005:  # 5mm safety margin
        margin = r['floor_clearance_min']
        if margin < best_margin:
            best_margin = margin
            best_idx = i

rec = results[best_idx]
print(f"\nOptimal hip base height: {rec['hip_z']:.4f}m")
print(f"  At this height:")
print(f"    Min foot Z: {rec['min_foot_z']:.4f}m")
print(f"    Max foot Z: {rec['max_foot_z']:.4f}m")
print(f"    Safety margin from floor: {rec['floor_clearance_min']:.4f}m")
print(f"\nHip should oscillate around: {rec['hip_z']:.4f}m (±0.01m)")
print(f"Feet trajectory should oscillate around: {rec['max_foot_z']:.4f}m (±0.003m)")
print("\n" + "=" * 80)
