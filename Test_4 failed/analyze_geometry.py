#!/usr/bin/env python3
"""
Check actual robot geometry and reachable foot positions
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

print("="*80)
print("ROBOT GEOMETRY ANALYSIS")
print("="*80)

# Set neutral pose
data.qpos[:] = 0
data.qpos[2] = 0.42  # hip height
mujoco.mj_forward(model, data)

hip_id = model.body(name='hip').id
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id
foot1_body_id = model.body(name='foot_1').id
foot2_body_id = model.body(name='foot_2').id

hip_pos = data.xpos[hip_id]
foot1_pos = data.site_xpos[foot1_site_id]
foot2_pos = data.site_xpos[foot2_site_id]

print(f"\nNeutral pose (all joints=0):")
print(f"  Hip position: {hip_pos}")
print(f"  Foot1 position: {foot1_pos}")
print(f"  Foot2 position: {foot2_pos}")

print(f"\nFoot separation:")
print(f"  Foot1 Y offset from hip: {(foot1_pos[1] - hip_pos[1])*1000:.2f}mm")
print(f"  Foot2 Y offset from hip: {(foot2_pos[1] - hip_pos[1])*1000:.2f}mm")
print(f"  Total foot separation: {(foot2_pos[1] - foot1_pos[1])*1000:.2f}mm")

# Try to reach extreme positions
print(f"\n" + "="*80)
print("TESTING FOOT REACHABILITY")
print("="*80)

def test_reach(leg_joints, target, name):
    """Test if a leg can reach a target position"""
    from scipy.optimize import minimize
    
    def reach_error(q):
        data.qpos[leg_joints] = q
        mujoco.mj_forward(model, data)
        if 'foot1' in name:
            actual = data.site_xpos[foot1_site_id]
        else:
            actual = data.site_xpos[foot2_site_id]
        return np.linalg.norm(actual - target)
    
    # Try to minimize
    result = minimize(reach_error, np.zeros(3), method='Nelder-Mead')
    return result.fun, result.x

# Test current trajectory targets
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

print("\nCurrent trajectory targets:")
print(f"  Foot1 Y range: [{foot1_traj[:, 1].min()*1000:.2f}, {foot1_traj[:, 1].max()*1000:.2f}]mm")
print(f"  Foot2 Y range: [{foot2_traj[:, 1].min()*1000:.2f}, {foot2_traj[:, 1].max()*1000:.2f}]mm")

# Test if feet can reach their target Y positions
print(f"\nTesting reachability of current trajectory Y positions:")

leg1_joints = [3, 4, 5]
leg2_joints = [6, 7, 8]

# Test reaching the target Y positions
test_target1 = np.array([0, foot1_traj[0, 1], 0.43])
test_target2 = np.array([0, foot2_traj[0, 1], 0.43])

err1, q1 = test_reach(leg1_joints, test_target1, "foot1")
err2, q2 = test_reach(leg2_joints, test_target2, "foot2")

print(f"  Foot1 target Y={test_target1[1]*1000:.2f}mm:")
print(f"    Achievable error: {err1*1000:.2f}mm")
print(f"    Best joint angles: {np.degrees(q1)}")

print(f"  Foot2 target Y={test_target2[1]*1000:.2f}mm:")
print(f"    Achievable error: {err2*1000:.2f}mm")
print(f"    Best joint angles: {np.degrees(q2)}")

# Check joint limits
print(f"\n" + "="*80)
print("JOINT LIMITS AND RANGES")
print("="*80)

for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if j < model.jnt_range.shape[0]:
        qmin, qmax = model.jnt_range[j]
        print(f"  {name}: [{np.degrees(qmin):.1f}°, {np.degrees(qmax):.1f}°]")
