"""
Analyze asymmetry and find solution
"""
import mujoco

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

# Check positions
bodies = ['hip', 'link_2_1', 'link_2_2', 'foot_1', 'foot_2']

print("="*70)
print("CURRENT GEOMETRY ANALYSIS")
print("="*70)

for name in bodies:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos = model.body_pos[bid]
    print(f"{name:12} X={pos[0]*1000:8.2f}mm  Y={pos[1]*1000:8.2f}mm  Z={pos[2]*1000:8.2f}mm")

mujoco.mj_forward(model, data)

foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_1')
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_2')
hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hip')

foot1_pos = data.xpos[foot1_id]
foot2_pos = data.xpos[foot2_id]
hip_pos = data.xpos[hip_id]

print("\n" + "="*70)
print("ACTUAL FOOT POSITIONS (zero angles)")
print("="*70)

print(f"Left foot:  Y={foot1_pos[1]*1000:8.2f}mm")
print(f"Right foot: Y={foot2_pos[1]*1000:8.2f}mm")
print(f"Hip:        Y={hip_pos[1]*1000:8.2f}mm")

asymmetry = abs(foot1_pos[1] - foot2_pos[1])
print(f"\n⚠️  Asymmetry: {asymmetry*1000:.2f}mm")
print(f"    Left is {(foot1_pos[1] - hip_pos[1])*1000:.2f}mm from hip")
print(f"    Right is {(foot2_pos[1] - hip_pos[1])*1000:.2f}mm from hip")

print(f"\n💡 SOLUTION: Scale gait trajectory Y values by asymmetry ratio")
print(f"   This keeps legs connected but balances the forces")
