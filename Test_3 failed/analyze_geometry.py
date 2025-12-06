"""
Analyze robot geometry and find proper hip centering
"""
import mujoco

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

print("="*70)
print("CURRENT ROBOT GEOMETRY")
print("="*70)

# Check all body positions
bodies_of_interest = ['hip', 'link_2_1', 'link_2_2', 'foot_1', 'foot_2']

for name in bodies_of_interest:
    try:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = model.body_pos[bid]
        print(f"\n{name}:")
        print(f"  Pos: X={pos[0]:8.4f}, Y={pos[1]:8.4f}, Z={pos[2]:8.4f} (m)")
        print(f"       X={pos[0]*1000:8.2f}mm, Y={pos[1]*1000:8.2f}mm, Z={pos[2]*1000:8.2f}mm")
    except:
        pass

# Check at zero angles
mujoco.mj_forward(model, data)

foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_1')
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_2')
hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hip')

foot1_pos = data.xpos[foot1_id]
foot2_pos = data.xpos[foot2_id]
hip_pos = data.xpos[hip_id]

print("\n" + "="*70)
print("FOOT POSITIONS (Zero angles)")
print("="*70)

print(f"\nFoot 1 (Left): Y = {foot1_pos[1]*1000:7.2f}mm")
print(f"Foot 2 (Right): Y = {foot2_pos[1]*1000:7.2f}mm")
print(f"Hip: Y = {hip_pos[1]*1000:7.2f}mm")

left_y = foot1_pos[1]
right_y = foot2_pos[1]
hip_y = hip_pos[1]

print(f"\nAsymmetry analysis:")
print(f"  Left foot distance from hip: {(left_y - hip_y)*1000:.2f}mm")
print(f"  Right foot distance from hip: {(right_y - hip_y)*1000:.2f}mm")
print(f"  Total distance L-R: {(left_y - right_y)*1000:.2f}mm")

# Calculate proper centering
center_y = (left_y + right_y) / 2
print(f"\n✅ Hip should be at Y = {center_y*1000:.2f}mm for perfect centering")
print(f"   Currently at: {hip_y*1000:.2f}mm")
print(f"   Offset needed: {(center_y - hip_y)*1000:+.2f}mm")
