"""
Diagnose robot walking on sand
"""
import mujoco as mj
import numpy as np

# Load model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)
mj.mj_forward(model, data)

# Check initial positions
print("[+] ROBOT INITIAL STATE:")
print(f"    Hip position: X={data.body('hip').xpos[0]:.4f}, Y={data.body('hip').xpos[1]:.4f}, Z={data.body('hip').xpos[2]:.4f}")
print(f"    Left foot: X={data.body('foot_1').xpos[0]:.4f}, Y={data.body('foot_1').xpos[1]:.4f}, Z={data.body('foot_1').xpos[2]:.4f}")
print(f"    Right foot: X={data.body('foot_2').xpos[0]:.4f}, Y={data.body('foot_2').xpos[1]:.4f}, Z={data.body('foot_2').xpos[2]:.4f}")

print("\n[+] SAND PARAMETERS:")
print(f"    Sand starts at X=0.150m")
print(f"    Sand ends at X~0.482m")
print(f"    Sand surface Z~0.450m")

# Check joint structure
print("\n[+] ROBOT JOINTS:")
for i in range(model.nq):
    jnt = model.jnt_range[i]
    name = mj.id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
    print(f"    Joint {i}: {name}")

print("\n[+] BODIES:")
for i in range(model.nbody):
    name = mj.id2name(model, mj.mjtObj.mjOBJ_BODY, i)
    if name and ('foot' in name or 'hip' in name):
        pos = data.xpos[i]
        print(f"    Body {i}: {name} at ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
