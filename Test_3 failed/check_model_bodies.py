import mujoco
import numpy as np

# Load the model
model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

print("Model structure:")
print(f"Number of bodies: {model.nbody}")
print(f"Number of DOF: {model.nv}")
print(f"Number of qpos: {model.nq}")

for i in range(model.nbody):
    print(f"\nBody {i}: {model.body(i).name}")
    print(f"  Parent: {model.body(i).parentid}")
    print(f"  Inertia mass: {model.body(i).mass}")

print("\nDOF structure:")
for i in range(model.nv):
    print(f"DOF {i}: {model.dof_jntid[i]} -> {model.joint(model.dof_jntid[i]).name}")

# Set a specific configuration and check body positions
data.qpos[0] = 0.0  # x
data.qpos[1] = 0.0  # y
data.qpos[2] = 0.0  # z (orientation)
data.qpos[3] = 0.0  # left hip
data.qpos[4] = 0.0  # left knee
data.qpos[5] = 0.0  # left ankle
data.qpos[6] = 0.0  # right hip
data.qpos[7] = 0.0  # right knee
data.qpos[8] = 0.0  # right ankle

mujoco.mj_forward(model, data)

print("\n\nBody positions with all joints at 0:")
for i in range(model.nbody):
    print(f"Body {i} ({model.body(i).name}): {data.xpos[i]}")
