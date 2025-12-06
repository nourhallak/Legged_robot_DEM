import numpy as np

# Check structure of saved arrays
traj = np.load('biped_10step_sinusoidal.npy', allow_pickle=True)
ik = np.load('biped_ik_solutions.npy', allow_pickle=True)

print("Trajectory shape:", traj.shape if hasattr(traj, 'shape') else "Object")
print("Trajectory type:", type(traj))
if hasattr(traj, 'item'):
    traj_data = traj.item()
    print("After .item():", type(traj_data))
    if isinstance(traj_data, dict):
        print("Keys:", traj_data.keys())
        for k, v in traj_data.items():
            print(f"  {k}: {v.shape if hasattr(v, 'shape') else len(v)}")

print("\nIK shape:", ik.shape if hasattr(ik, 'shape') else "Object")
print("IK type:", type(ik))
if hasattr(ik, 'item'):
    ik_data = ik.item()
    print("After .item():", type(ik_data))
    if isinstance(ik_data, dict):
        print("Keys:", ik_data.keys())
