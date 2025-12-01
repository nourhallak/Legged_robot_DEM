#!/usr/bin/env python3
"""Check foot heights with proper sequential IK and 95-5 smoothing"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

base_traj = np.load('base_trajectory.npy')
com_traj = np.load('com_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

com_id = model.site(name='com_site').id
f1_id = model.site(name='foot1_site').id
f2_id = model.site(name='foot2_site').id

def compute_ik(qpos, base_t, com_t, f1_t, f2_t, max_iters=50, debug=False):
    initial_qpos = qpos.copy()
    for iter_count in range(max_iters):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        base_pos = qpos[0:3].copy()
        com_pos = data.site_xpos[com_id].copy()
        f1_pos = data.site_xpos[f1_id].copy()
        f2_pos = data.site_xpos[f2_id].copy()
        
        errs = np.concatenate([base_t-base_pos, com_t-com_pos, f1_t-f1_pos, f2_t-f2_pos])
        err_norm = np.linalg.norm(errs)
        
        if err_norm < 0.002:
            if debug:
                print(f"  Converged at iteration {iter_count}")
            break
        
        J = np.zeros((12, 9))
        J[0:3, 0:3] = np.eye(3)
        
        for j in range(6):
            qp = qpos.copy()
            qp[6+j] += 1e-6
            data.qpos[:] = qp
            mujoco.mj_forward(model, data)
            J[:, 3+j] = (np.concatenate([qp[0:3]-base_t, data.site_xpos[com_id]-com_t, 
                                        data.site_xpos[f1_id]-f1_t, data.site_xpos[f2_id]-f2_t]) - errs) / 1e-6
        
        # Normalize errors AFTER Jacobian computation to improve numerical conditioning
        # Scale each error component to be roughly equal magnitude
        errs_normalized = errs / (err_norm + 1e-10)
        
        if debug and iter_count == 0:
            print(f"  Jacobian condition: {np.linalg.cond(J):.6e}")
            print(f"  Jacobian min: {J.min():.6e}, max: {J.max():.6e}")
        
        try:
            J_pinv = np.linalg.pinv(J, rcond=1e-6)
            if debug and iter_count == 0:
                print(f"  Pinv condition: {np.linalg.cond(J_pinv):.6e}, min: {J_pinv.min():.6e}, max: {J_pinv.max():.6e}")
        except:
            if debug:
                print(f"  Pinv FAILED")
            return qpos
        
        dq = 0.08 * J_pinv @ errs_normalized
        
        if debug and iter_count == 0:
            print(f"  dq[0:3] (base): {dq[0:3]}")
            print(f"  ||dq||: {np.linalg.norm(dq):.6e}")
        
        dq = np.clip(dq, -0.2, 0.2)
        qpos[0:3] += dq[0:3]
        for i in range(6):
            qpos[6+i] = np.clip(qpos[6+i] + dq[3+i], model.jnt_range[i,0], model.jnt_range[i,1])
        qpos[3:7] = [1,0,0,0]
    
    return qpos

print("\n=== FOOT HEIGHT CHECK (with 95-5 smoothing) ===\n")
print("Step | Base Z | Foot1 Z (mm) | Foot2 Z (mm) | Status")
print("-" * 60)

q = data.qpos.copy()
prev_q = q.copy()

for step in range(0, len(base_traj), 100):  # Fewer steps for clearer output
    # Compute IK with debug on first step
    debug = (step == 0)
    q_before = q.copy()
    q = compute_ik(q, base_traj[step], com_traj[step], foot1_traj[step], foot2_traj[step], debug=debug)
    # Apply 95-5 smoothing
    q_smooth = 0.95 * q + 0.05 * prev_q
    prev_q = q_smooth.copy()
    
    # Get foot positions
    data.qpos[:] = q_smooth
    mujoco.mj_forward(model, data)
    
    base_z = q_smooth[2]
    f1_z = data.site_xpos[f1_id, 2] * 1000  # Convert to mm
    f2_z = data.site_xpos[f2_id, 2] * 1000
    
    # Expected values
    f1_target = foot1_traj[step, 2] * 1000
    f2_target = foot2_traj[step, 2] * 1000
    
    f1_err = f1_target - f1_z
    f2_err = f2_target - f2_z
    
    status = "✓" if abs(f1_err) < 5 and abs(f2_err) < 5 else ("✗ Foot1" if abs(f1_err) > 5 else "✗ Foot2")
    
    print(f"{step:3d} | IK: {q_before[2]:.4f}→{q[2]:.4f}, Smooth: {q_smooth[2]:.4f} | {f1_z:8.1f}mm | {f2_z:8.1f}mm | {status}")

print("\nTarget ranges: Foot Z = 210-225mm (ground + 15mm swing)")
