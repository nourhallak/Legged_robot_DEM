#!/usr/bin/env python3
"""
Complete FK + IK Pipeline

1. Analyze reachable workspace (FK)
2. Generate feasible trajectories within workspace  
3. Solve IK to find joint angles
"""

import numpy as np
from scipy.optimize import minimize
import mujoco
import matplotlib.pyplot as plt

def complete_pipeline():
    """Execute complete FK + IK pipeline."""
    
    print("\n" + "█"*80)
    print("█" + "  FORWARD KINEMATICS + INVERSE KINEMATICS PIPELINE".center(78) + "█")
    print("█"*80)
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load model with corrected mesh paths
    xml_path = os.path.join(script_dir, "legged_robot_ik.xml")
    with open(xml_path, 'r') as f:
        xml_content = f.read()
    xml_content = xml_content.replace('Legged_robot/meshes/', '../Legged_robot/meshes/')
    temp_xml = os.path.join(script_dir, "temp.xml")
    with open(temp_xml, 'w') as f:
        f.write(xml_content)
    
    model = mujoco.MjModel.from_xml_path(temp_xml)
    data = mujoco.MjData(model)
    
    # ========================================================================
    # STEP 1: Analyze Reachable Workspace
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: FORWARD KINEMATICS - WORKSPACE ANALYSIS")
    print("="*80)
    
    print("\nSampling workspace with joint ranges...")
    hip_samples = np.linspace(-np.pi/2.5, 0, 8)
    knee_samples = np.linspace(-np.pi/2.5, 0, 8)
    ankle_samples = np.linspace(0, np.pi/2.5, 8)
    
    all_positions = []
    for hip in hip_samples:
        for knee in knee_samples:
            for ankle in ankle_samples:
                data.qpos[3:6] = [hip, knee, ankle]
                mujoco.mj_forward(model, data)
                all_positions.append(data.site_xpos[0].copy())
    
    all_positions = np.array(all_positions)
    
    # Get workspace bounds with 5% margin
    x_range = (all_positions[:, 0].max() - all_positions[:, 0].min()) * 0.05
    y_range = (all_positions[:, 1].max() - all_positions[:, 1].min()) * 0.05
    z_range = (all_positions[:, 2].max() - all_positions[:, 2].min()) * 0.05
    
    x_min, x_max = all_positions[:, 0].min() + x_range, all_positions[:, 0].max() - x_range
    y_min, y_max = all_positions[:, 1].min() + y_range, all_positions[:, 1].max() - y_range
    z_min, z_max = all_positions[:, 2].min() + z_range, all_positions[:, 2].max() - z_range
    
    print(f"✓ Workspace bounds (mm):")
    print(f"  X: {x_min*1000:7.2f} to {x_max*1000:7.2f}")
    print(f"  Y: {y_min*1000:7.2f} to {y_max*1000:7.2f}")
    print(f"  Z: {z_min*1000:7.2f} to {z_max*1000:7.2f}")
    
    # ========================================================================
    # STEP 2: Generate Feasible Trajectories
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: GENERATE FEASIBLE WALKING TRAJECTORIES")
    print("="*80)
    
    NUM_STEPS = 300
    CYCLE = 100
    STANCE = 60
    SWING = 40
    
    stride = 0.008  # 8 mm per step - wider stride
    z_center = (z_min + z_max) / 2
    z_swing = (z_max - z_min) / 4
    
    # Hip body is defined at Z=0.42 in XML, so qpos Z is offset from this
    HIP_INIT_Z = 0.42  # From XML: <body name="hip" pos="0 -0.0049659 0.42">
    HIP_INIT_Y = -0.0049659
    
    base = np.zeros((NUM_STEPS, 3))
    foot1 = np.zeros((NUM_STEPS, 3))
    foot2 = np.zeros((NUM_STEPS, 3))
    
    print(f"\nGenerating with stride={stride*1000:.2f}mm, Z_center={z_center*1000:.1f}mm...")
    
    for step in range(NUM_STEPS):
        # Base: hip positioned directly above the center between feet (in stance)
        # NOTE: Hip body has only X and Y sliding joints (no Z slider)
        # So we can only control X and Y position, not Z
        contact = step // CYCLE
        base_x_world = stride * contact
        base_y_world = 0.0
        
        # Set base as qpos (only X and Y, Z is rotation which we set to 0)
        base[step, 0] = base_x_world
        base[step, 1] = base_y_world - HIP_INIT_Y  # Y offset from initial
        base[step, 2] = 0.0  # Z is rotation (hinge), keep at 0
        
        # Feet: relative positions (Y coordinates are actual foot positions, not offsets)
        # Left foot at Y = -13.43 mm, Right foot at Y = -6.4 mm (both on left side)
        LEFT_Y = -0.01343
        RIGHT_Y = -0.0064
        cycle = step % CYCLE
        
        if cycle < STANCE:
            # Stance phase
            contact = step // CYCLE
            foot1[step] = [stride * contact, LEFT_Y, z_min]
            foot2[step] = [stride * contact, RIGHT_Y, z_min]
        else:
            # Swing phase
            prog = (cycle - STANCE) / SWING
            contact = step // CYCLE
            x_curr = stride * contact
            x_next = stride * (contact + 1)
            x_swing = x_curr + (x_next - x_curr) * prog
            
            z_lift = z_swing * np.sin(np.pi * prog)
            
            foot1[step] = [x_swing, LEFT_Y, z_min + z_lift]
            foot2[step] = [stride * (contact + 1), RIGHT_Y, z_min]  # Right in stance
    
    # Alternate feet phases
    LEFT_Y = -0.01343
    RIGHT_Y = -0.0064
    
    for step in range(NUM_STEPS):
        cycle = (step + CYCLE//2) % CYCLE
        
        if cycle < STANCE:
            contact = (step + CYCLE//2) // CYCLE
            foot2[step] = [stride * contact, RIGHT_Y, z_min]
        else:
            prog = (cycle - STANCE) / SWING
            contact = (step + CYCLE//2) // CYCLE
            x_curr = stride * contact
            x_next = stride * (contact + 1)
            x_swing = x_curr + (x_next - x_curr) * prog
            z_lift = z_swing * np.sin(np.pi * prog)
            foot2[step] = [x_swing, RIGHT_Y, z_min + z_lift]
    
    print(f"✓ Trajectories generated")
    print(f"  Base: X {base[:, 0].min()*1000:.1f}-{base[:, 0].max()*1000:.1f} mm")
    print(f"  Foot1: X {foot1[:, 0].min()*1000:.1f}-{foot1[:, 0].max()*1000:.1f} mm, Z {foot1[:, 2].min()*1000:.1f}-{foot1[:, 2].max()*1000:.1f} mm")
    
    np.save(os.path.join(script_dir, "base_feasible.npy"), base)
    np.save(os.path.join(script_dir, "foot1_feasible.npy"), foot1)
    np.save(os.path.join(script_dir, "foot2_feasible.npy"), foot2)
    
    # ========================================================================
    # STEP 3: Solve Inverse Kinematics
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: INVERSE KINEMATICS - SOLVE FOR JOINT ANGLES")
    print("="*80)
    
    print(f"\nSolving IK for {NUM_STEPS} points...")
    
    q_left = np.zeros((NUM_STEPS, 3))
    q_right = np.zeros((NUM_STEPS, 3))
    err_left = np.zeros(NUM_STEPS)
    err_right = np.zeros(NUM_STEPS)
    
    CYCLE = 100
    STANCE = 60
    FLAT_FOOT_ANGLE = 0.0  # Ankle angle to keep foot flat on ground
    
    for step in range(NUM_STEPS):
        cycle = step % CYCLE
        in_stance = cycle < STANCE
        
        # Solve left foot
        def obj_left(q):
            if in_stance:
                # Stance: fix ankle to keep foot flat
                data.qpos[3:6] = [q[0], q[1], FLAT_FOOT_ANGLE]
            else:
                # Swing: let ankle move freely
                data.qpos[3:6] = q
            mujoco.mj_forward(model, data)
            return np.linalg.norm(data.site_xpos[0] - foot1[step])
        
        if in_stance:
            # Only optimize first 2 joints (hip, knee)
            q_init = q_left[step-1, :2] if step > 0 else np.array([0, -np.pi/4])
            res = minimize(obj_left, q_init, method='L-BFGS-B', 
                          bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], options={'maxiter': 100})
            q_left[step, :2] = res.x
            q_left[step, 2] = FLAT_FOOT_ANGLE
        else:
            # Optimize all 3 joints
            q_init = q_left[step-1] if step > 0 else np.array([0, -np.pi/4, np.pi/4])
            res = minimize(obj_left, q_init, method='L-BFGS-B', 
                          bounds=[(-np.pi, np.pi)]*3, options={'maxiter': 100})
            q_left[step] = res.x
        
        err_left[step] = obj_left(q_left[step, :2] if in_stance else q_left[step])
        
        # Solve right foot
        def obj_right(q):
            if in_stance:
                data.qpos[6:9] = [q[0], q[1], FLAT_FOOT_ANGLE]
            else:
                data.qpos[6:9] = q
            mujoco.mj_forward(model, data)
            return np.linalg.norm(data.site_xpos[1] - foot2[step])
        
        if in_stance:
            q_init = q_right[step-1, :2] if step > 0 else np.array([0, -np.pi/4])
            res = minimize(obj_right, q_init, method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], options={'maxiter': 100})
            q_right[step, :2] = res.x
            q_right[step, 2] = FLAT_FOOT_ANGLE
        else:
            q_init = q_right[step-1] if step > 0 else np.array([0, -np.pi/4, np.pi/4])
            res = minimize(obj_right, q_init, method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi)]*3, options={'maxiter': 100})
            q_right[step] = res.x
        
        err_right[step] = obj_right(q_right[step, :2] if in_stance else q_right[step])
        
        if (step + 1) % 50 == 0:
            print(f"  {step+1}/{NUM_STEPS}: L err {err_left[step]*1000:.3f}mm, R err {err_right[step]*1000:.3f}mm")
    
    success_l = np.sum(err_left < 0.001)
    success_r = np.sum(err_right < 0.001)
    
    print(f"\n✓ IK Complete")
    print(f"  Left:  {success_l}/{NUM_STEPS} converged, mean error {err_left.mean()*1000:.3f}mm")
    print(f"  Right: {success_r}/{NUM_STEPS} converged, mean error {err_right.mean()*1000:.3f}mm")
    
    np.save(os.path.join(script_dir, "q_left_feasible.npy"), q_left)
    np.save(os.path.join(script_dir, "q_right_feasible.npy"), q_right)
    np.save(os.path.join(script_dir, "err_left_feasible.npy"), err_left)
    np.save(os.path.join(script_dir, "err_right_feasible.npy"), err_right)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nJoint Angle Ranges (degrees):")
    print(f"  Left:  Hip {np.degrees(q_left[:, 0].min()):.1f}° to {np.degrees(q_left[:, 0].max()):.1f}°")
    print(f"         Knee {np.degrees(q_left[:, 1].min()):.1f}° to {np.degrees(q_left[:, 1].max()):.1f}°")
    print(f"         Ankle {np.degrees(q_left[:, 2].min()):.1f}° to {np.degrees(q_left[:, 2].max()):.1f}°")
    
    print(f"\n  Right: Hip {np.degrees(q_right[:, 0].min()):.1f}° to {np.degrees(q_right[:, 0].max()):.1f}°")
    print(f"         Knee {np.degrees(q_right[:, 1].min()):.1f}° to {np.degrees(q_right[:, 1].max()):.1f}°")
    print(f"         Ankle {np.degrees(q_right[:, 2].min()):.1f}° to {np.degrees(q_right[:, 2].max()):.1f}°")
    
    print(f"\nGenerated Files:")
    print(f"  ✓ base_feasible.npy, foot1_feasible.npy, foot2_feasible.npy")
    print(f"  ✓ q_left_feasible.npy, q_right_feasible.npy")
    print(f"  ✓ err_left_feasible.npy, err_right_feasible.npy")
    
    print("\n" + "█"*80)
    print("█" + "  PIPELINE COMPLETE - READY FOR SIMULATION".center(78) + "█")
    print("█"*80 + "\n")
    
    # Cleanup temp file
    import os as os2
    if os.path.exists(temp_xml):
        os2.remove(temp_xml)


if __name__ == "__main__":
    complete_pipeline()
