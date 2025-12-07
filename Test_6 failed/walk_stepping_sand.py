"""
Proper stepping gait - feet lift and step forward on sand
"""
import mujoco as mj
import numpy as np

# Load model with low friction
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("=" * 80)
print("STEPPING GAIT ON SAND - Feet lift and step forward")
print("=" * 80)

# Reset
mj.mj_resetData(model, data)

# Joint IDs
joint_names = ['hip_link_2_1', 'link_2_1_link_1_1', 'link_1_1_foot_1',
               'hip_link_2_2', 'link_2_2_link_1_2', 'link_1_2_foot_2']
joint_ids = [model.joint(name).id for name in joint_names]

# Control parameters
Kp = 600.0
Kd = 60.0
hip_id = model.body("hip").id

# Simulation
dt = model.opt.timestep
total_time = 50.0
time_dilation = 10.0
step_period = 2.0  # 2 seconds per step

# Step parameters
step_forward = 0.15  # How far forward each step moves
step_height = 0.10  # Lift height for swing phase
stance_phase_frac = 0.7  # 70% of cycle is stance

print(f"[+] Step period: {step_period:.2f}s")
print(f"[+] Step forward: {step_forward:.3f}m per step")
print(f"[+] Step height: {step_height:.3f}m")
print(f"[+] Control: Kp={Kp}, Kd={Kd}")
print(f"[+] Starting at X=0.150m")
print()

t = 0.0
last_print = 0.0

while t < total_time:
    cycle_frac = (t / step_period) % 1.0
    
    # Determine which leg is stepping and how far in the cycle
    if cycle_frac < 0.5:
        # Right leg stepping (swinging forward)
        swing_progress = cycle_frac * 2.0  # 0->1
        stepping_leg = "RIGHT"
        
        # Right leg: swing forward
        target_r_hip = step_forward / 2.0 * np.sin(np.pi * swing_progress)  # Forward
        target_r_knee = -0.2 - 0.5 * np.sin(np.pi * swing_progress)  # Lift knee
        target_r_ankle = 0.3 + 0.3 * np.sin(np.pi * swing_progress)  # Lift ankle
        
        # Left leg: stance (support)
        target_l_hip = -0.05
        target_l_knee = -0.7
        target_l_ankle = 0.6
        
    else:
        # Left leg stepping
        swing_progress = (cycle_frac - 0.5) * 2.0  # 0->1
        stepping_leg = "LEFT"
        
        # Left leg: swing forward
        target_l_hip = step_forward / 2.0 * np.sin(np.pi * swing_progress)
        target_l_knee = -0.2 - 0.5 * np.sin(np.pi * swing_progress)
        target_l_ankle = 0.3 + 0.3 * np.sin(np.pi * swing_progress)
        
        # Right leg: stance
        target_r_hip = -0.05
        target_r_knee = -0.7
        target_r_ankle = 0.6
    
    # Joint control
    ctrl = np.zeros(model.nu)
    targets = [target_l_hip, target_l_knee, target_l_ankle,
               target_r_hip, target_r_knee, target_r_ankle]
    
    for i, jid in enumerate(joint_ids):
        q = data.qpos[jid]
        dq = data.qvel[jid]
        error = targets[i] - q
        ctrl[i] = Kp * error - Kd * dq
    
    # Forward force only during stance phase
    if cycle_frac < stance_phase_frac:
        force_mag = 1500.0  # Strong push during stance
    else:
        force_mag = 200.0   # Gentle forward during swing
    
    data.xfrc_applied[hip_id, 0] = force_mag
    
    data.ctrl[:] = ctrl
    mj.mj_step(model, data)
    t += dt * time_dilation
    
    # Print
    if t - last_print >= 1.0:
        x_pos = data.body("hip").xpos[0]
        dist = x_pos - 0.150
        
        try:
            l_z = data.body("foot_1").xpos[2]
            r_z = data.body("foot_2").xpos[2]
            l_s = "ON" if l_z < 0.455 else "UP"
            r_s = "ON" if r_z < 0.455 else "UP"
        except:
            l_s = r_s = "?"
        
        phase_pct = cycle_frac * 100
        print(f"T: {t:6.1f}s | Step: {stepping_leg} ({phase_pct:5.1f}%) | X: {x_pos:.4f}m | Dist: {dist:+.4f}m | L: {l_s} | R: {r_s}")
        last_print = t

print(f"\n[+] Final distance: {data.body('hip').xpos[0] - 0.150:.4f}m")
print(f"[+] Total steps: {(t / step_period):.0f}")
