"""
Stepping gait - explicit left/right foot advancement
"""
import mujoco as mj
import mujoco.viewer
import numpy as np
import time

# Load model with sand
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted.xml')
data = mj.MjData(model)

print("=" * 80)
print("STEPPING GAIT - EXPLICIT LEFT/RIGHT FOOT ADVANCEMENT")
print("=" * 80)

# Reset to initial state
mj.mj_resetData(model, data)

# Initial joint positions (standing)
# Left leg: hip, knee, ankle
# Right leg: hip, knee, ankle
initial_q = np.array([
    0.0,   # L-hip
    -0.5,  # L-knee (bent)
    0.5,   # L-ankle
    0.0,   # R-hip
    -0.5,  # R-knee (bent)
    0.5    # R-ankle
])

# Get joint IDs
joint_names = ['hip_link_2_1', 'link_2_1_link_1_1', 'link_1_1_foot_1',
               'hip_link_2_2', 'link_2_2_link_1_2', 'link_1_2_foot_2']
joint_ids = []
for name in joint_names:
    try:
        joint_ids.append(model.joint(name).id)
    except:
        print(f"Joint not found: {name}")

# Set initial positions
for i, jid in enumerate(joint_ids):
    data.qpos[jid] = initial_q[i]

mj.mj_forward(model, data)

# Control parameters
Kp = 600.0
Kd = 60.0

# Get body ID for force application
hip_id = model.body("hip").id

# Simulation parameters
dt = model.opt.timestep
total_time = 30.0  # 30 seconds
time_dilation = 10.0  # 10x slower
forward_force = 800.0

# Stepping gait parameters
step_height = 0.08  # Lift height for swinging foot
step_forward = 0.12  # Forward distance per step
step_period = 1.0  # Seconds per step
stance_phase = 0.6  # Fraction of cycle in stance (push) phase

print(f"[+] Stepping gait initialized")
print(f"[+] Step forward: {step_forward:.3f}m per leg, Step height: {step_height:.3f}m")
print(f"[+] Step period: {step_period:.2f}s, Stance phase: {stance_phase:.1%}")
print(f"[+] Control: Kp={Kp}, Kd={Kd}")
print(f"[+] Forward force: {forward_force}N (applied during stance)")
print(f"[+] Starting at X = 0.150m (sand beginning)")
print()

t = 0.0
last_print = 0.0

while t < total_time:
    # Calculate gait phase (0-1)
    gait_phase = (t / step_period) % 1.0
    
    # Determine left vs right stance phase
    # Left stance: 0.0-0.5, Right stance: 0.5-1.0
    left_stance = gait_phase < 0.5
    right_stance = not left_stance
    
    # Target joint positions based on gait phase
    target_q = np.zeros(6)
    
    if left_stance:
        # Left leg pushes (stance), right leg swings (swing)
        swing_phase = (gait_phase - 0.5) / 0.5  # 0->1 during right swing
        
        # Left leg (stance): compressed, pushing
        target_q[0] = 0.0          # L-hip stays back
        target_q[1] = -0.7         # L-knee compressed  
        target_q[2] = 0.7          # L-ankle compressed
        
        # Right leg (swing): lifts and moves forward
        target_q[3] = 0.15 * np.sin(np.pi * swing_phase)  # R-hip forward during swing
        target_q[4] = -0.3 + 0.4 * np.sin(np.pi * swing_phase)  # R-knee lifts
        target_q[5] = 0.3 + 0.2 * np.sin(np.pi * swing_phase)   # R-ankle lifts
        
    else:
        # Right leg pushes (stance), left leg swings (swing)
        swing_phase = (gait_phase) / 0.5  # 0->1 during left swing
        
        # Right leg (stance): compressed, pushing
        target_q[3] = 0.0          # R-hip stays back
        target_q[4] = -0.7         # R-knee compressed
        target_q[5] = 0.7          # R-ankle compressed
        
        # Left leg (swing): lifts and moves forward
        target_q[0] = 0.15 * np.sin(np.pi * swing_phase)  # L-hip forward during swing
        target_q[1] = -0.3 + 0.4 * np.sin(np.pi * swing_phase)  # L-knee lifts
        target_q[2] = 0.3 + 0.2 * np.sin(np.pi * swing_phase)   # L-ankle lifts
    
    # Joint control
    ctrl = np.zeros(model.nu)
    for i, jid in enumerate(joint_ids):
        q = data.qpos[jid]
        dq = data.qvel[jid]
        error = target_q[i] - q
        ctrl[i] = Kp * error - Kd * dq
    
    # Apply forward force during stance
    force_magnitude = forward_force if (left_stance or right_stance) else 0.0
    data.xfrc_applied[hip_id, 0] = force_magnitude
    
    # Set control
    data.ctrl[:] = ctrl
    
    # Step simulation
    mj.mj_step(model, data)
    t += dt * time_dilation
    
    # Print status every 1 second
    if t - last_print >= 1.0:
        x_pos = data.body("hip").xpos[0]
        dist = x_pos - 0.150
        phase_pct = (gait_phase * 100)
        leg = "LEFT" if left_stance else "RIGHT"
        
        # Get foot heights
        try:
            l_foot_z = data.body("foot_1").xpos[2]
            r_foot_z = data.body("foot_2").xpos[2]
            l_status = "ON" if l_foot_z < 0.455 else "UP"
            r_status = "ON" if r_foot_z < 0.455 else "UP"
        except:
            l_foot_z = r_foot_z = 0.0
            l_status = r_status = "?"
        
        print(f"T: {t:6.1f}s | Gait: {phase_pct:5.1f}% ({leg:5s}) | X: {x_pos:.4f}m | Dist: {dist:+.4f}m | L-foot: Z={l_foot_z:.4f}m {l_status} | R-foot: Z={r_foot_z:.4f}m {r_status}")
        last_print = t

print("\n[+] Simulation complete")
print(f"[+] Final position: X = {data.body('hip').xpos[0]:.4f}m")
print(f"[+] Distance traveled: {data.body('hip').xpos[0] - 0.150:.4f}m")
