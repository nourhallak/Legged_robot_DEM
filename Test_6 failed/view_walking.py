#!/usr/bin/env python3
"""
Visualize the robot walking on sand in real-time
"""
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

# Initial pose - both legs ready with flexed knees
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.6
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0
data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.6
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

# Create viewer
with mjv.launch_passive(model, data) as viewer:
    sim_time = 0
    t_end = 20.0  # 20 seconds of walking
    
    print("\n" + "="*80)
    print("ROBOT WALKING VISUALIZATION")
    print("="*80)
    print("Viewing real-time walking on sand...")
    print("Close the viewer window to end simulation")
    print("="*80 + "\n")
    
    initial_x = data.body('hip').xpos[0]
    
    while sim_time < t_end and viewer.is_running():
        phase = (sim_time % 2.0) / 2.0
        
        # Leg 1 control
        if phase < 0.25:
            data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.3
            data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
            data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
        elif phase < 0.5:
            data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.0
            data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
            data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
        else:
            data.ctrl[model.actuator("hip_link_2_1_motor").id] = -0.3
            data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.3
            data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
        
        # Leg 2 control (opposite phase)
        if phase < 0.25:
            data.ctrl[model.actuator("hip_link_2_2_motor").id] = -0.3
            data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.3
            data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
        elif phase < 0.5:
            data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.0
            data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
            data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
        else:
            data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.3
            data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
            data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
        
        mj.mj_step(model, data)
        viewer.sync()
        sim_time = data.time
        
        current_x = data.body('hip').xpos[0]
        displacement = current_x - initial_x
        
        # Print progress
        if int(sim_time * 2) % 10 == 0:
            print(f"Time: {sim_time:6.2f}s | Position: {current_x:.4f}m | Displacement: {displacement:+.4f}m | VelX: {data.body('hip').cvel[3]:+.4f}m/s")
    
    final_x = data.body('hip').xpos[0]
    total_disp = final_x - initial_x
    print("\n" + "="*80)
    print(f"FINAL DISPLACEMENT: {total_disp:+.4f} m ({total_disp*100:+.2f} cm)")
    print("="*80 + "\n")
