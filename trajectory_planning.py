# c:\Users\hplap\OneDrive\Desktop\Masters\1. Fall2025\MECH620 - Intermediate Dynamics\Project\DEM using Python\trajectory_planning.py
import numpy as np

def generate_and_save_trajectories(duration=4.0, dt=0.01):
    """
    Generates walking trajectories for a bipedal robot on rigid ground.
    This creates coordinated motion for the Center of Mass (CoM) and both feet.
    """
    num_steps = int(duration / dt)
    t = np.linspace(0, duration, num_steps)

    # --- Gait Parameters ---
    # These parameters define the walking motion.
    # NOTE: These values are based on the small scale of the URDF. You may need to tune them.
    step_length = 0.04  # meters
    step_height = 0.02  # meters
    com_height = 0.05   # meters, average height of the CoM
    foot_y_separation = 0.02 # meters, lateral distance between feet
    step_time = 1.0     # seconds per step
    swing_fraction = 0.5 # fraction of step_time the foot is in the air

    # --- Initial Foot Positions ---
    # These are the neutral standing positions for the feet.
    foot1_neutral = np.array([0, -foot_y_separation / 2, 0])
    foot2_neutral = np.array([0,  foot_y_separation / 2, 0])

    # --- Trajectory Arrays ---
    com_traj = np.zeros((num_steps, 3))
    foot1_traj = np.zeros((num_steps, 3))
    foot2_traj = np.zeros((num_steps, 3))

    # --- Generate Trajectories Step-by-Step ---
    for i in range(num_steps):
        # Determine which foot is swinging based on time
        # Foot 1 swings first, then Foot 2, and so on.
        step_count = int(t[i] / step_time)
        is_foot1_swing = (step_count % 2 == 0)

        # Phase of the current step (from 0 to 1)
        phase = (t[i] % step_time) / step_time

        # Calculate current forward progression
        x_progression = step_count * step_length

        # --- Foot Trajectories ---
        if is_foot1_swing:
            # Foot 1 is swinging, Foot 2 is in stance
            stance_foot_pos = foot2_neutral + [x_progression, 0, 0]
            foot2_traj[i, :] = stance_foot_pos

            if phase < swing_fraction: # Swing part of the step
                swing_phase = phase / swing_fraction
                # Foot lifts, moves forward, and lands in a smooth motion
                foot1_traj[i, 0] = foot1_neutral[0] + x_progression + step_length * swing_phase
                foot1_traj[i, 1] = foot1_neutral[1]
                foot1_traj[i, 2] = foot1_neutral[2] + step_height * np.sin(swing_phase * np.pi)
            else: # Stance part of the step (foot is on the ground)
                foot1_traj[i, :] = foot1_neutral + [x_progression + step_length, 0, 0]

        else:
            # Foot 2 is swinging, Foot 1 is in stance
            stance_foot_pos = foot1_neutral + [x_progression, 0, 0]
            foot1_traj[i, :] = stance_foot_pos

            if phase < swing_fraction: # Swing part of the step
                swing_phase = phase / swing_fraction
                foot2_traj[i, 0] = foot2_neutral[0] + x_progression + step_length * swing_phase
                foot2_traj[i, 1] = foot2_neutral[1]
                foot2_traj[i, 2] = foot2_neutral[2] + step_height * np.sin(swing_phase * np.pi)
            else: # Stance part of the step
                foot2_traj[i, :] = foot2_neutral + [x_progression + step_length, 0, 0]

        # --- CoM Trajectory ---
        # The CoM shifts laterally to stay balanced over the stance foot.
        com_traj[i, 0] = (foot1_traj[i, 0] + foot2_traj[i, 0]) / 2 # Move forward with average foot position
        com_traj[i, 1] = stance_foot_pos[1] * 0.5 # Shift CoM towards stance foot
        com_traj[i, 2] = com_height

    # Save trajectories
    np.save("com_trajectory.npy", com_traj)
    np.save("foot1_trajectory.npy", foot1_traj)
    np.save("foot2_trajectory.npy", foot2_traj)
    print("Generated and saved 'com_trajectory.npy', 'foot1_trajectory.npy', and 'foot2_trajectory.npy'")

if __name__ == "__main__":
    generate_and_save_trajectories()

