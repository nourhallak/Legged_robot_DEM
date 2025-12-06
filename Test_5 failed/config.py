#!/usr/bin/env python3
"""
Configuration File for Biped Robot Walking

Centralized configuration for all walking trajectory and simulation parameters.
Edit this file to customize the walking behavior.
"""

# =============================================================================
# TRAJECTORY GENERATION PARAMETERS
# =============================================================================

TRAJECTORY_CONFIG = {
    # Total number of trajectory points
    'num_steps': 400,
    
    # Forward progress per step (meters)
    'stride_length': 0.005,  # 5 mm
    
    # Steps per half gait cycle
    'cycle_steps': 100,
    
    # Fraction of cycle spent in stance phase
    'stance_fraction': 0.60,  # 60% stance, 40% swing
    
    # Ground contact height (meters)
    'ground_z': 0.210,  # 210 mm
    
    # Foot clearance above ground during swing (meters)
    'swing_clearance': 0.010,  # 10 mm
    
    # Hip vertical oscillation amplitude (meters)
    'z_oscillation_amp': 0.001,  # 1 mm - REDUCED
    
    # Left-right distance between feet (meters)
    'foot_spacing': 0.020,  # 20 mm
}

# =============================================================================
# INVERSE KINEMATICS PARAMETERS
# =============================================================================

IK_CONFIG = {
    # Model file path
    'model_path': 'legged_robot_ik.xml',
    
    # Proportional gain (increase for tighter control)
    'kp': 100.0,
    
    # Derivative gain
    'kd': 10.0,
    
    # Maximum optimization iterations per point
    'max_iter': 100,
    
    # Position error threshold for success (meters)
    'error_threshold': 1e-4,  # 0.1 mm
    
    # Joint angle bounds (radians, ±pi = ±180°)
    'joint_bounds': (-3.14159, 3.14159),
}

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

SIMULATION_CONFIG = {
    # Model file path
    'model_path': 'legged_robot_ik.xml',
    
    # Simulation duration (seconds)
    'duration': 10.0,
    
    # Enable visualization/rendering
    'render': False,
    
    # Proportional controller gain
    'kp': 100.0,
    
    # Derivative controller gain
    'kd': 10.0,
    
    # Maximum torque limits (Nm)
    'torque_limits': 100.0,
    
    # Enable data logging
    'log_data': True,
    
    # Log file name
    'log_file': 'simulation_data.npy',
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

ANALYSIS_CONFIG = {
    # Generate text report
    'generate_report': True,
    
    # Report filename
    'report_file': 'trajectory_report.txt',
    
    # Generate detailed plots
    'generate_plots': True,
    
    # Plot filename
    'plot_file': 'detailed_analysis.png',
    
    # Plot DPI resolution
    'dpi': 150,
    
    # Number of steps to show in phase diagram
    'phase_diagram_steps': 150,
}

# =============================================================================
# GAIT VARIATIONS
# =============================================================================

# Predefined gait configurations (uncomment to use)

SLOW_WALK = {
    'stride_length': 0.003,  # 3 mm steps
    'swing_clearance': 0.008,  # 8 mm clearance
    'z_oscillation_amp': 0.0005,  # 0.5 mm oscillation
}

NORMAL_WALK = {
    'stride_length': 0.005,  # 5 mm steps
    'swing_clearance': 0.010,  # 10 mm clearance
    'z_oscillation_amp': 0.001,  # 1 mm oscillation - MINIMIZED
}

FAST_WALK = {
    'stride_length': 0.008,  # 8 mm steps
    'swing_clearance': 0.012,  # 12 mm clearance
    'z_oscillation_amp': 0.002,  # 2 mm oscillation - MINIMIZED
}

STIFF_WALK = {
    'z_oscillation_amp': 0.002,  # Minimal hip bobbing (rigid)
    'swing_clearance': 0.015,  # Higher foot clearance
}

SMOOTH_WALK = {
    'z_oscillation_amp': 0.002,  # Minimal hip motion (smooth)
    'swing_clearance': 0.008,  # Lower foot clearance
}

# =============================================================================
# FILE PATHS
# =============================================================================

FILE_PATHS = {
    # Input files
    'model': 'legged_robot_ik.xml',
    
    # Output trajectory files
    'base_trajectory': 'base_trajectory.npy',
    'foot1_trajectory': 'foot1_trajectory.npy',
    'foot2_trajectory': 'foot2_trajectory.npy',
    
    # IK solution files
    'left_leg_angles': 'left_leg_angles.npy',
    'right_leg_angles': 'right_leg_angles.npy',
    'left_leg_success': 'left_leg_success.npy',
    'right_leg_success': 'right_leg_success.npy',
    'left_leg_errors': 'left_leg_errors.npy',
    'right_leg_errors': 'right_leg_errors.npy',
    
    # Visualization files
    'trajectory_viz': 'walking_trajectories.png',
    'analysis_viz': 'detailed_analysis.png',
    
    # Report files
    'trajectory_report': 'trajectory_report.txt',
    'simulation_log': 'simulation_data.npy',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def apply_gait_variation(base_config, variation):
    """
    Apply a gait variation to the base configuration.
    
    Args:
        base_config: Base trajectory configuration dict
        variation: Gait variation dict (e.g., SLOW_WALK, FAST_WALK)
    
    Returns:
        Updated configuration dict
    """
    config = base_config.copy()
    config.update(variation)
    return config


def print_config(config, title="Configuration"):
    """
    Pretty-print a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        title: Section title
    """
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)
    for key, value in config.items():
        if isinstance(value, float):
            if key.endswith('_z') or 'height' in key or 'ground' in key or 'clear' in key:
                print(f"  {key:25s}: {value*1000:8.2f} mm")
            else:
                print(f"  {key:25s}: {value:8.6f}")
        elif isinstance(value, tuple):
            print(f"  {key:25s}: {value}")
        else:
            print(f"  {key:25s}: {value}")
    print("="*60)


# =============================================================================
# MAIN CONFIGURATION (Default)
# =============================================================================

# Default active configuration - change these to modify default behavior
ACTIVE_TRAJECTORY_CONFIG = TRAJECTORY_CONFIG
ACTIVE_IK_CONFIG = IK_CONFIG
ACTIVE_SIMULATION_CONFIG = SIMULATION_CONFIG

if __name__ == "__main__":
    """Show current configuration when run directly."""
    print("\n" + "█"*60)
    print("  BIPED ROBOT CONFIGURATION")
    print("█"*60)
    
    print_config(TRAJECTORY_CONFIG, "Trajectory Generation")
    print_config(IK_CONFIG, "Inverse Kinematics")
    print_config(SIMULATION_CONFIG, "Simulation")
    print_config(ANALYSIS_CONFIG, "Analysis")
    
    print("\n" + "█"*60)
    print("  GAIT VARIATIONS AVAILABLE")
    print("█"*60)
    print(f"  • SLOW_WALK    - Shorter strides, slower movement")
    print(f"  • NORMAL_WALK  - Standard bipedal walking")
    print(f"  • FAST_WALK    - Longer strides, faster movement")
    print(f"  • STIFF_WALK   - Minimal hip motion, rigid gait")
    print(f"  • SMOOTH_WALK  - Enhanced hip motion, smooth gait")
    
    print("\n  Usage:")
    print("    from config import *")
    print("    config = apply_gait_variation(TRAJECTORY_CONFIG, FAST_WALK)")
    print("    print_config(config)")
    print("\n" + "█"*60 + "\n")
