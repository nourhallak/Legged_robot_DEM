#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Biped Robot Walking on Sand (DEM Simulation)
Complete standalone simulation with visualization and GIF export
"""

import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("2D BIPED ROBOT WALKING ON DISCRETE ELEMENT METHOD (DEM) SAND")
print("=" * 80)

# ============================================================================
# SECTION 1: Robot Parameters and Configuration
# ============================================================================

# Robot leg dimensions
L_thigh = 0.35      # Thigh length (m)
L_shank = 0.35      # Shank (calf) length (m)
L_foot = 0.1        # Foot length (m)
hip_height = 0.6    # Hip height (m)

# Robot mass distribution (kg)
mass_torso = 10.0       # Central body mass
mass_thigh = 3.5        # Each thigh mass
mass_shank = 2.5        # Each shank mass
mass_foot = 1.0         # Each foot mass
total_mass = mass_torso + 2*(mass_thigh + mass_shank + mass_foot)  # Total robot mass

# Walking parameters
step_length = 0.3       # Distance of each step (m)
step_height = 0.12      # Maximum foot swing height (m)
num_steps = 6           # Number of steps
frames_per_step = 60    # Frames per step
total_frames = num_steps * frames_per_step
hip_osc = 0.02          # Hip oscillation amplitude (m)
ground_level = 0.2      # Ground reference level (shifted up)

# Sand container
sand_container_left = -0.2
sand_container_right = 2.0
sand_container_bottom = -0.3  # Shifted up from -0.5
sand_container_top = 0.2      # Shifted up from 0.0

# Particle configuration
particle_radius = 0.015
grid_spacing = particle_radius * 2.1  # Tighter packing for denser initial sand

# Contact parameters
foot_contact_radius = 0.08
foot_contact_height = 0.15  # Vertical range above foot for sand interaction
compression_damping = 0.85
stiffness = 500.0  # Spring stiffness for sand compression

# Gravity and force calculation
g = 9.81  # Gravitational acceleration (m/s^2)

# Particle-to-particle collision parameters
particle_stiffness = 1000.0  # Spring constant for particle collisions
particle_damping = 0.8       # Damping coefficient for particle collisions
collision_check_radius = 0.04  # Search radius for nearby particles (2x particle radius)
gravity = 0                # Much weaker gravity (or 0 to disable)
time_step = 0.01             # Time step for velocity integration

print("[OK] Configuration loaded")
print(f"  Robot: thigh={L_thigh}m, shank={L_shank}m")
print(f"  Walking: {num_steps} steps, {frames_per_step} frames/step")
print(f"  Total frames: {total_frames}")

# ============================================================================
# SECTION 2: Kinematics Functions
# ============================================================================

def clamp(value, min_val=-1.0, max_val=1.0):
    """Clamp value for numerical stability - MATLAB safe acos"""
    return np.clip(value, min_val, max_val)

def inverse_kinematics(foot_x, foot_z):
    """Compute IK for 3-DOF leg - MATLAB version with safe acos and flat foot"""
    x, z = foot_x, foot_z
    
    # MATLAB: cos_knee = (x^2+z^2-L1^2-L2^2)/(2*L1*L2)
    cos_knee = (x**2 + z**2 - L_thigh**2 - L_shank**2) / (2 * L_thigh * L_shank)
    cos_knee = clamp(cos_knee)
    
    # MATLAB: knee = acos(clamp(cos_knee))
    knee_angle = np.arccos(cos_knee)
    
    # MATLAB: sin_knee from sqrt(1 - cos^2)
    sin_knee = np.sqrt(1 - cos_knee**2)
    
    # MATLAB: hip = atan2(-z,x) - atan2(L2*sin_knee, L1 + L2*cos_knee)
    hip_angle = np.arctan2(-z, x) - np.arctan2(
        L_shank * sin_knee,
        L_thigh + L_shank * cos_knee
    )
    
    # MATLAB: ankle = 0 initially, then set to -(hip+knee) for flat foot
    ankle_angle = -(hip_angle + knee_angle)
    
    return hip_angle, knee_angle, ankle_angle

def forward_kinematics(hip_angle, knee_angle, ankle_angle, hip_pos):
    """Compute FK joint positions"""
    points = np.zeros((4, 2))
    
    points[0] = hip_pos
    
    points[1, 0] = hip_pos[0] + L_thigh * np.cos(hip_angle)
    points[1, 1] = hip_pos[1] - L_thigh * np.sin(hip_angle)
    
    thigh_knee_angle = hip_angle + knee_angle
    points[2, 0] = points[1, 0] + L_shank * np.cos(thigh_knee_angle)
    points[2, 1] = points[1, 1] - L_shank * np.sin(thigh_knee_angle)
    
    foot_angle = thigh_knee_angle + ankle_angle
    points[3, 0] = points[2, 0] + L_foot * np.cos(foot_angle)
    points[3, 1] = points[2, 1] - L_foot * np.sin(foot_angle)
    
    return points

print("[OK] Kinematics functions defined")

# ============================================================================
# SECTION 3: Sand Particle System
# ============================================================================

def initialize_sand():
    """Create sand particle grid"""
    positions = []
    x = sand_container_left + particle_radius
    while x < sand_container_right - particle_radius:
        z = sand_container_bottom + particle_radius
        while z < sand_container_top - particle_radius:
            positions.append([x, z])
            z += grid_spacing
        x += grid_spacing
    
    positions = np.array(positions)
    n = len(positions)
    
    return {
        'position': positions.copy(),
        'position_initial': positions.copy(),
        'velocity': np.zeros((n, 2)),
        'radius': np.full(n, particle_radius),
        'compaction': np.zeros(n),
        'depth': np.zeros(n)
    }

sand_particles = initialize_sand()
print(f"[OK] Sand initialized with {len(sand_particles['position'])} particles")

# ============================================================================
# SECTION 4: Particle-to-Particle Collisions
# ============================================================================

def update_particle_collisions(sand_particles):
    """Handle particle-to-particle collisions using spring model"""
    positions = sand_particles['position']
    velocities = sand_particles['velocity']
    radii = sand_particles['radius']
    n = len(positions)
    
    # Apply gravity ONLY to particles that are above the ground
    # Particles on/near ground are supported by ground contact
    above_ground = positions[:, 1] > (sand_container_bottom + radii + 0.02)
    velocities[above_ground, 1] -= gravity * time_step
    
    # Update positions based on velocities
    positions[:, :] += velocities * time_step
    
    # Collision detection and response (simplified O(n²) approach)
    # For better performance with many particles, use spatial hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Distance between particle centers
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Check if particles are overlapping
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist and dist > 1e-6:
                # Normalize collision direction
                nx = dx / dist
                ny = dy / dist
                
                # Penetration depth
                penetration = min_dist - dist
                
                # Relative velocity
                dvx = velocities[j, 0] - velocities[i, 0]
                dvy = velocities[j, 1] - velocities[i, 1]
                
                # Relative velocity in collision direction
                dvn = dvx * nx + dvy * ny
                
                # Only process if particles are moving toward each other
                if dvn < 0:
                    # Spring force (repulsive)
                    force = particle_stiffness * penetration
                    
                    # Damping force
                    damping_force = particle_damping * dvn
                    
                    # Total normal force
                    total_force = force + damping_force
                    
                    # Apply impulse to both particles
                    # Equal and opposite forces (Newton's 3rd law)
                    impulse_x = total_force * nx * time_step
                    impulse_y = total_force * ny * time_step
                    
                    velocities[i, 0] -= impulse_x
                    velocities[i, 1] -= impulse_y
                    velocities[j, 0] += impulse_x
                    velocities[j, 1] += impulse_y
    
    # Ground contact - prevent particles from sinking below bottom
    at_ground = positions[:, 1] <= (sand_container_bottom + radii)
    positions[at_ground, 1] = sand_container_bottom + radii[at_ground]  # Clamp to ground
    velocities[at_ground, 1] = np.maximum(velocities[at_ground, 1], 0)  # No downward motion
    
    # Boundary constraints - reflect particles off walls
    positions[:, 0] = np.clip(positions[:, 0],
                              sand_container_left + radii,
                              sand_container_right - radii)
    positions[:, 1] = np.clip(positions[:, 1],
                              sand_container_bottom + radii,
                              sand_container_top)
    
    # Damping to avoid excessive oscillation
    velocities[:, :] *= 0.98

print("[OK] Particle collision system defined")

# ============================================================================
# SECTION 5: Contact & Deformation
# ============================================================================

def update_sand_contact(sand_particles, foot_left, foot_right):
    """Update particles based on foot contact with rectangular contact zone"""
    positions = sand_particles['position']
    velocities = sand_particles['velocity']
    radii = sand_particles['radius']
    sand_particles['depth'] = np.zeros(len(positions))
    
    # Track normal forces for documentation
    normal_forces = {
        'left_foot': [],
        'right_foot': [],
        'total_left': 0.0,
        'total_right': 0.0
    }
    
    # Rectangle contact zone dimensions (matching foot)
    total_length = L_foot * 1.3     # Total contact length (increased by 30%)
    foot_front = total_length / 2   # Half extends in front of foot (0.065m)
    foot_back = total_length / 2    # Half extends behind foot (0.065m)
    foot_half_width = 0.06         # Half foot width in Y direction
    
    # First pass: Check how many feet are in contact
    left_in_contact = False
    right_in_contact = False
    
    if not np.isnan(foot_left[0]):
        dx_left = positions[:, 0] - foot_left[0]
        dy_left = positions[:, 1] - foot_left[1]
        horizontal_contact = (dx_left > -foot_back) & (dx_left < foot_front)
        vertical_contact = (dy_left >= -foot_half_width) & (dy_left < foot_contact_height)
        contact_left = horizontal_contact & vertical_contact
        left_in_contact = np.sum(contact_left) > 0
    
    if not np.isnan(foot_right[0]):
        dx_right = positions[:, 0] - foot_right[0]
        dy_right = positions[:, 1] - foot_right[1]
        horizontal_contact = (dx_right > -foot_back) & (dx_right < foot_front)
        vertical_contact = (dy_right >= -foot_half_width) & (dy_right < foot_contact_height)
        contact_right = horizontal_contact & vertical_contact
        right_in_contact = np.sum(contact_right) > 0
    
    # Determine weight distribution based on number of feet in contact
    feet_in_contact = sum([left_in_contact, right_in_contact])
    if feet_in_contact == 0:
        weight_per_foot = 0.0  # No feet in contact
    elif feet_in_contact == 1:
        weight_per_foot = total_mass * g  # One foot supports full weight
    else:  # feet_in_contact == 2
        weight_per_foot = (total_mass * g) / 2.0  # Two feet split the weight
    
    # Left foot contact
    if not np.isnan(foot_left[0]):
        # Check if particles are within rectangular contact zone
        dx_left = positions[:, 0] - foot_left[0]
        dy_left = positions[:, 1] - foot_left[1]
        
        # Contact occurs if within rectangular bounds:
        # X: from -foot_back to +foot_front (split before/after foot center)
        # Y: particles must be AT or ABOVE foot (dy >= 0) for actual contact
        horizontal_contact = (dx_left > -foot_back) & (dx_left < foot_front)
        vertical_contact = (dy_left >= -foot_half_width) & (dy_left < foot_contact_height)
        contact_left = horizontal_contact & vertical_contact
        
        # Calculate normal force based on robot weight (depends on number of feet in contact)
        num_particles_left = np.sum(contact_left)
        if num_particles_left > 0 and weight_per_foot > 0:
            # Normal force = weight allocated to this foot
            foot_weight = weight_per_foot
            
            # Spring deflection (how much sand compresses under the weight)
            spring_deflection = foot_weight / stiffness
            
            # Force per particle (distribute weight evenly)
            force_per_particle = foot_weight / num_particles_left
            
            # Track total normal force for this foot
            normal_forces['total_left'] += foot_weight
        
        for idx in np.where(contact_left)[0]:
            dx = dx_left[idx]
            dy = dy_left[idx]
            
            if num_particles_left > 0:
                # Use weight-based normal force
                normal_force = force_per_particle
                force_mag = normal_force
                
                normal_forces['left_foot'].append(normal_force)
                
                if dy > 0:  # Particle is ABOVE the foot
                    # Upward velocity impulse (scale down for mass-based forces)
                    velocities[idx, 1] += force_mag * 0.001  # ~0.12 m/s per contact
                else:  # Particle is BELOW foot - compress it
                    # Downward velocity impulse (subtle compression)
                    velocities[idx, 1] -= force_mag * 0.0005  # ~0.06 m/s per contact
                
                # Increase compaction
                compaction_increment = min(0.25, normal_force / 100.0)
                sand_particles['compaction'][idx] = min(1.0, 
                                                       sand_particles['compaction'][idx] + compaction_increment)
                sand_particles['depth'][idx] = spring_deflection
    
    # Right foot contact
    if not np.isnan(foot_right[0]):
        # Check if particles are within rectangular contact zone
        dx_right = positions[:, 0] - foot_right[0]
        dy_right = positions[:, 1] - foot_right[1]
        
        # Contact occurs if within rectangular bounds
        horizontal_contact = (dx_right > -foot_back) & (dx_right < foot_front)
        vertical_contact = (dy_right >= -foot_half_width) & (dy_right < foot_contact_height)
        contact_right = horizontal_contact & vertical_contact
        
        # Calculate normal force based on robot weight (depends on number of feet in contact)
        num_particles_right = np.sum(contact_right)
        if num_particles_right > 0 and weight_per_foot > 0:
            # Normal force = weight allocated to this foot
            foot_weight = weight_per_foot
            
            # Spring deflection (how much sand compresses under the weight)
            spring_deflection = foot_weight / stiffness
            
            # Force per particle (distribute weight evenly)
            force_per_particle = foot_weight / num_particles_right
            
            # Track total normal force for this foot
            normal_forces['total_right'] += foot_weight
        
        for idx in np.where(contact_right)[0]:
            dx = dx_right[idx]
            dy = dy_right[idx]
            
            if num_particles_right > 0:
                # Use weight-based normal force
                normal_force = force_per_particle
                force_mag = normal_force
                
                normal_forces['right_foot'].append(normal_force)
                
                if dy > 0:  # Particle is ABOVE the foot
                    # Upward velocity impulse (scale down for mass-based forces)
                    velocities[idx, 1] += force_mag * 0.001  # ~0.12 m/s per contact
                else:  # Particle is BELOW foot - compress it
                    # Downward velocity impulse (subtle compression)
                    velocities[idx, 1] -= force_mag * 0.0005  # ~0.06 m/s per contact
                
                # Increase compaction
                compaction_increment = min(0.25, normal_force / 100.0)
                sand_particles['compaction'][idx] = min(1.0,
                                                       sand_particles['compaction'][idx] + compaction_increment)
                sand_particles['depth'][idx] = max(sand_particles['depth'][idx], spring_deflection)
    
    # Boundary constraints
    sand_particles['position'][:, 0] = np.clip(positions[:, 0],
                                              sand_container_left + radii,
                                              sand_container_right - radii)
    sand_particles['position'][:, 1] = np.clip(positions[:, 1],
                                              sand_container_bottom + radii,
                                              sand_container_top)
    
    # Physics - only apply to particles near feet (within interaction radius)
    # This prevents sand far from robot from continuously falling
    interaction_radius = foot_contact_radius * 2.0
    
    active_particles = np.zeros(len(positions), dtype=bool)
    
    if not np.isnan(foot_left[0]):
        dist_left = np.sqrt((positions[:, 0] - foot_left[0])**2 + 
                           (positions[:, 1] - foot_left[1])**2)
        active_particles |= (dist_left < interaction_radius)
    
    if not np.isnan(foot_right[0]):
        dist_right = np.sqrt((positions[:, 0] - foot_right[0])**2 + 
                            (positions[:, 1] - foot_right[1])**2)
        active_particles |= (dist_right < interaction_radius)
    
    # Apply gravity and damping only to active particles
    sand_particles['velocity'][active_particles, 1] -= 0.005
    sand_particles['velocity'][active_particles] *= compression_damping
    sand_particles['position'][active_particles] += sand_particles['velocity'][active_particles] * 0.01
    
    # Apply strong damping to inactive particles to settle them
    sand_particles['velocity'][~active_particles] *= 0.9
    
    return normal_forces

print("[OK] Contact physics defined")

# ============================================================================
# SECTION 5: Walking Trajectory
# ============================================================================

def foot_swing_trajectory(t_normalized):
    """Smooth foot swing trajectory - MATLAB version"""
    # MATLAB: [-step_length/2 + step_length*t, step_height*sin(pi*t)]
    x_rel = -step_length / 2 + step_length * t_normalized
    z_rel = step_height * np.sin(np.pi * t_normalized)
    return x_rel, z_rel

def get_sand_surface_height(sand_particles, foot_x, foot_radius):
    """Get sand surface height under foot for proper contact"""
    positions = sand_particles['position']
    # Find particles near the foot horizontally
    under_foot = np.where(np.abs(positions[:, 0] - foot_x) < foot_radius)[0]
    if len(under_foot) > 0:
        # Return highest particle position
        return np.max(positions[under_foot, 1])
    return ground_level

def generate_walking():
    """Generate complete walking sequence - MATLAB algorithm with sand contact"""
    walking_data = {
        'hip_positions': [],
        'left_leg_points': [],
        'right_leg_points': [],
        'foot_left': [],
        'foot_right': [],
    }
    
    # MATLAB: Initialize stance_foot, left_swing
    stance_foot = np.array([0.0, ground_level])
    stance_foot_fixed = stance_foot.copy()  # Keep stance foot FIXED during stance phase
    hip_pos = np.array([0.0, hip_height])
    left_swing = True
    
    for frame in range(total_frames):
        t = (frame % frames_per_step) / frames_per_step
        
        # MATLAB: Swing foot trajectory
        swing_rel = foot_swing_trajectory(t)
        swing_pos = stance_foot_fixed + swing_rel
        
        # Get sand surface height ONLY at stance foot position (which is fixed)
        stance_surface = get_sand_surface_height(sand_particles, stance_foot_fixed[0], foot_contact_radius)
        
        # MATLAB: Hip motion - hip height is determined by STANCE foot position (constant during step)
        # Hip moves horizontally from stance foot toward swing foot
        hip_x = stance_foot_fixed[0] + 0.5 * (swing_pos[0] - stance_foot_fixed[0])
        hip_z = stance_surface + hip_height + hip_osc * np.sin(np.pi * t)
        hip_pos = np.array([hip_x, hip_z])
        
        # MATLAB: Inverse kinematics for both legs
        if left_swing:
            # Left leg is swing
            hip_l, knee_l, ankle_l = inverse_kinematics(
                swing_pos[0] - hip_pos[0],
                swing_pos[1] - hip_pos[1]
            )
            ankle_l = -(hip_l + knee_l)  # Enforce flat foot
            
            # Right leg is stance - FIXED position
            hip_r, knee_r, ankle_r = inverse_kinematics(
                stance_foot_fixed[0] - hip_pos[0],
                stance_surface - hip_pos[1]
            )
            ankle_r = -(hip_r + knee_r)  # Enforce flat foot
            
            foot_left = swing_pos.copy()
            foot_right = np.array([stance_foot_fixed[0], stance_surface])
        else:
            # Right leg is swing
            hip_r, knee_r, ankle_r = inverse_kinematics(
                swing_pos[0] - hip_pos[0],
                swing_pos[1] - hip_pos[1]
            )
            ankle_r = -(hip_r + knee_r)  # Enforce flat foot
            
            # Left leg is stance - FIXED position
            hip_l, knee_l, ankle_l = inverse_kinematics(
                stance_foot_fixed[0] - hip_pos[0],
                stance_surface - hip_pos[1]
            )
            ankle_l = -(hip_l + knee_l)  # Enforce flat foot
            
            foot_left = np.array([stance_foot_fixed[0], stance_surface])
            foot_right = swing_pos.copy()
        
        # MATLAB: Forward kinematics
        left_points = forward_kinematics(hip_l, knee_l, ankle_l, hip_pos)
        right_points = forward_kinematics(hip_r, knee_r, ankle_r, hip_pos)
        
        walking_data['hip_positions'].append(hip_pos.copy())
        walking_data['left_leg_points'].append(left_points.copy())
        walking_data['right_leg_points'].append(right_points.copy())
        walking_data['foot_left'].append(foot_left.copy())
        walking_data['foot_right'].append(foot_right.copy())
        
        # MATLAB: Switch stance leg at end of step
        if (frame + 1) % frames_per_step == 0:
            stance_foot_fixed = swing_pos.copy()
            # Ensure stance foot rests on sand surface before next iteration
            stance_surface = get_sand_surface_height(sand_particles, stance_foot_fixed[0], foot_contact_radius)
            stance_foot_fixed[1] = stance_surface  # Enforce no sinking into sand
            left_swing = not left_swing
    
    return walking_data

print("[Processing] Generating walking trajectory...")
walking_data = generate_walking()
print(f"[OK] Walking trajectory generated ({total_frames} frames)")

# ============================================================================
# SECTION 6: Run Simulation
# ============================================================================

print("\nRunning simulation...")
simulation_frames = []
packing_density_history = []
normal_force_history = {'left': [], 'right': []}

for frame_idx in range(total_frames):
    foot_left = walking_data['foot_left'][frame_idx]
    foot_right = walking_data['foot_right'][frame_idx]
    
    # Update particle collisions and physics
    update_particle_collisions(sand_particles)
    
    # Update foot contact
    normal_forces = update_sand_contact(sand_particles, foot_left, foot_right)
    
    max_penetration = np.max(sand_particles['depth']) if len(sand_particles['depth']) > 0 else 0
    avg_compaction = np.mean(sand_particles['compaction'])
    
    total_area = (sand_container_right - sand_container_left) * \
                (sand_container_top - sand_container_bottom)
    particle_area = np.sum(np.pi * sand_particles['radius']**2)
    packing = particle_area / total_area
    packing_density_history.append(packing)
    
    # Track normal forces
    normal_force_history['left'].append(normal_forces['total_left'])
    normal_force_history['right'].append(normal_forces['total_right'])
    
    # CRITICAL: Update hip height based on CURRENT sand surface
    # (not the pre-calculated height from walking trajectory)
    foot_left = walking_data['foot_left'][frame_idx]
    foot_right = walking_data['foot_right'][frame_idx]
    current_hip = walking_data['hip_positions'][frame_idx].copy()
    
    # Get actual sand surface heights at current foot positions
    left_surface = get_sand_surface_height(sand_particles, foot_left[0], foot_contact_radius)
    right_surface = get_sand_surface_height(sand_particles, foot_right[0], foot_contact_radius)
    
    # Adjust hip height based on actual current sand surface
    foot_contact_level = max(left_surface, right_surface)
    t = (frame_idx % frames_per_step) / frames_per_step
    current_hip[1] = foot_contact_level + hip_height + hip_osc * np.sin(np.pi * t)
    
    # Recalculate leg points with NEW hip position based on sand deformation
    # Get the angles from the original walking data
    # We need to recalculate FK with the new hip position
    left_leg_original = walking_data['left_leg_points'][frame_idx]
    right_leg_original = walking_data['right_leg_points'][frame_idx]
    
    # The leg joints relative to hip - calculate from original positions
    # Original hip was at walking_data['hip_positions'][frame_idx]
    old_hip = walking_data['hip_positions'][frame_idx]
    
    # Translate leg points by the hip height change
    hip_height_change = current_hip[1] - old_hip[1]
    
    left_leg_updated = left_leg_original.copy()
    right_leg_updated = right_leg_original.copy()
    left_leg_updated[:, 1] += hip_height_change
    right_leg_updated[:, 1] += hip_height_change
    
    simulation_frames.append({
        'frame': frame_idx,
        'sand_position': sand_particles['position'].copy(),
        'sand_compaction': sand_particles['compaction'].copy(),
        'sand_depth': sand_particles['depth'].copy(),
        'robot_hip': current_hip,  # Use updated hip position
        'robot_left': left_leg_updated,  # Use height-adjusted leg points
        'robot_right': right_leg_updated,  # Use height-adjusted leg points
        'max_penetration': max_penetration,
        'avg_compaction': avg_compaction,
        'packing_density': packing,
        'normal_force_left': normal_forces['total_left'],
        'normal_force_right': normal_forces['total_right']
    })
    
    if frame_idx % 60 == 0:
        nf_l = normal_forces['total_left']
        nf_r = normal_forces['total_right']
        print(f"  Frame {frame_idx}/{total_frames}: F_left={nf_l:.1f}N, F_right={nf_r:.1f}N, "
              f"compaction={avg_compaction:.4f}")

print("[OK] Simulation complete!")

# ============================================================================
# SECTION 7: Render to Images
# ============================================================================

def render_frame(frame_data, slow_motion_idx, total_slow_frames):
    """Render a single frame as PIL Image"""
    
    # Image size - MATLAB-like axis: [-0.5, 2, -0.2, 0.8]
    width, height = 1600, 1000
    
    # World bounds matching MATLAB
    world_left = -0.5
    world_right = 2.0
    world_bottom = -0.2
    world_top = 0.8
    
    # Calculate scale to fit world into image
    world_width = world_right - world_left
    world_height = world_top - world_bottom
    
    margin = 50
    scale_x = (width - 2*margin) / world_width
    scale_y = (height - 2*margin) / world_height
    dpi_scale = min(scale_x, scale_y)  # Use uniform scale
    
    # Create image
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Origin for coordinate transformation (centered)
    origin_x = margin - world_left * dpi_scale
    origin_y = height - margin + world_bottom * dpi_scale
    
    def world_to_image(x, z):
        """Convert world coords to image coords"""
        img_x = origin_x + x * dpi_scale
        img_y = origin_y - z * dpi_scale  # Flip Y
        return (int(img_x), int(img_y))
    
    # Draw container
    container_tl = world_to_image(sand_container_left, sand_container_top)
    container_br = world_to_image(sand_container_right, sand_container_bottom)
    draw.rectangle([container_tl, container_br], outline='black', width=3)
    
    # Draw sand particles
    sand_pos = frame_data['sand_position']
    sand_comp = frame_data['sand_compaction']
    
    for i, pos in enumerate(sand_pos):
        img_x, img_y = world_to_image(pos[0], pos[1])
        r_pix = max(2, int(particle_radius * dpi_scale))
        
        # Color based on compaction (white to brown)
        comp_level = min(int(sand_comp[i] * 255), 255)
        color = (255, 200 - comp_level//2, 150 - comp_level//2)
        
        draw.ellipse([img_x - r_pix, img_y - r_pix,
                     img_x + r_pix, img_y + r_pix],
                    fill=color, outline='black', width=1)
    
    # Draw robot legs
    # Left leg
    left_leg = frame_data['robot_left']
    left_points = [world_to_image(p[0], p[1]) for p in left_leg]
    for i in range(len(left_points) - 1):
        draw.line([left_points[i], left_points[i+1]], fill='blue', width=4)
    for p in left_points:
        draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill='blue')
    
    # Right leg
    right_leg = frame_data['robot_right']
    right_points = [world_to_image(p[0], p[1]) for p in right_leg]
    for i in range(len(right_points) - 1):
        draw.line([right_points[i], right_points[i+1]], fill='red', width=4)
    for p in right_points:
        draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill='red')
    
    # Hip
    hip = frame_data['robot_hip']
    hip_img = world_to_image(hip[0], hip[1])
    draw.ellipse([hip_img[0]-8, hip_img[1]-8,
                 hip_img[0]+8, hip_img[1]+8], fill='black')
    
    # Torso
    torso_img = world_to_image(hip[0], hip[1] + 0.25)
    draw.ellipse([torso_img[0]-10, torso_img[1]-10,
                 torso_img[0]+10, torso_img[1]+10], fill='darkgray')
    
    # Draw info text with normal forces
    nf_left = frame_data['normal_force_left']
    nf_right = frame_data['normal_force_right']
    text = f"Frame: {slow_motion_idx}/{total_slow_frames-1} | " \
           f"F_left: {nf_left:.1f}N | F_right: {nf_right:.1f}N | " \
           f"Compaction: {frame_data['avg_compaction']:.4f}"
    draw.text((20, 20), text, fill='black')
    
    return img

# Render frames for GIF
print("\n[Processing] Rendering frames for GIF...")
output_dir = r"C:\Users\braid\OneDrive\Desktop\biped_dem"
os.makedirs(output_dir, exist_ok=True)

slow_motion_factor = 2
frame_indices = list(range(0, total_frames, slow_motion_factor))
gif_frames = []

for idx, frame_idx in enumerate(frame_indices):
    frame_data = simulation_frames[frame_idx]
    img = render_frame(frame_data, idx, len(frame_indices))
    gif_frames.append(img)
    
    if idx % 30 == 0:
        print(f"  Rendered frame {idx}/{len(frame_indices)}")

print(f"[OK] Rendered {len(gif_frames)} frames")

# ============================================================================
# SECTION 8: Save Animation Frames
# ============================================================================

print("\n[Processing] Saving animation frames...")
print("\n[Processing] Saving animation GIF...")

try:
    # Create GIF with reduced frame rate for faster encoding
    # Use every 2nd frame to reduce file size and encoding time
    reduced_frames = gif_frames[::2]  # Every 2nd frame
    
    gif_path = os.path.join(output_dir, "biped_robot_walking.gif")
    
    # Save GIF with PIL - simpler approach without optimize
    reduced_frames[0].save(
        gif_path,
        save_all=True,
        append_images=reduced_frames[1:],
        duration=150,  # 150ms per frame (fast playback)
        loop=0,
        optimize=False  # Disable optimization for speed
    )
    
    file_size = os.path.getsize(gif_path) / (1024*1024)
    duration = len(reduced_frames) * 0.15
    
    print(f"[OK] GIF saved: biped_robot_walking.gif")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Frames: {len(reduced_frames)}")
    print(f"  Duration: {duration:.1f} seconds per loop")
    
except Exception as e:
    print(f"Error saving GIF: {e}")
    print("Saving preview frame instead...")
    try:
        preview_path = os.path.join(output_dir, "animation_preview.png")
        gif_frames[0].save(preview_path, 'PNG')
        print(f"[OK] Preview saved: animation_preview.png")
    except Exception as e2:
        print(f"Error saving preview: {e2}")

# ============================================================================
# Generate Plots
# ============================================================================

print("\n[Processing] Generating analysis plots...")

penetrations = [f['max_penetration'] * 1000 for f in simulation_frames]
compactions = [f['avg_compaction'] for f in simulation_frames]
normal_left = [f['normal_force_left'] for f in simulation_frames]
normal_right = [f['normal_force_right'] for f in simulation_frames]
frame_times = np.arange(len(simulation_frames)) / 60.0  # Convert to seconds

# Plot 1: Compactness over time
plt.figure(figsize=(10, 6))
plt.plot(frame_times, compactions, linewidth=2, color='brown')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Average Sand Compaction', fontsize=12)
plt.title('Sand Compaction During Walking', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
compaction_plot = os.path.join(output_dir, "compaction_plot.png")
plt.savefig(compaction_plot, dpi=150)
plt.close()
print(f"[OK] Compaction plot saved: {compaction_plot}")

# Plot 2: Normal forces over time
plt.figure(figsize=(10, 6))
plt.plot(frame_times, normal_left, linewidth=2, label='Left Foot', color='blue')
plt.plot(frame_times, normal_right, linewidth=2, label='Right Foot', color='red')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Normal Force (N)', fontsize=12)
plt.title('Normal Forces During Walking', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
force_plot = os.path.join(output_dir, "normal_force_plot.png")
plt.savefig(force_plot, dpi=150)
plt.close()
print(f"[OK] Normal force plot saved: {force_plot}")

# Plot 3: Combined visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(frame_times, compactions, linewidth=2, color='brown')
ax1.set_ylabel('Sand Compaction', fontsize=11)
ax1.set_title('Sand Compaction and Normal Forces Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(frame_times, normal_left, linewidth=2, label='Left Foot', color='blue')
ax2.plot(frame_times, normal_right, linewidth=2, label='Right Foot', color='red')
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Normal Force (N)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
combined_plot = os.path.join(output_dir, "combined_analysis.png")
plt.savefig(combined_plot, dpi=150)
plt.close()
print(f"[OK] Combined plot saved: {combined_plot}")

print("\n" + "=" * 80)
print("SIMULATION SUMMARY")
print("=" * 80)

print(f"\nRobot Configuration:")
print(f"  Thigh: {L_thigh}m | Shank: {L_shank}m | Foot: {L_foot}m")
print(f"  Steps: {num_steps} | Frames/step: {frames_per_step}")
print(f"  Total frames: {total_frames}")

print(f"\nSand Configuration:")
print(f"  Particles: {len(sand_particles['position'])}")
print(f"  Particle radius: {particle_radius}m")
print(f"  Container: {sand_container_right - sand_container_left:.2f}m x {sand_container_top - sand_container_bottom:.2f}m")

print(f"\nNormal Force Metrics (Contact Force between Robot and Sand):")
print(f"  Left foot - Max: {max(normal_left):.1f} N, Avg: {np.mean(normal_left):.1f} N")
print(f"  Right foot - Max: {max(normal_right):.1f} N, Avg: {np.mean(normal_right):.1f} N")
print(f"  Total contact impulse: {sum(normal_left) + sum(normal_right):.1f} N*frames")

print(f"\nFoot Penetration Metrics:")
print(f"  Maximum: {max(penetrations):.2f} mm")
print(f"  Average: {np.mean(penetrations):.2f} mm")
print(f"  Minimum: {min(penetrations):.2f} mm")

print(f"\nSand Compaction Metrics:")
print(f"  Final average: {compactions[-1]:.6f}")
print(f"  Peak: {max(compactions):.6f}")
print(f"  Compacted particles (>10%): {np.sum(simulation_frames[-1]['sand_compaction'] > 0.1)}")

print(f"\nPacking Density:")
print(f"  Initial: {packing_density_history[0]:.6f}")
print(f"  Final: {packing_density_history[-1]:.6f}")
print(f"  Change: {(packing_density_history[-1] - packing_density_history[0]) / packing_density_history[0] * 100:.2f}%")

print("\n" + "=" * 80)
print("[SUCCESS] SIMULATION COMPLETE AND GIF SAVED")
print("=" * 80)
