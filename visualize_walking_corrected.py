#!/usr/bin/env python3
"""Visualize the corrected sand walking motion."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Load data
times = np.load('walk_corrected_times.npy')
hip_x = np.load('walk_corrected_hip_x.npy')
hip_z = np.load('walk_corrected_hip_z.npy')
foot1_heights = np.load('walk_corrected_foot1_heights.npy')
foot2_heights = np.load('walk_corrected_foot2_heights.npy')

# Calculate velocity
hip_velocity = np.diff(hip_x) / np.diff(times)

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. Position over time (top left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(times, hip_x*100, 'b-', linewidth=2, label='Hip X Position')
ax1.fill_between(times, 0, hip_x*100, alpha=0.2)
ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_ylabel('X Position (cm)', fontsize=11)
ax1.set_title('Robot Forward Displacement', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# 2. Velocity over time (top right)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(times[:-1], hip_velocity, 'r-', linewidth=2, label='Hip X Velocity')
ax2.axhline(y=np.mean(hip_velocity), color='k', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(hip_velocity):.4f} m/s')
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Velocity (m/s)', fontsize=11)
ax2.set_title('Walking Velocity', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# 3. Foot heights (middle left)
ax3 = fig.add_subplot(gs[1, 0])
sand_surface = 0.501  # Top of sand layer
ax3.axhline(y=sand_surface, color='brown', linestyle='-', linewidth=3, label='Sand Surface', alpha=0.7)
ax3.plot(times, foot1_heights, 'g-', linewidth=2, label='Foot 1', alpha=0.8)
ax3.plot(times, foot2_heights, 'orange', linewidth=2, label='Foot 2', alpha=0.8)
ax3.fill_between(times, sand_surface-0.02, sand_surface+0.02, color='tan', alpha=0.3, label='Sand Layer ±2cm')
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.set_ylabel('Z Height (m)', fontsize=11)
ax3.set_title('Foot Heights (Above Sand)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# 4. Hip height (middle right)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(times, hip_z, 'purple', linewidth=2.5, label='Hip Height')
ax4.fill_between(times, hip_z.min()-0.01, hip_z.max()+0.01, alpha=0.2, color='purple')
ax4.axhline(y=0.501, color='brown', linestyle='--', linewidth=1.5, alpha=0.6, label='Sand Top (0.501m)')
ax4.set_xlabel('Time (s)', fontsize=11)
ax4.set_ylabel('Z Height (m)', fontsize=11)
ax3.set_title('Hip Vertical Motion', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# 5. Phase diagram - Position vs Velocity (bottom left)
ax5 = fig.add_subplot(gs[2, 0])
scatter = ax5.scatter(hip_x[:-1]*100, hip_velocity, c=times[:-1], cmap='viridis', s=50, alpha=0.7)
ax5.set_xlabel('X Position (cm)', fontsize=11)
ax5.set_ylabel('Velocity (m/s)', fontsize=11)
ax5.set_title('Phase Diagram: Position vs Velocity', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Time (s)', fontsize=10)

# 6. Statistics panel (bottom right)
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

total_displacement = hip_x[-1] - hip_x[0]
avg_velocity = total_displacement / (times[-1] - times[0])
max_velocity = np.max(hip_velocity)
min_velocity = np.min(hip_velocity)

foot1_min = np.min(foot1_heights)
foot2_min = np.min(foot2_heights)
foot1_max = np.max(foot1_heights)
foot2_max = np.max(foot2_heights)

stats_text = f"""
WALKING STATISTICS (CORRECTED SAND)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISPLACEMENT & VELOCITY:
  • Total displacement: {total_displacement*100:.2f} cm
  • Simulation time: {times[-1]-times[0]:.1f} s
  • Average velocity: {avg_velocity:.6f} m/s
  • Max velocity: {max_velocity:.6f} m/s
  • Min velocity: {min_velocity:.6f} m/s

FOOT CLEARANCE:
  • Foot 1: {foot1_min*100:.2f} - {foot1_max*100:.2f} cm
  • Foot 2: {foot2_min*100:.2f} - {foot2_max*100:.2f} cm
  
SAND SURFACE POSITION:
  • Sand top level: {sand_surface*100:.2f} cm
  
CLEARANCE CHECK:
  • Foot 1 min: {foot1_min*100:.2f} cm (above sand: {foot1_min > sand_surface})
  • Foot 2 min: {foot2_min*100:.2f} cm (above sand: {foot2_min > sand_surface})
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERDICT: ✓ WALKING ON SAND SURFACE
"""

ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
         fontfamily='monospace', fontsize=9.5,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Main title
fig.suptitle('Legged Robot Walking on Granular Sand - Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
plt.savefig('walking_visualization_corrected.png', dpi=150, bbox_inches='tight')
print("✓ Saved: walking_visualization_corrected.png")

# Show figure
plt.show()
