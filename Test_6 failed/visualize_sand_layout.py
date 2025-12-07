#!/usr/bin/env python3
"""
Visualization of sand particle layout in the simulation.
This script shows where the 966 sand balls are located in 3D space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_sand_layout():
    """Create visualization of sand particle distribution."""
    
    # Parameters matching generate_sand_xml.py
    ball_radius = 0.0075  # 7.5mm
    spacing = 2 * ball_radius  # 0.015m = 15mm
    
    z_layers = [0.445, 0.460, 0.475]  # 3 layers
    
    # Grid parameters
    y_min, y_max = -0.285, 0.285
    x_min, x_max = 0.0, 0.9
    
    balls_per_layer = 966 // 3  # ~322 per layer
    balls_per_row_y = int(np.sqrt(balls_per_layer * 0.6))  # ~16 balls
    balls_per_row_x = balls_per_layer // balls_per_row_y  # ~20 balls
    
    y_positions = np.linspace(y_min, y_max, balls_per_row_y)
    x_positions = np.linspace(x_min, x_max, balls_per_row_x)
    
    # Collect all ball positions
    all_x = []
    all_y = []
    all_z = []
    
    for z in z_layers:
        for x in x_positions:
            for y in y_positions:
                all_x.append(x)
                all_y.append(y)
                all_z.append(z)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    
    print(f"Total sand balls: {len(all_x)}")
    print(f"X range: {all_x.min():.3f} to {all_x.max():.3f}m")
    print(f"Y range: {all_y.min():.3f} to {all_y.max():.3f}m")
    print(f"Z range: {all_z.min():.3f} to {all_z.max():.3f}m")
    print()
    
    # Create visualizations
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(all_x, all_y, all_z, c=all_z, cmap='terrain', s=10)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Sand Particle Distribution\n(966 balls in 3 layers)')
    plt.colorbar(scatter, ax=ax1, label='Height (m)')
    
    # Top-down view (X-Y plane)
    ax2 = fig.add_subplot(132)
    for z_val in z_layers:
        mask = np.abs(all_z - z_val) < 0.001
        ax2.scatter(all_x[mask], all_y[mask], s=20, alpha=0.6, label=f'Z={z_val:.3f}m')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (X-Y plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Side view (X-Z plane)
    ax3 = fig.add_subplot(133)
    for i, y_val in enumerate([y_min, 0, y_max]):
        mask = np.abs(all_y - y_val) < 0.05
        ax3.scatter(all_x[mask], all_z[mask], s=20, alpha=0.6, label=f'Y={y_val:.3f}m')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (X-Z plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sand_layout_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: sand_layout_visualization.png")
    plt.show()
    
    # Print statistics
    print("\nSand Statistics:")
    print(f"  Total particles: {len(all_x)}")
    print(f"  Particles per layer: {len(all_x) // 3}")
    print(f"  Ball diameter: {2*ball_radius*1000:.1f}mm")
    print(f"  Ball spacing: {spacing*1000:.1f}mm")
    print(f"  Covered area: {(all_x.max() - all_x.min()) * (all_y.max() - all_y.min()):.3f}m²")
    print(f"  Total volume (if packed): {len(all_x) * (4/3 * np.pi * ball_radius**3):.6f}m³")
    
    # Layer statistics
    for z_val in z_layers:
        mask = np.abs(all_z - z_val) < 0.001
        count = np.sum(mask)
        print(f"\nLayer at Z={z_val:.3f}m:")
        print(f"  Particles: {count}")
        print(f"  Grid: {balls_per_row_x} × {balls_per_row_y}")

if __name__ == "__main__":
    visualize_sand_layout()
