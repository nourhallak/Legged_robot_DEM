#!/usr/bin/env python3
"""
Create interactive 3D trajectory visualization using Plotly
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load trajectories
print("Loading trajectories...")
base_traj = np.load("base_trajectory.npy")
com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

num_steps = len(com_traj)

# Create figure with subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("3D Walking Path", "Forward Progression (X)", 
                    "Height Profile (Z)", "Top View (XY)"),
    specs=[[{"type": "scatter3d"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}]]
)

# ============ 3D Trajectory ============
fig.add_trace(
    go.Scatter3d(x=base_traj[:, 0], y=base_traj[:, 1], z=base_traj[:, 2],
                 mode='lines', name='Base', 
                 line=dict(color='black', width=4)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter3d(x=com_traj[:, 0], y=com_traj[:, 1], z=com_traj[:, 2],
                 mode='lines', name='COM',
                 line=dict(color='red', width=3)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter3d(x=foot1_traj[:, 0], y=foot1_traj[:, 1], z=foot1_traj[:, 2],
                 mode='lines', name='Foot 1 (Left)',
                 line=dict(color='blue', width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter3d(x=foot2_traj[:, 0], y=foot2_traj[:, 1], z=foot2_traj[:, 2],
                 mode='lines', name='Foot 2 (Right)',
                 line=dict(color='green', width=2)),
    row=1, col=1
)

# ============ X Position ============
time_steps = np.arange(num_steps)
fig.add_trace(
    go.Scatter(x=time_steps, y=base_traj[:, 0], mode='lines', name='Base X',
               line=dict(color='black', width=2)),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=time_steps, y=com_traj[:, 0], mode='lines', name='COM X',
               line=dict(color='red', width=2)),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=time_steps, y=foot1_traj[:, 0], mode='lines', name='Foot1 X',
               line=dict(color='blue', width=1, dash='dash')),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=time_steps, y=foot2_traj[:, 0], mode='lines', name='Foot2 X',
               line=dict(color='green', width=1, dash='dash')),
    row=1, col=2
)

# ============ Z Position ============
fig.add_trace(
    go.Scatter(x=time_steps, y=base_traj[:, 2], mode='lines', name='Base Z',
               line=dict(color='black', width=2)),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=time_steps, y=com_traj[:, 2], mode='lines', name='COM Z',
               line=dict(color='red', width=2)),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=time_steps, y=foot1_traj[:, 2], mode='lines', name='Foot1 Z',
               line=dict(color='blue', width=1)),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=time_steps, y=foot2_traj[:, 2], mode='lines', name='Foot2 Z',
               line=dict(color='green', width=1)),
    row=2, col=1
)
# Add ground line
fig.add_hline(y=0.212548, line_dash="dot", line_color="black", 
              annotation_text="Ground", row=2, col=1)

# ============ Top View (XY) ============
fig.add_trace(
    go.Scatter(x=base_traj[:, 0], y=base_traj[:, 1], mode='lines', name='Base path',
               line=dict(color='black', width=2)),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=com_traj[:, 0], y=com_traj[:, 1], mode='lines', name='COM path',
               line=dict(color='red', width=2)),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=foot1_traj[:, 0], y=foot1_traj[:, 1], mode='lines', name='Foot1 path',
               line=dict(color='blue', width=1, dash='dash')),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=foot2_traj[:, 0], y=foot2_traj[:, 1], mode='lines', name='Foot2 path',
               line=dict(color='green', width=1, dash='dash')),
    row=2, col=2
)

# Update axes labels and titles
fig.update_xaxes(title_text="X (forward) [m]", row=1, col=1)
fig.update_yaxes(title_text="Y (lateral) [m]", row=1, col=1)
fig.update_zaxes(title_text="Z (height) [m]", row=1, col=1)

fig.update_xaxes(title_text="Step", row=1, col=2)
fig.update_yaxes(title_text="X Position [m]", row=1, col=2)

fig.update_xaxes(title_text="Step", row=2, col=1)
fig.update_yaxes(title_text="Z Position [m]", row=2, col=1)

fig.update_xaxes(title_text="X (forward) [m]", row=2, col=2)
fig.update_yaxes(title_text="Y (lateral) [m]", row=2, col=2)

fig.update_layout(height=1000, width=1600, title_text="Humanoid Walking Trajectories")
fig.write_html("trajectories_interactive.html")
print("✓ Saved: trajectories_interactive.html")

# ============ Separate detailed 3D plot ============
fig_3d = go.Figure()

fig_3d.add_trace(go.Scatter3d(
    x=base_traj[:, 0], y=base_traj[:, 1], z=base_traj[:, 2],
    mode='lines+markers',
    name='Base',
    line=dict(color='black', width=4),
    marker=dict(size=3, opacity=0.6)
))

fig_3d.add_trace(go.Scatter3d(
    x=com_traj[:, 0], y=com_traj[:, 1], z=com_traj[:, 2],
    mode='lines+markers',
    name='COM',
    line=dict(color='red', width=3),
    marker=dict(size=3, opacity=0.6)
))

fig_3d.add_trace(go.Scatter3d(
    x=foot1_traj[:, 0], y=foot1_traj[:, 1], z=foot1_traj[:, 2],
    mode='lines',
    name='Foot 1 (Left)',
    line=dict(color='blue', width=2)
))

fig_3d.add_trace(go.Scatter3d(
    x=foot2_traj[:, 0], y=foot2_traj[:, 1], z=foot2_traj[:, 2],
    mode='lines',
    name='Foot 2 (Right)',
    line=dict(color='green', width=2)
))

fig_3d.update_layout(
    title='3D Walking Trajectories - Interactive View',
    scene=dict(
        xaxis_title='X (forward) [m]',
        yaxis_title='Y (lateral) [m]',
        zaxis_title='Z (height) [m]',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
    ),
    width=1200, height=800
)

fig_3d.write_html("trajectories_3d_interactive.html")
print("✓ Saved: trajectories_3d_interactive.html")

print("\nOpen the HTML files in your browser for interactive plots:")
print("  - trajectories_interactive.html (4 subplots)")
print("  - trajectories_3d_interactive.html (3D view)")
