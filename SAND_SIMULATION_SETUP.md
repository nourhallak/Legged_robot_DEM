# Robot Walking on Sand - Setup Complete

## Summary
Successfully created a biped robot that can walk on a sandbox with 966 sand particles arranged in 3 layers.

## Files Created

### 1. `generate_sand_xml.py`
- Programmatically generates 966 sand balls in a 3-layer configuration
- **Sand Layout:**
  - Radius: 0.0075m (7.5mm per ball)
  - Spacing: 0.015m center-to-center (prevents penetration)
  - X range: 0.0 to 0.9m
  - Y range: -0.285 to 0.285m  
  - 3 Z-layers at: 0.445m, 0.460m, 0.475m
  - ~322 balls per layer
  - Each ball mass: 0.001 kg (light, easily pushed)

### 2. `legged_robot_sand.xml`
- Complete MuJoCo model with:
  - Full biped robot kinematics
  - Ground plane at z=0.431m
  - 966 sand particles (3 layers)
  - Contact physics enabled
  - Friction: 1.0 (default)
- **Total file size:** 1050 lines (966 sand body definitions + robot)

### 3. `walk_with_sand.py`
- Main walking simulator for sand terrain
- **Features:**
  - Loads pre-computed joint angle trajectories (from `generate_simple_ik.py`)
  - Uses cubic spline interpolation for smooth motion
  - PD control for joint actuation (Kp=300, Kd=30)
  - Base motion control with forward velocity (6mm/s)
  - External force to push robot forward and damp oscillations
  - Real-time MuJoCo viewer with side-view camera (azimuth=90°)
  - Progress reporting every ~1 second of simulation

## Running the Simulation

### Quick Start (assumes you've already run generate_simple_ik.py):
```bash
python walk_with_sand.py
```

This will:
1. Load the sand model (966 particles in 3 layers)
2. Load joint angle trajectories
3. Start 50-second gait cycles (repeating)
4. Robot walks forward over sand at 6mm/s
5. Feet push sand particles as they walk

### Complete Setup (from scratch):
```bash
# 1. Generate joint trajectories (if not already done)
python generate_simple_ik.py

# 2. Generate sand XML with 966 balls
python generate_sand_xml.py

# 3. Run walking simulation on sand
python walk_with_sand.py
```

## Robot Parameters

### Walking Control
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Gait period | 50s | One complete walking cycle |
| Hip swing amplitude | ±0.6 rad | Side-to-side leg motion |
| Knee flexion | 0.2-0.3 rad | Up/down leg swing |
| Forward velocity | 6 mm/s | Continuous forward motion |
| Joint Kp | 300 | Joint position control gain |
| Joint Kd | 30 | Joint damping |
| Base Kp | 500 | Forward force strength |
| Base Kd | 50 | Oscillation damping |

### Sand Properties
| Property | Value |
|----------|-------|
| Ball radius | 7.5 mm |
| Ball mass | 1 gram |
| Friction | 1.0 |
| Total balls | 966 |
| Layers | 3 |
| Layer spacing | 15mm |

## Expected Behavior

When you run `walk_with_sand.py`:

1. **Viewer opens** with side view of robot on sand
2. **Robot stands** initially at base position
3. **Walking begins** with sinusoidal joint patterns
4. **Feet push sand** - you should see sand balls deform/compress under foot contact
5. **Robot progresses** forward (~6mm each second)
6. **Gait cycles** repeat every 50 seconds
7. **Output console** shows progress: time, cycle count, base position

Example output:
```
[t] t=   0.20s | Cycle #0 | Progress:   0.4% | Base X=   0.0010m
[t] t=   1.00s | Cycle #0 | Progress:   2.0% | Base X=   0.0060m
[t] t=  50.00s | Cycle #1 | Progress:   0.0% | Base X=   0.3000m
```

## Customization

### Adjust Walking Speed
Edit `walk_with_sand.py`, line with:
```python
target_base_x = data.time * 0.006  # Change 0.006 to desired m/s
```

### Change Sand Particle Count
Edit `generate_sand_xml.py`:
```python
sand_xml, num_sand = generate_sand_xml(num_balls=1000, ...)  # Change 1000 to desired count
```

### Adjust Sand Ball Size
Edit `generate_sand_xml.py`:
```python
def generate_sand_xml(num_balls=1000, ball_radius=0.01):  # Change 0.0075 to desired radius
```

### Change Gait Parameters
Edit `generate_simple_ik.py` to adjust:
- `gait_period` - walking cycle duration
- `hip_swing` - leg side-to-side amplitude
- `knee_flex` - leg up/down motion amplitude

## Physics Notes

- **Contact detection:** Enabled in XML (flag contact="enable")
- **Gravity:** 9.81 m/s² downward
- **Timestep:** 5ms (0.005s) - good balance of accuracy and speed
- **Sand particles:** Free to move, not constrained (they'll shift/compress under robot weight)
- **Friction:** Sand-robot and sand-ground friction is 1.0 (moderate grip)
- **Mass:** Each sand ball is 1 gram - light enough to be pushed easily

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Robot feet penetrate sand | Reduce ball_radius in generate_sand_xml.py or decrease forward velocity |
| Sand doesn't move | Check friction values, or increase robot mass if physics seems off |
| Slow simulation | Reduce num_balls in generate_sand_xml.py, or use a faster computer |
| Feet sliding on sand | Increase friction in XML (geom friction="X Y Z" default="1.0") |

## Next Steps

1. **Run the simulation** and observe sand interaction
2. **Analyze sand deformation** patterns
3. **Measure footprint depth** in sand
4. **Record joint torques** to understand pushing forces
5. **Vary sand density/friction** to simulate different soil types
6. **Compare with real bipedal locomotion** data

---

**Status:** ✅ Complete and ready to run
**Total sand particles:** 966 balls in 3 layers
**Robot model:** Full 6-DOF biped with 6 actuators
**Simulation mode:** Real-time viewer with physics

