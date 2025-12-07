#!/usr/bin/env python3
import mujoco

model = mujoco.MjModel.from_xml_path('legged_robot_sand.xml')
data = mujoco.MjData(model)

# Print body names and IDs
print("=" * 60)
print("BODY INFORMATION")
print("=" * 60)
for i in range(min(25, model.nbody)):
    try:
        name = model.body(i).name
        print(f'Body {i:3d}: {name}')
    except:
        print(f'Body {i:3d}: [error getting name]')

print()
print(f'Total bodies: {model.nbody}')
print(f'Total geoms: {model.ngeom}')

# Print geometry info
print()
print("=" * 60)
print("GEOMETRY INFORMATION (first 25)")
print("=" * 60)
for i in range(min(25, model.ngeom)):
    try:
        geom = model.geom(i)
        body_id = model.geom_bodyid[i]
        name = geom.name
        print(f'Geom {i:3d}: {name:30s} -> Body {body_id:4d}')
    except Exception as e:
        print(f'Geom {i:3d}: [error] {str(e)[:40]}')

# Find foot geoms
print()
print("=" * 60)
print("FOOT GEOMETRY")
print("=" * 60)
for i in range(model.ngeom):
    try:
        name = model.geom(i).name.lower()
        if 'foot' in name or 'l_foot' in name or 'r_foot' in name:
            body_id = model.geom_bodyid[i]
            print(f'Found foot: Geom {i} ({model.geom(i).name}) -> Body {body_id}')
    except:
        pass

# Find sand geoms
print()
print("=" * 60)
print("SAND GEOMETRY")
print("=" * 60)
sand_count = 0
for i in range(model.ngeom):
    try:
        name = model.geom(i).name.lower()
        if 'sand' in name:
            body_id = model.geom_bodyid[i]
            if sand_count < 5:  # Show first 5
                print(f'Found sand: Geom {i} ({model.geom(i).name}) -> Body {body_id}')
            sand_count += 1
    except:
        pass

print(f'Total sand geoms: {sand_count}')
