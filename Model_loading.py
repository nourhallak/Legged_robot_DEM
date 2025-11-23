# c:\Users\hplap\OneDrive\Desktop\Masters\1. Fall2025\MECH620 - Intermediate Dynamics\Project\DEM using Python\Model_loading.py
import pybullet as p
import time
import pybullet_data
import trimesh
import os
import numpy as np

def load_step_and_create_bodies(step_file_path, scale=1.0):
    """
    Loads a STEP file, separates its geometries, and creates visual and collision
    shapes for PyBullet. This is a simplified example and might need adjustments
    for complex assemblies.

    Args:
        step_file_path (str): Path to the STEP file.
        scale (float): Scale factor for the model.

    Returns:
        list: A list of tuples, where each tuple contains
        (visualShapeId, collisionShapeId, mass, inertia_tensor, center_of_mass, principal_axes_quat).
    """
    if not os.path.exists(step_file_path):
        raise FileNotFoundError(f"STEP file not found at: {step_file_path}")

    # Use trimesh to load the STEP file. It can handle assemblies.
    # It returns a scene object which may contain multiple geometries.
    scene = trimesh.load(step_file_path, force='scene')
    # If the scene contains a single geometry, convert it to a scene
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)

    body_parts = []
    
    # Process each geometry in the scene as a separate body
    for geom_name in scene.geometry:
        mesh = scene.geometry[geom_name]
        
        # --- Create Visual and Collision Shapes for PyBullet ---
        # For simplicity, we use the same mesh for both.
        # In a real application, you might use a simplified convex hull for collision.
        
        # Create visual shape
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=mesh.vertices,
            indices=mesh.faces.flatten(),
            rgbaColor=[0.6, 0.6, 0.6, 1]
        )

        # Create collision shape
        # IMPORTANT: For GEOM_MESH, PyBullet requires a convex mesh for stability.
        # We create a convex hull of the original mesh for the collision shape.
        convex_hull_mesh = mesh.convex_hull
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            vertices=convex_hull_mesh.vertices,
            indices=convex_hull_mesh.faces.flatten()
        )

        # --- Define Physical Properties ---
        # Using trimesh to estimate mass and inertia.
        # Assume a uniform density.
        density = 1000  # kg/m^3
        mesh.density = density
        mass = mesh.mass
        center_of_mass = mesh.center_mass
        
        # Get the full 3x3 inertia tensor relative to the center of mass
        inertia_tensor = mesh.moment_inertia

        print(f"Processing geometry: {geom_name}")
        print(f"  - Calculated mass: {mass}")
        print(f"  - Center of Mass: {center_of_mass}")
        print(f"  - Inertia Tensor: {inertia_tensor}")

        # We pass None for the last two elements as they are no longer needed
        body_parts.append((visual_shape_id, collision_shape_id, mass, inertia_tensor, center_of_mass, None))
        
    if not body_parts:
        raise ValueError("No geometries found in the STEP file.")
        
    return body_parts


def main():
    # --- PyBullet Setup ---
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")

    # --- Load Model and Create Bodies ---
    try:
        # NOTE: Replace with the actual path to your STEP file
        step_file = "Legged_robot.step"
        body_parts = load_step_and_create_bodies(step_file)

        # This example assumes the STEP file has at least two parts to connect.
        if len(body_parts) < 2:
            print("Model has fewer than two parts. Simulating as a single rigid body.")
            part1 = body_parts[0]
            # The base position is the desired world position of the CoM minus the local CoM offset.
            base_pos = np.array([0, 0, 1]) - part1[4] 
            p.createMultiBody(
                baseMass=part1[2],
                baseCollisionShapeIndex=part1[1],
                baseVisualShapeIndex=part1[0],
                basePosition=base_pos.tolist(),
                baseOrientation=p.getQuaternionFromEuler([0,0,0]),
            )
        else:
            print(f"Found {len(body_parts)} parts. Assembling them into a robot.")
            
            # Define part indices based on your STEP file structure
            foot_idx, link1_idx, foot1_idx, link2_idx, link3_idx, link2_4_idx, hip_idx = 0, 1, 2, 3, 4, 5, 6

            # Base part (hip)
            base_part = body_parts[hip_idx]
            base_position = np.array([0, 0, 1.5])

            # Define links relative to the base
            # Each link is defined by its properties and its joint connecting to the parent
            # The parent index -1 refers to the base.
            link_definitions = [
                # Leg 1
                {'part_idx': link2_idx, 'parent_idx': -1,        'joint_pos': [0, -0.15, 0], 'joint_axis': [1, 0, 0]},
                {'part_idx': link1_idx, 'parent_idx': 0,         'joint_pos': [0, -0.15, 0], 'joint_axis': [1, 0, 0]},
                {'part_idx': foot_idx,  'parent_idx': 1,         'joint_pos': [0, -0.15, 0], 'joint_axis': [1, 0, 0]},
                # Leg 2
                {'part_idx': link2_4_idx, 'parent_idx': -1,      'joint_pos': [0, 0.15, 0], 'joint_axis': [1, 0, 0]},
                {'part_idx': link3_idx,   'parent_idx': 3,       'joint_pos': [0, 0.15, 0], 'joint_axis': [1, 0, 0]},
                {'part_idx': foot1_idx,   'parent_idx': 4,       'joint_pos': [0, 0.15, 0], 'joint_axis': [1, 0, 0]},
            ]

            # Prepare lists for createMultiBody
            linkMasses = []
            linkCollisionShapeIndices = []
            linkVisualShapeIndices = []
            linkPositions = [] # Position of the link's CoM in parent's CoM frame
            linkOrientations = []
            linkInertialFramePositions = [] # Position of the link's inertial frame in its CoM frame
            linkInertialFrameOrientations = []
            linkParentIndices = []
            linkJointTypes = []
            linkJointAxis = []

            for link_def in link_definitions:
                part_idx = link_def['part_idx']
                part = body_parts[part_idx]
                
                linkMasses.append(part[2])
                linkCollisionShapeIndices.append(part[1])
                linkVisualShapeIndices.append(part[0])
                
                # The link's position is its joint position relative to the parent's joint frame.
                # For a simple chain, this is the vector from the parent's origin to the child's origin.
                linkPositions.append(link_def['joint_pos'])
                linkOrientations.append([0, 0, 0, 1]) # Default orientation
                
                # The inertial frame position is the CoM offset from the link's origin (joint frame)
                linkInertialFramePositions.append(part[4].tolist())
                linkInertialFrameOrientations.append([0, 0, 0, 1]) # Assuming principal axes align with local frame
                
                linkParentIndices.append(link_def['parent_idx'])
                linkJointTypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append(link_def['joint_axis'])

            # Create the single multi-body robot
            robot_id = p.createMultiBody(
                baseMass=base_part[2],
                baseCollisionShapeIndex=base_part[1],
                baseVisualShapeIndex=base_part[0],
                basePosition=base_position.tolist(),
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                baseInertialFramePosition=base_part[4].tolist(),
                linkMasses=linkMasses,
                linkCollisionShapeIndices=linkCollisionShapeIndices,
                linkVisualShapeIndices=linkVisualShapeIndices,
                linkPositions=linkPositions,
                linkOrientations=linkOrientations,
                linkInertialFramePositions=linkInertialFramePositions,
                linkInertialFrameOrientations=linkInertialFrameOrientations,
                linkParentIndices=linkParentIndices,
                linkJointTypes=linkJointTypes,
                linkJointAxis=linkJointAxis
            )

        # Reset camera to focus on the assembly
        p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=70, cameraPitch=-25, cameraTargetPosition=[0.25, 0, 1])

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        p.disconnect()
        return

    # --- Simulation Loop ---
    p.setRealTimeSimulation(0)
    print("Simulation started. Close the GUI window to exit.")
    while p.isConnected():
        p.stepSimulation()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
