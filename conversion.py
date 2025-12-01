import mujoco
import os
import re
import numpy as np

# --- 1. Define Paths and Constants ---

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the absolute path to the 'Legged_robot' package folder
package_dir = os.path.join(script_dir, "Legged_robot")
meshes_dir = os.path.join(package_dir, "meshes")

# Define the path to the URDF file
urdf_path = os.path.join(package_dir, "urdf", "Legged_robot.urdf")

# Define the output MJCF file path (this is the final, corrected file)
mjcf_output_path = os.path.join(script_dir, "legged_robot_ik.xml")

print(f"--- Debugging MuJoCo URDF Conversion (Post-Process Injection) ---")
print(f"Script Directory: {script_dir}")
print(f"Expected URDF Path: {urdf_path}")
print(f"Expected Meshes Dir: {meshes_dir}")

# CRITICAL CHECK 1: Ensure the meshes directory exists
if not os.path.isdir(meshes_dir):
    print(f"❌ ERROR: The meshes directory was not found at the expected location: {meshes_dir}")
    print("Please ensure your folder structure is: YOUR_FOLDER/Legged_robot/meshes/*.STL")
    exit()

def inject_sites_into_mjcf(mjcf_content):
    """
    Manually injects required <site> tags into the MJCF content by targeting 
    the specific <body> names that correspond to the URDF links.
    Uses a more robust approach to find the exact closing </body> tag.
    Sites are created but made invisible for cleaner visualization.
    """
    
    print("\n--- MJCF Site Injection Process ---")

    # Site definitions for MJCF - with very small size to make nearly invisible
    foot1_site_tag = '<site name="foot1_site" pos="0 0 0" size="0.0001" rgba="1 0 0 0"/>'
    foot2_site_tag = '<site name="foot2_site" pos="0 0 0" size="0.0001" rgba="0 0 1 0"/>'
    com_site_tag = '<site name="com_site" pos="0 0 0" size="0.0001" rgba="0 1 0 0"/>'
    
    injected_content = mjcf_content

    # Map of URDF Link Name -> MJCF Site Tag
    injections = {
        'link_2_1': com_site_tag,
        'foot_1': foot1_site_tag,
        'foot_2': foot2_site_tag,
    }

    for body_name, site_tag in injections.items():
        # Find the body opening tag
        open_pattern = r'<body\s+name=["\']' + re.escape(body_name) + r'["\'][^>]*>'
        open_match = re.search(open_pattern, injected_content)
        
        if not open_match:
            print(f"❌ Warning: Could not find <body> '{body_name}' for site injection.")
            continue
        
        # Find the matching closing </body> tag by counting open/close tags
        start_pos = open_match.end()
        open_count = 1
        pos = start_pos
        close_tag_pos = -1
        
        while pos < len(injected_content) and open_count > 0:
            next_open = injected_content.find('<body', pos)
            next_close = injected_content.find('</body>', pos)
            
            if next_close == -1:
                print(f"❌ Warning: Mismatched tags for body '{body_name}'.")
                break
            
            if next_open != -1 and next_open < next_close:
                open_count += 1
                pos = next_open + 1
            else:
                open_count -= 1
                if open_count == 0:
                    close_tag_pos = next_close
                pos = next_close + 1
        
        if close_tag_pos != -1:
            # Insert the site before the closing </body> tag
            new_content = (injected_content[:close_tag_pos] + 
                          f'\n        {site_tag}' + 
                          injected_content[close_tag_pos:])
            injected_content = new_content
            print(f"-> Injected site into body '{body_name}' (Success).")
        else:
            print(f"❌ Warning: Could not locate closing tag for body '{body_name}'.")

    return injected_content

def inject_environment_into_mjcf(mjcf_content):
    """
    Injects a floor and a set of dynamic stones into the worldbody.
    """
    print("\n--- Injecting Dynamic Environment ---")
    
    # Define the floor
    floor_xml = '    <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.1" rgba="0.8 0.9 0.8 1" condim="3"/>\n'
    
    # Define the stones
    stones_xml = ""
    num_stones = 0 # Set to 0 to disable stones

    # Find the <worldbody> tag and insert the floor and stones
    worldbody_pattern = r'(<worldbody>)'
    replacement = r'\1\n' + floor_xml + stones_xml
    
    injected_content, count = re.subn(worldbody_pattern, replacement, mjcf_content, count=1)
    if num_stones > 0:
        print(f"-> Injected floor and {num_stones} stones into worldbody.")
    else:
        print("-> Injected floor into worldbody.")
    return injected_content

def add_compiler_meshdir(mjcf_content):
    """
    Adds the meshdir attribute to the compiler tag to ensure meshes are found.
    """
    # The relative path from the script's directory to the meshes directory
    mesh_directory = "Legged_robot/meshes/"
    
    # Find the <compiler> tag and add the meshdir attribute
    pattern = r'(<compiler\s*)'
    replacement = rf'\1 meshdir="{mesh_directory}" '
    return re.sub(pattern, replacement, mjcf_content, count=1)

def scale_inertias(mjcf_content):
    """
    Scales up inertias and masses to prevent numerical instability.
    Multiplies all mass values by a scale factor.
    """
    print("\n--- Scaling Inertias for Stability ---")
    
    scale_factor = 1000  # Scale up masses by 1000x to prevent numerical issues
    
    # Find all inertial tags and scale their mass and inertias
    def scale_mass(match):
        full_tag = match.group(0)
        mass_str = match.group(1)
        original_mass = float(mass_str)
        scaled_mass = original_mass * scale_factor
        return full_tag.replace(f'mass="{mass_str}"', f'mass="{scaled_mass:.10e}"')
    
    def scale_inertia(match):
        full_tag = match.group(0)
        inertia_str = match.group(1)
        original_inertia = float(inertia_str)
        scaled_inertia = original_inertia * (scale_factor ** 2)  # Inertia scales as mass * length^2
        return full_tag.replace(f'diaginertia="{inertia_str} {match.group(2)} {match.group(3)}"', 
                              f'diaginertia="{scaled_inertia:.10e} {float(match.group(2))*scale_factor**2:.10e} {float(match.group(3))*scale_factor**2:.10e}"')
    
    # Scale mass values
    mass_pattern = r'(<inertial[^>]*mass=")([^"]+)(")'
    mjcf_content = re.sub(mass_pattern, lambda m: m.group(1) + str(float(m.group(2)) * scale_factor) + m.group(3), mjcf_content)
    
    # Scale diagonal inertias
    inertia_pattern = r'diaginertia="([^ ]+) ([^ ]+) ([^ ]+)"'
    def replace_inertia(match):
        i1, i2, i3 = float(match.group(1)), float(match.group(2)), float(match.group(3))
        return f'diaginertia="{i1*scale_factor**2:.10e} {i2*scale_factor**2:.10e} {i3*scale_factor**2:.10e}"'
    mjcf_content = re.sub(inertia_pattern, replace_inertia, mjcf_content)
    
    print(f"-> Scaled masses and inertias by factor {scale_factor}.")
    return mjcf_content

def enable_collisions(mjcf_content):
    """
    Enables collision detection for robot body parts and ground.
    Sets up geom collision groups to prevent penetration.
    """
    print("\n--- Enabling Collision Detection ---")
    
    # Enable floor collisions
    mjcf_content = mjcf_content.replace(
        '<geom name="floor" type="plane" pos="0 0 0" size="1 1 0.1" rgba="0.8 0.9 0.8 1" condim="3"/>',
        '<geom name="floor" type="plane" pos="0 0 0" size="1 1 0.1" rgba="0.8 0.9 0.8 1" condim="3" conaffinity="1" contype="1"/>'
    )
    
    # Add collision attributes to mesh geoms - be very careful with the regex
    # Pattern: <geom type="mesh" rgba="..." mesh="..."/>
    def add_collision_attrs(match):
        tag = match.group(0)
        # Add collision attributes before the closing />
        if tag.endswith('/>'):
            return tag[:-2] + ' conaffinity="1" contype="1"/>'
        else:
            return tag + ' conaffinity="1" contype="1"'
    
    geom_pattern = r'<geom\s+type="mesh"\s+rgba="[^"]+"\s+mesh="[^"]+"/?>'
    mjcf_content = re.sub(geom_pattern, add_collision_attrs, mjcf_content)
    
    print("-> Enabled collision detection for all geoms.")
    return mjcf_content

def inject_actuators(mjcf_content):
    """
    Injects a <motor> actuator for each hinge joint in the model.
    """
    print("\n--- Injecting Actuators ---")
    
    # Find all hinge joint names
    # In the intermediate MJCF, URDF 'revolute' joints are converted to 'hinge' joints.
    # We must search for 'hinge' here.
    # Since 'hinge' is the default type, the attribute may be omitted.
    # This pattern finds any <joint> that is NOT explicitly a 'free', 'ball', or 'slide' joint.
    joint_pattern = r'<joint\s+name="([^"]+)"(?![^>]*\s+type="(?:free|ball|slide)")'
    joint_names = re.findall(joint_pattern, mjcf_content)
    
    if not joint_names:
        print("-> No hinge joints found to actuate.")
        return mjcf_content
        
    actuators_xml = "\n<actuator>\n"
    for joint_name in joint_names:
        # Create a position-controlled motor for each joint with reduced gear ratio
        actuators_xml += f'    <motor name="{joint_name}_motor" joint="{joint_name}" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1"/>\n'
    actuators_xml += "</actuator>\n"
    
    # Inject the actuator block before the closing </mujoco> tag
    injected_content = mjcf_content.replace("</mujoco>", actuators_xml + "</mujoco>")
    print(f"-> Injected {len(joint_names)} motor actuators.")
    return injected_content
try:
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # The goal now is a clean conversion, so we don't need the in-memory URDF injection step.

    # Define the search pattern for mesh files
    MESH_PATTERN = r'filename="package://Legged_robot/meshes/([^"]+\.STL)"'
    all_mesh_files = set(re.findall(MESH_PATTERN, urdf_content))
    
    if not all_mesh_files:
        print("❌ ERROR: Could not find any mesh files matching the pattern 'package://Legged_robot/meshes/*.STL' in the URDF.")
        exit()

    # --- 2. Build the Assets Dictionary (Reading file content as bytes) ---
    assets = {}
    print(f"\nBuilding Assets Dictionary for {len(all_mesh_files)} files:")
    for mesh_file in all_mesh_files:
        # Construct the absolute path
        abs_path = os.path.join(meshes_dir, mesh_file)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Mesh file not found at: {abs_path}")
            
        # Read the file content as bytes ('rb' mode)
        with open(abs_path, 'rb') as f:
            asset_content = f.read()
        
        assets[mesh_file] = asset_content 
        print(f"  - '{mesh_file}' (Size: {len(asset_content)} bytes)")

    # --- 3. Modify URDF Content to use ONLY raw filenames ---
    search_prefix = "package://Legged_robot/meshes/"
    modified_urdf_content = urdf_content.replace(search_prefix, "")
    
    print("\nAttempting preliminary model load from URDF string...")
    # --- 4. Load the model from string (clean URDF, no sites yet) ---
    # This generates the intermediate MJCF XML structure
    model = mujoco.MjModel.from_xml_string(modified_urdf_content, assets=assets) 
    print("Preliminary model loaded into memory successfully.")

    # --- 5. Save the intermediate MJCF model to a temporary path ---
    temp_mjcf_path = os.path.join(script_dir, "temp_legged_robot.xml")
    mujoco.mj_saveLastXML(temp_mjcf_path, model)
    print(f"Intermediate MJCF saved to '{temp_mjcf_path}'.")

    # --- 6. Read the intermediate MJCF content ---
    with open(temp_mjcf_path, 'r', encoding='utf-8') as f:
        intermediate_mjcf_content = f.read()
    
    # --- 7b. Add the mesh directory to the compiler tag ---
    mjcf_with_meshdir = add_compiler_meshdir(intermediate_mjcf_content)

    # --- 7b2. Scale inertias for numerical stability ---
    mjcf_scaled = scale_inertias(mjcf_with_meshdir)

    # --- 7c. Inject Actuators first (before reconstruction) ---
    mjcf_with_actuators = inject_actuators(mjcf_scaled)

    # --- 7d. Inject Sites into the intermediate structure ---
    mjcf_with_sites = inject_sites_into_mjcf(mjcf_with_actuators)

    # --- 7e. Enable collisions to prevent penetration ---
    mjcf_with_collisions = enable_collisions(mjcf_with_sites)

    # --- 7f. Rebuild the XML for a robust floating base and environment ---
    # Extract the robot's body definitions (the two legs)
    worldbody_match = re.search(r'<worldbody>([\s\S]*?)</worldbody>', mjcf_with_collisions)
    worldbody_content = worldbody_match.group(1)
    
    # Remove the original static hip geom that was part of the URDF conversion
    static_hip_geom_pattern = r'\s*<geom\s+type="mesh"\s+rgba="[^"]+"\s+mesh="hip"\s*/?\s*>\s*'
    robot_bodies_xml = re.sub(static_hip_geom_pattern, '', worldbody_content)

    # Create the new XML structure from scratch
    final_mjcf_content = mjcf_with_collisions.split('<worldbody>')[0]  # Get everything before worldbody
    final_mjcf_content += "  <worldbody>\n"
    
    # Add the environment (floor only)
    final_mjcf_content += '    <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.1" rgba="0.8 0.9 0.8 1" condim="3"/>\n'

    # Add the floating hip body and nest the robot legs inside
    final_mjcf_content += '    <body name="hip" pos="0 0 0.1">\n'
    final_mjcf_content += '      <inertial mass="0.5" pos="0 0 0" diaginertia="0.001 0.001 0.001"/>\n'
    final_mjcf_content += '      <joint name="root" type="free" damping="0.1"/>\n'
    final_mjcf_content += '      <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="hip"/>\n'
    final_mjcf_content += robot_bodies_xml
    final_mjcf_content += "    </body>\n"
    final_mjcf_content += "  </worldbody>\n"
    
    # Get everything that should come after worldbody
    after_worldbody = mjcf_with_collisions.split('</worldbody>')[1]
    final_mjcf_content += after_worldbody

    # --- 8. Save the corrected MJCF content to the final path ---
    final_mjcf_content = final_mjcf_content.lstrip() # Clean any leading whitespace
    with open(mjcf_output_path, 'w', encoding='utf-8') as f:
        f.write(final_mjcf_content)
    
    print(f"✅ Successfully saved corrected MJCF to '{mjcf_output_path}'.")

    # --- 9. Final Verification: Load the corrected MJCF file itself ---
    print("\nAttempting final model load from corrected MJCF file for verification...")
    # Note: MuJoCo's load functions automatically handle the assets specified in the MJCF file if they are in the same directory (which they are, due to the previous asset definition).
    # Since we need to verify the sites are present, we load the saved XML.
    
    # We load it from the file path, as this is the standard way to verify 
    # a saved model. The assets dictionary is not needed here as the MJCF references them directly.
    model_final = mujoco.MjModel.from_xml_path(mjcf_output_path)
    
    try:
        model_final.site(name='com_site').id
        model_final.site(name='foot1_site').id
        model_final.site(name='foot2_site').id
        print("✅ Final verification successful: All target sites found in the saved MJCF model.")
    except KeyError as e:
        raise RuntimeError(f"❌ Final Verification FAILED: Target site missing in the saved MJCF. Error: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_mjcf_path):
            os.remove(temp_mjcf_path)
        print(f"Cleaned up temporary file: {temp_mjcf_path}")


    print("\nAll conversion steps completed successfully. You can now run 'ik_simulation.py'.")

except Exception as e:
    print(f"❌ MJCF Conversion failed. Error: {e}")
    print("\nIf this error persists, there may be an issue with the original URDF file content.")