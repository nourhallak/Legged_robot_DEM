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
    """
    
    print("\n--- MJCF Site Injection Process ---")

    # Site definitions for MJCF (using "site" tag within the body)
    foot1_site_tag = '<site name="foot1_site" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>' 
    foot2_site_tag = '<site name="foot2_site" pos="0 0 0" size="0.01" rgba="0 0 1 1"/>' 
    com_site_tag = '<site name="com_site" pos="0 0 0" size="0.01" rgba="0 1 0 1"/>' 
    
    injected_content = mjcf_content

    # Map of URDF Link Name -> MJCF Site Tag
    injections = {
        # The URDF 'hip' link becomes the MJCF 'hip' body
        'hip': com_site_tag,
        # The URDF 'foot_1' link becomes the MJCF 'foot_1' body
        'foot_1': foot1_site_tag,
        # The URDF 'foot_2' link becomes the MJCF 'foot_2' body
        'foot_2': foot2_site_tag,
    }

    for body_name, site_tag in injections.items():
        # Pattern to find the closing tag of the target body
        # We look for: (<body name="BODY_NAME" ... > ... ) (</body>)
        # Using a non-greedy match (.*?) to capture content up to the closing tag of the *inner* body.
        # MuJoCo often nests elements like geoms/joints right before the closing body tag.
        
        # This regex targets the first occurrence of the body's closing tag, 
        # ensuring the site is placed correctly inside the body.
        pattern = r'(<link\s+name="' + re.escape(body_name) + r'"[\s\S]*?)(</link>)'
        
        def replacement(match):
            # Insert the site tag before the closing </body> tag
            return match.group(1) + f'\n        {site_tag}\n    ' + match.group(2)

        # Use re.subn to perform the replacement once
        new_content, count = re.subn(pattern, replacement, injected_content, count=1, flags=re.IGNORECASE)
        
        if count == 1:
            injected_content = new_content
            print(f"-> Injected site into body '{body_name}' (Success).")
        else:
            print(f"❌ Warning: Could not find <link> '{body_name}' for site injection.")

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
    num_stones = 50
    np.random.seed(42) # for reproducibility
    for i in range(num_stones):
        # Randomly position stones in a patch
        pos_x = np.random.uniform(-0.1, 0.2)
        pos_y = np.random.uniform(-0.1, 0.1)
        pos_z = 0.005 # Start just above the floor
        stone_size = np.random.uniform(0.004, 0.008)
        
        stones_xml += f"""
    <body name="stone_{i}" pos="{pos_x} {pos_y} {pos_z}">
      <joint type="free" damping="0.01"/>
      <geom type="sphere" size="{stone_size}" mass="0.001" rgba="0.4 0.4 0.4 1" condim="3" friction="0.8 0.1 0.1"/>
    </body>
"""

    # Find the <worldbody> tag and insert the floor and stones
    worldbody_pattern = r'(<worldbody>)'
    replacement = r'\1\n' + floor_xml + stones_xml
    
    injected_content, count = re.subn(worldbody_pattern, replacement, mjcf_content, count=1)
    print(f"-> Injected floor and {num_stones} stones into worldbody.")
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
    mujoco.mj_printModel(model, temp_mjcf_path)
    print(f"Intermediate MJCF saved to '{temp_mjcf_path}'.")

    # --- 6. Read the intermediate MJCF content ---
    with open(temp_mjcf_path, 'r', encoding='utf-8') as f:
        intermediate_mjcf_content = f.read()
    
    # --- 7. Inject Sites into the MJCF XML string ---
    mjcf_with_sites = inject_sites_into_mjcf(intermediate_mjcf_content)

    # --- 7b. Inject the Environment ---
    final_mjcf_content = inject_environment_into_mjcf(mjcf_with_sites)
    
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