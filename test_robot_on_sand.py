#!/usr/bin/env python3
"""
Quick test: Robot on sand with simple joint commands
"""

import numpy as np
import mujoco

def main():
    """Test robot on sand."""
    
    print("\n" + "="*80)
    print("ROBOT ON SAND - BASIC TEST")
    print("="*80 + "\n")
    
    # Load model with sand
    try:
        model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")
        data = mujoco.MjData(model)
        print(f"[+] Loaded model with {model.nbody} bodies and {model.ngeom} geoms")
    except Exception as e:
        print(f"[-] Failed to load model: {e}")
        return
    
    # Check for foot geometries
    print("\n[+] Checking geometries:")
    foot_geoms = []
    sand_geoms = list(range(1, 1001))  # Sand geoms are 1-1000
    
    # Find foot geoms by looking at foot bodies
    for body_name in ['foot_1', 'foot_2']:
        try:
            body_id = model.body(body_name).id
            body_obj = model.body(body_id)
            geomadr = int(body_obj.geomadr)
            geomnum = int(body_obj.geomnum)
            for geom_id in range(geomadr, geomadr + geomnum):
                foot_geoms.append(geom_id)
                print(f"   Found foot geom {geom_id} for {body_name}")
        except Exception as e:
            print(f"   Error finding foot geom for {body_name}: {e}")
    
    print(f"\n[+] Geometry count: {len(foot_geoms)} foot geoms, {len(sand_geoms)} sand geoms (expected 1000)")
    
    # Simple test: move joints and check for contacts
    print(f"\n[+] Running 1 second simulation with simple joint motions...")
    
    contact_count = 0
    contact_detail = {}
    
    for step in range(100):  # 100 * 0.01s = 1 second
        # Simple sine wave motion on joints
        for i in range(6):
            data.ctrl[i] = 0.5 * np.sin(data.time * 2 * np.pi)
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Check contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check if foot-sand contact (foot geom IDs are 1004, 1007; sand are 1-1000)
            is_foot_contact = geom1 in foot_geoms or geom2 in foot_geoms
            is_sand_contact = geom1 in sand_geoms or geom2 in sand_geoms
            
            if is_foot_contact and is_sand_contact:
                contact_count += 1
                foot_geom = geom1 if geom1 in foot_geoms else geom2
                sand_geom = geom2 if geom1 in foot_geoms else geom1
                
                key = f"foot {foot_geom} <-> sand {sand_geom}"
                contact_detail[key] = contact_detail.get(key, 0) + 1
    
    print(f"\n[+] Total foot-sand contacts detected: {contact_count}")
    if contact_detail:
        print("[+] Contact details:")
        for contact_pair, count in sorted(contact_detail.items(), key=lambda x: -x[1])[:10]:
            print(f"    {contact_pair}: {count} contacts")
    else:
        print("[-] NO FOOT-SAND CONTACTS DETECTED!")
    
    # Check foot positions
    print(f"\n[+] Final positions:")
    for body_name in ['foot_1', 'foot_2']:
        body_id = model.body(body_name).id
        pos = data.xpos[body_id]
        print(f"   {body_name}: X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")
    
    print("\n[+] Test complete\n")

if __name__ == "__main__":
    main()
