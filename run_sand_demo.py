#!/usr/bin/env python3
"""
QUICK START - Run this to test the sand simulation immediately
"""

import subprocess
import sys
import os

def main():
    print("\n" + "="*70)
    print("BIPED ROBOT WALKING ON SAND - QUICK START")
    print("="*70)
    
    # Check if required files exist
    required_files = [
        "legged_robot_sand.xml",
        "walk_with_sand.py",
        "ik_times.npy",
        "ik_left_hip.npy",
    ]
    
    print("\n[*] Checking required files...")
    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"    [OK] {f}")
        else:
            print(f"    [MISSING] {f}")
            missing.append(f)
    
    if missing:
        print(f"\n[-] ERROR: Missing {len(missing)} file(s)")
        print("[*] Run this first:")
        print("    python generate_simple_ik.py")
        print("    python generate_sand_xml.py")
        return 1
    
    print("\n[+] All files ready!")
    print("\n[*] Starting sand walking simulation...")
    print("[*] Close the MuJoCo viewer window to stop\n")
    
    # Run the walking simulation
    try:
        result = subprocess.run(["python", "walk_with_sand.py"], check=False)
        return result.returncode
    except Exception as e:
        print(f"\n[-] ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
