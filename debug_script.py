#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to test imports and environment
"""

import os
import sys

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Script location:", os.path.abspath(__file__))

print("\nTrying to import utils module...")
try:
    import utils
    print("✓ utils module imported successfully!")
    print("utils.__file__:", utils.__file__)
except Exception as e:
    print("✗ Failed to import utils:", e)

print("\nChecking for required files...")
required_files = [
    "affinity_generator_v34.0.14.py",
    "affinity_config_v34.12.json",
    "utils.py",
    "/Users/calebcarter/PycharmProjects/PythonProject/datasources/verified_raw_travel_concepts.txt"
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✓ Found {file_path} (size: {os.path.getsize(file_path)} bytes)")
    else:
        print(f"✗ Missing {file_path}")

print("\nChecking output directory...")
output_dir = "./output_v34.14"
if os.path.exists(output_dir):
    print(f"✓ Output directory exists: {output_dir}")
    print("Contents:")
    for item in os.listdir(output_dir):
        print(f"  - {item}")
else:
    print(f"✗ Output directory missing: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Created output directory: {output_dir}")
    except Exception as e:
        print(f"  Failed to create output directory: {e}")

print("\nDebug script completed successfully") 