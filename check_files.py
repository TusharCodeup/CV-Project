"""
Check if all required files exist and have content
"""

import os
import sys

required_files = [
    "src/__init__.py",
    "src/feature_extraction.py",
    "src/feature_matching.py",
    "src/camera_estimation.py",
    "src/triangulation.py",
    "src/bundle_adjustment.py",
    "src/visualization.py",
    "src/utils.py",
    "main.py",
    "data/download_real_dataset.py",
]

print("=" * 60)
print("Checking project files...")
print("=" * 60)

all_ok = True

for file_path in required_files:
    if os.path.exists(file_path):
        # Check if file has content
        size = os.path.getsize(file_path)
        if size > 0:
            print(f"✅ {file_path} - OK ({size} bytes)")
        else:
            print(f"⚠️  {file_path} - EMPTY FILE!")
            all_ok = False
    else:
        print(f"❌ {file_path} - MISSING!")
        all_ok = False

print("=" * 60)

if all_ok:
    print("✅ All files present and have content!")
else:
    print("❌ Some files are missing or empty. Please create them.")
    
# Check Python path
print("\nPython path:")
for path in sys.path:
    print(f"  {path}")
    
print(f"\nCurrent directory: {os.getcwd()}")