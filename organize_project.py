"""
Organize project files into correct structure
"""

import os
import shutil
from pathlib import Path

def organize_project():
    """Organize files into correct folders"""
    
    print("=" * 60)
    print("Organizing Project Structure")
    print("=" * 60)
    
    # Create necessary folders
    folders = ["src", "data", "outputs"]
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        print(f"✅ Created/verified folder: {folder}")
    
    # Files that should be in src folder
    src_files = [
        "feature_extraction.py",
        "feature_matching.py", 
        "camera_estimation.py",
        "triangulation.py",
        "bundle_adjustment.py",
        "visualization.py",
        "utils.py"
    ]
    
    # Move files to src if they exist in root
    moved_count = 0
    for file in src_files:
        if Path(file).exists():
            shutil.move(file, f"src/{file}")
            print(f"📦 Moved: {file} -> src/{file}")
            moved_count += 1
        elif Path(f"src/{file}").exists():
            print(f"✅ Already in src: {file}")
        else:
            print(f"⚠️  Missing: {file} (will be created if needed)")
    
    # Check if __init__.py exists in src
    if not Path("src/__init__.py").exists():
        with open("src/__init__.py", "w") as f:
            f.write("# SfM Package\n")
        print("✅ Created: src/__init__.py")
    
    if not Path("data/__init__.py").exists():
        with open("data/__init__.py", "w") as f:
            f.write("# Data package\n")
        print("✅ Created: data/__init__.py")
    
    print("\n" + "=" * 60)
    print(f"Organization complete! Moved {moved_count} files")
    print("=" * 60)
    
    # Show final structure
    print("\n📁 Final Project Structure:")
    print("CV PROJECT/")
    print("├── main.py")
    print("├── simple_main.py (if exists)")
    print("├── check_files.py (if exists)")
    print("├── fix_project.py (if exists)")
    print("├── src/")
    for file in src_files:
        if Path(f"src/{file}").exists():
            print(f"│   ├── {file}")
    print("├── data/")
    print("│   ├── download_real_dataset.py")
    print("│   └── test_sequence/ (generated)")
    print("└── outputs/ (generated)")

if __name__ == "__main__":
    organize_project()
    