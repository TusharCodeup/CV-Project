"""
Complete Integration: SfM + AirCanvas Pro
Full 3D drawing studio with reconstructed objects
"""

import os
import sys
from pathlib import Path

def run_full_pipeline():
    """Run complete SfM and AirCanvas Pro pipeline"""
    
    print("\n" + "="*70)
    print("🚀 FULL INTEGRATION: SfM + AirCanvas Pro")
    print("="*70)
    
    # Step 1: Run SfM if needed
    sfm_dir = "outputs/my_reconstruction"
    
    if not Path(sfm_dir).exists():
        print("\n📸 Step 1: Running SfM reconstruction...")
        os.system("python fixed_main.py --image_dir data/test_sequence")
    
    # Step 2: Launch AirCanvas Pro
    print("\n🎨 Step 2: Launching AirCanvas Pro...")
    os.system("python aircanvas_pro.py")

if __name__ == "__main__":
    run_full_pipeline()
    