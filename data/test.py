#!/usr/bin/env python3
"""
Test script to verify datasets and basic functionality
"""

import sys
import cv2
import numpy as np
from pathlib import Path

def test_dataset(dataset_path):
    """
    Test if dataset is ready for SfM
    """
    print(f"\n{'='*50}")
    print(f"Testing dataset: {dataset_path}")
    print(f"{'='*50}")
    
    # Find all images
    images = list(Path(dataset_path).glob("*.jpg")) + list(Path(dataset_path).glob("*.png"))
    
    if len(images) < 2:
        print(f"❌ Error: Need at least 2 images, found {len(images)}")
        return False
    
    print(f"✅ Found {len(images)} images")
    
    # Try loading images
    loaded_images = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            loaded_images.append(img)
            print(f"   ✅ {img_path.name}: {img.shape}")
        else:
            print(f"   ❌ {img_path.name}: Failed to load")
    
    if len(loaded_images) < 2:
        print("❌ Failed to load enough images")
        return False
    
    # Check SIFT availability
    try:
        sift = cv2.SIFT_create()
        print("✅ SIFT available")
        
        # Test feature extraction on first image
        gray = cv2.cvtColor(loaded_images[0], cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        print(f"✅ Feature extraction working: {len(kp)} features found")
        
    except Exception as e:
        print(f"❌ SIFT issue: {e}")
        print("   Try: pip install opencv-contrib-python")
        return False
    
    print(f"\n✅ Dataset {dataset_path} is ready for SfM!")
    return True

def test_all_datasets():
    """
    Test all available datasets
    """
    data_path = Path("data")
    
    # Find all dataset directories
    datasets = []
    for item in data_path.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            images = list(item.glob("*.jpg")) + list(item.glob("*.png"))
            if len(images) > 0:
                datasets.append(item)
    
    if not datasets:
        print("❌ No datasets found! Run dataset generator first:")
        print("   python data/download_real_dataset.py")
        return False
    
    results = []
    for dataset in datasets:
        result = test_dataset(dataset)
        results.append((dataset.name, result))
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for name, result in results:
        status = "✅ READY" if result else "❌ NOT READY"
        print(f"{status}: {name}")
    
    return any(result for _, result in results)

if __name__ == "__main__":
    print("\n🎯 SfM Dataset Tester")
    print("="*50)
    
    if len(sys.argv) > 1:
        # Test specific dataset
        dataset_path = sys.argv[1]
        test_dataset(dataset_path)
    else:
        # Test all datasets
        test_all_datasets()
    
    print("\n💡 Next steps:")
    print("1. Run SfM pipeline: python main.py --image_dir data/test_sequence")
    print("2. View results: Check outputs/ folder")
    print("3. Submit project: Follow GitHub submission guidelines")