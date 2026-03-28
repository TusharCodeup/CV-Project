#!/usr/bin/env python3
"""
Complete dataset downloader and generator for SfM project
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import tarfile
import shutil

def create_test_sequence():
    """
    Create a high-quality test sequence with proper texture and viewpoint changes
    """
    output_dir = Path("data/test_sequence")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📸 Creating test sequence in {output_dir}")
    
    width, height = 800, 600
    num_images = 6
    
    for i in range(num_images):
        # Create blank image
        img = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add gradient background (for texture)
        for y in range(height):
            for x in range(width):
                img[y, x] = [
                    int(100 + x/width * 100),
                    int(100 + y/height * 100),
                    int(150 + (x+y)/(width+height) * 100)
                ]
        
        # Add grid pattern (creates features)
        grid_spacing = 50
        for x in range(0, width, grid_spacing):
            cv2.line(img, (x, 0), (x, height), (100, 100, 100), 1)
        for y in range(0, height, grid_spacing):
            cv2.line(img, (0, y), (width, y), (100, 100, 100), 1)
        
        # Simulate camera movement by shifting the main object
        angle = -20 + (i * 8)  # Gradual viewpoint change
        
        # Main building (cube-like structure)
        center_x = width // 2 + int(angle * 3)
        center_y = height // 2
        
        # Draw front face
        cv2.rectangle(img, 
                     (center_x - 120, center_y - 100),
                     (center_x + 120, center_y + 100),
                     (70, 130, 180), -1)  # Steel blue color
        
        # Draw side face (simulating 3D)
        cv2.rectangle(img,
                     (center_x + 100, center_y - 100 + int(angle/3)),
                     (center_x + 180, center_y + 100 + int(angle/3)),
                     (100, 100, 150), -1)
        
        # Add windows (good features)
        for wx in range(-80, 100, 40):
            for wy in range(-60, 80, 40):
                cv2.rectangle(img,
                             (center_x + wx, center_y + wy),
                             (center_x + wx + 25, center_y + wy + 25),
                             (200, 200, 50), -1)
        
        # Add random textured spots (SIFT features)
        for _ in range(300):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            size = np.random.randint(2, 8)
            color = np.random.randint(0, 255, 3).tolist()
            cv2.circle(img, (x, y), size, color, -1)
        
        # Add corners (distinct features)
        corners = [
            (center_x - 120, center_y - 100),
            (center_x + 120, center_y - 100),
            (center_x - 120, center_y + 100),
            (center_x + 120, center_y + 100),
            (center_x + 180, center_y - 100 + int(angle/3)),
            (center_x + 180, center_y + 100 + int(angle/3))
        ]
        
        for corner in corners:
            cv2.circle(img, corner, 8, (0, 0, 255), -1)
        
        # Add text to identify view
        cv2.putText(img, f"View {i+1}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save
        img_path = output_dir / f"view_{i+1:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        print(f"  ✅ Created: {img_path.name}")
    
    print(f"✅ Created {num_images} test images")
    return output_dir

def create_synthetic_cube_dataset():
    """
    Create a synthetic dataset of a rotating cube (classic SfM test)
    """
    output_dir = Path("data/cube_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎲 Creating cube dataset in {output_dir}")
    
    width, height = 800, 600
    num_images = 8
    
    for i in range(num_images):
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Rotation angle
        angle = (i * 45)  # 45-degree increments
        
        # Draw a 3D cube (simulated)
        center_x = width // 2
        center_y = height // 2
        
        # Cube parameters
        size = 150
        offset_x = int(np.sin(np.radians(angle)) * 30)
        offset_y = int(np.cos(np.radians(angle)) * 20)
        
        # Front face
        cv2.rectangle(img,
                     (center_x - size//2 + offset_x, center_y - size//2 + offset_y),
                     (center_x + size//2 + offset_x, center_y + size//2 + offset_y),
                     (255, 100, 100), -1)
        
        # Top face
        pts = np.array([
            [center_x - size//2 + offset_x, center_y - size//2 + offset_y],
            [center_x + size//2 + offset_x, center_y - size//2 + offset_y],
            [center_x + size//2 + offset_x + 30, center_y - size//2 + offset_y - 30],
            [center_x - size//2 + offset_x + 30, center_y - size//2 + offset_y - 30]
        ], np.int32)
        cv2.fillPoly(img, [pts], (100, 100, 255))
        
        # Right face
        pts = np.array([
            [center_x + size//2 + offset_x, center_y - size//2 + offset_y],
            [center_x + size//2 + offset_x, center_y + size//2 + offset_y],
            [center_x + size//2 + offset_x + 30, center_y + size//2 + offset_y - 30],
            [center_x + size//2 + offset_x + 30, center_y - size//2 + offset_y - 30]
        ], np.int32)
        cv2.fillPoly(img, [pts], (100, 255, 100))
        
        # Add texture (checkerboard pattern)
        for x in range(-size//2, size//2, 30):
            for y in range(-size//2, size//2, 30):
                if (x + y) % 60 == 0:
                    cv2.rectangle(img,
                                 (center_x + x + offset_x, center_y + y + offset_y),
                                 (center_x + x + 25 + offset_x, center_y + y + 25 + offset_y),
                                 (0, 0, 0), -1)
        
        # Save
        img_path = output_dir / f"cube_{i+1:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        print(f"  ✅ Created: {img_path.name}")
    
    print(f"✅ Created {num_images} cube images")
    return output_dir

def create_realistic_scene():
    """
    Create a realistic indoor/outdoor scene simulation
    """
    output_dir = Path("data/realistic_scene")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🏠 Creating realistic scene dataset in {output_dir}")
    
    width, height = 1024, 768
    num_images = 5
    
    for i in range(num_images):
        img = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Sky gradient
        for y in range(height//3):
            img[y, :] = [135, 206, 235]  # Sky blue
        
        # Ground
        img[height//2:, :] = [34, 139, 34]  # Forest green
        
        # Simulate camera movement
        camera_x = (i - 2) * 50
        
        # House
        house_x = width//2 + camera_x
        house_y = height//2
        
        # House body
        cv2.rectangle(img,
                     (house_x - 100, house_y - 50),
                     (house_x + 100, house_y + 100),
                     (139, 69, 19), -1)  # Brown
        
        # Roof
        pts = np.array([
            [house_x - 120, house_y - 50],
            [house_x + 120, house_y - 50],
            [house_x, house_y - 120]
        ], np.int32)
        cv2.fillPoly(img, [pts], (160, 82, 45))
        
        # Door
        cv2.rectangle(img,
                     (house_x - 30, house_y + 20),
                     (house_x + 30, house_y + 100),
                     (101, 67, 33), -1)
        
        # Windows
        cv2.rectangle(img,
                     (house_x - 70, house_y - 20),
                     (house_x - 20, house_y + 20),
                     (173, 216, 230), -1)
        cv2.rectangle(img,
                     (house_x + 20, house_y - 20),
                     (house_x + 70, house_y + 20),
                     (173, 216, 230), -1)
        
        # Add trees
        for tree_x in [house_x - 200, house_x + 200]:
            cv2.circle(img, (tree_x, house_y + 50), 40, (0, 100, 0), -1)
            cv2.rectangle(img,
                         (tree_x - 15, house_y + 50),
                         (tree_x + 15, house_y + 100),
                         (101, 67, 33), -1)
        
        # Add clouds (texture)
        for _ in range(10):
            cloud_x = np.random.randint(50, width-50)
            cloud_y = np.random.randint(50, height//3)
            cv2.circle(img, (cloud_x, cloud_y), 30, (255, 255, 255), -1)
            cv2.circle(img, (cloud_x+20, cloud_y-10), 25, (255, 255, 255), -1)
            cv2.circle(img, (cloud_x-20, cloud_y-10), 25, (255, 255, 255), -1)
        
        # Add noise for texture
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Save
        img_path = output_dir / f"scene_{i+1:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        print(f"  ✅ Created: {img_path.name}")
    
    print(f"✅ Created {num_images} realistic scene images")
    return output_dir

def list_available_datasets():
    """
    List all available datasets
    """
    data_path = Path("data")
    datasets = []
    
    for item in data_path.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            images = list(item.glob("*.jpg")) + list(item.glob("*.png"))
            if len(images) > 0:
                datasets.append((item.name, len(images)))
    
    return datasets

def check_dataset(dataset_path):
    """
    Check if dataset is valid for SfM
    """
    images = list(Path(dataset_path).glob("*.jpg")) + list(Path(dataset_path).glob("*.png"))
    
    if len(images) < 2:
        print(f"❌ Dataset {dataset_path} has less than 2 images")
        return False
    
    print(f"\n📊 Dataset: {dataset_path}")
    print(f"   Images: {len(images)}")
    
    # Check if all images can be loaded
    valid_count = 0
    for img_path in images[:3]:  # Check first 3 images
        img = cv2.imread(str(img_path))
        if img is not None:
            valid_count += 1
            print(f"   ✅ {img_path.name}: {img.shape}")
        else:
            print(f"   ❌ {img_path.name}: Failed to load")
    
    if valid_count == 0:
        print("   ❌ No valid images found!")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SfM Dataset Generator")
    parser.add_argument("--type", choices=["test", "cube", "realistic", "all"],
                       default="all", help="Dataset type to create")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 SfM Dataset Generator")
    print("="*60)
    
    if args.type in ["test", "all"]:
        print("\n📸 Creating test sequence...")
        create_test_sequence()
    
    if args.type in ["cube", "all"]:
        print("\n🎲 Creating cube dataset...")
        create_synthetic_cube_dataset()
    
    if args.type in ["realistic", "all"]:
        print("\n🏠 Creating realistic scene...")
        create_realistic_scene()
    
    # List all available datasets
    print("\n" + "="*60)
    print("📁 Available Datasets:")
    print("="*60)
    
    datasets = list_available_datasets()
    for name, count in datasets:
        print(f"   • {name}: {count} images")
        check_dataset(Path("data") / name)
    
    print("\n" + "="*60)
    print("✅ Dataset generation complete!")
    print("\nTo run SfM on a dataset:")
    print("   python main.py --image_dir data/test_sequence --output_dir outputs/test")
    print("="*60)