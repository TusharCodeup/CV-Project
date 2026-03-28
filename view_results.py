"""
Quick viewer for 3D reconstruction results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path

# Load point cloud
ply_path = Path("outputs/my_reconstruction/point_cloud.ply")
if ply_path.exists():
    # Parse PLY file
    with open(ply_path, 'r') as f:
        lines = f.readlines()
    
    # Find where vertices start
    vertex_start = 0
    for i, line in enumerate(lines):
        if "end_header" in line:
            vertex_start = i + 1
            break
    
    # Read vertices
    points = []
    for line in lines[vertex_start:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    points = np.array(points)
    
    # Load statistics
    stats_path = Path("outputs/my_reconstruction/reconstruction_stats.json")
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    
    # Create visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=points[:, 2], cmap='viridis', s=20, alpha=0.7)
    
    ax.set_xlabel('X (units)')
    ax.set_ylabel('Y (units)')
    ax.set_zlabel('Z (units)')
    ax.set_title(f'3D Reconstruction - {len(points)} Points')
    
    plt.colorbar(scatter, label='Depth')
    
    plt.tight_layout()
    plt.savefig('outputs/my_reconstruction/viewer_3d.png', dpi=150)
    plt.show()
    
    print(f"\n📊 Statistics:")
    print(f"   Points: {len(points)}")
    print(f"   X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"   Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"   Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    
else:
    print("No point cloud found. Run the pipeline first.")