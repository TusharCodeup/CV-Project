"""
3D visualization utilities using matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    """
    Visualize 3D reconstruction results
    """
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def visualize_point_cloud(self, points_3d: List[np.ndarray],
                              camera_positions: Optional[List[np.ndarray]] = None,
                              colors: Optional[np.ndarray] = None,
                              title: str = "3D Point Cloud",
                              output_path: Optional[str] = None):
        """
        Visualize 3D point cloud with camera positions
        
        Args:
            points_3d: List of 3D points
            camera_positions: Optional camera center positions
            colors: Optional colors for points
            title: Plot title
            output_path: Optional path to save visualization
        """
        # Create figure
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Convert points to array
        points = np.array(points_3d)
        
        # Plot points
        if colors is not None:
            scatter = self.ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                      c=colors, cmap='viridis', s=1, alpha=0.6)
            plt.colorbar(scatter, label='Color')
        else:
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c='blue', s=1, alpha=0.6)
        
        # Plot camera positions
        if camera_positions is not None:
            cam_pos = np.array(camera_positions)
            self.ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
                           c='red', s=50, marker='^', label='Cameras')
            
            # Plot camera viewing directions
            for pos in cam_pos:
                self.ax.quiver(pos[0], pos[1], pos[2],
                              pos[0]/10, pos[1]/10, pos[2]/10,
                              length=0.5, color='red', alpha=0.5)
        
        # Set labels and title
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.max([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]) / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if camera_positions:
            self.ax.legend()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_camera_trajectory(self, camera_positions: List[np.ndarray],
                                    title: str = "Camera Trajectory",
                                    output_path: Optional[str] = None):
        """
        Visualize camera trajectory
        
        Args:
            camera_positions: List of camera center positions
            title: Plot title
            output_path: Optional path to save visualization
        """
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        positions = np.array(camera_positions)
        
        # Plot camera positions
        self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       c='red', s=50, marker='o')
        
        # Plot trajectory
        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    'b-', linewidth=2, alpha=0.7)
        
        # Add labels
        for i, pos in enumerate(positions):
            self.ax.text(pos[0], pos[1], pos[2], f'Cam{i}', fontsize=10)
        
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title(title)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved trajectory to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_reprojection_error(self, errors: List[float],
                               title: str = "Reprojection Error Distribution",
                               output_path: Optional[str] = None):
        """
        Plot reprojection error histogram
        
        Args:
            errors: List of reprojection errors
            title: Plot title
            output_path: Optional path to save visualization
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        self.ax.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        self.ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
        self.ax.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.2f}')
        
        self.ax.set_xlabel('Reprojection Error (pixels)')
        self.ax.set_ylabel('Frequency')
        self.ax.set_title(title)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved error plot to {output_path}")
        else:
            plt.show()
        
        plt.close()