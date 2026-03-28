"""
3D point triangulation from multiple views
"""

import numpy as np
from typing import List, Tuple
import logging
from .utils import triangulate_point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Triangulator:
    """
    Triangulate 3D points from multiple camera views
    """
    
    def __init__(self):
        self.points_3d = []
        self.point_observations = []
        logger.info("Initialized Triangulator")
    
    def triangulate_pair(self, P1: np.ndarray, P2: np.ndarray,
                         points1: List, points2: List,
                         matches: List) -> List[np.ndarray]:
        """
        Triangulate points from a pair of views
        
        Args:
            P1: Projection matrix for first camera
            P2: Projection matrix for second camera
            points1: Points in first image
            points2: Points in second image
            matches: Matches between images
            
        Returns:
            List of 3D points
        """
        points_3d = []
        
        for match in matches:
            pt1 = np.array([points1[match.queryIdx].pt[0], points1[match.queryIdx].pt[1], 1.0])
            pt2 = np.array([points2[match.trainIdx].pt[0], points2[match.trainIdx].pt[1], 1.0])
            
            # Triangulate point
            X = triangulate_point(P1, P2, pt1, pt2)
            points_3d.append(X)
        
        logger.info(f"Triangulated {len(points_3d)} 3D points")
        
        return points_3d
    
    def triangulate_multi_view(self, projection_matrices: List[np.ndarray],
                               observations: List[List[Tuple[int, np.ndarray]]],
                               min_views: int = 2) -> List[np.ndarray]:
        """
        Triangulate points from multiple views using least squares
        
        Args:
            projection_matrices: List of camera projection matrices
            observations: List of observations for each 3D point
            min_views: Minimum number of views required
            
        Returns:
            List of 3D points
        """
        points_3d = []
        
        for point_obs in observations:
            if len(point_obs) < min_views:
                continue
            
            # Build constraint matrix for all observations
            A = []
            
            for view_idx, pt in point_obs:
                P = projection_matrices[view_idx]
                x, y = pt[0], pt[1]
                
                A.append([x * P[2, 0] - P[0, 0],
                         x * P[2, 1] - P[0, 1],
                         x * P[2, 2] - P[0, 2],
                         x * P[2, 3] - P[0, 3]])
                
                A.append([y * P[2, 0] - P[1, 0],
                         y * P[2, 1] - P[1, 1],
                         y * P[2, 2] - P[1, 2],
                         y * P[2, 3] - P[1, 3]])
            
            A = np.array(A)
            
            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]
            
            points_3d.append(X[:3])
        
        logger.info(f"Triangulated {len(points_3d)} points from multiple views")
        
        return points_3d
    
    def filter_points_by_depth(self, points_3d: List[np.ndarray],
                               min_depth: float = 0.1,
                               max_depth: float = 100.0) -> List[np.ndarray]:
        """
        Filter 3D points by depth range
        
        Args:
            points_3d: List of 3D points
            min_depth: Minimum depth
            max_depth: Maximum depth
            
        Returns:
            Filtered list of points
        """
        filtered_points = []
        
        for point in points_3d:
            depth = np.linalg.norm(point)
            if min_depth < depth < max_depth:
                filtered_points.append(point)
        
        logger.info(f"Filtered from {len(points_3d)} to {len(filtered_points)} points by depth")
        
        return filtered_points
    
    def compute_reprojection_error(self, points_3d: np.ndarray,
                                  P: np.ndarray,
                                  points_2d: np.ndarray) -> float:
        """
        Compute reprojection error for a set of 3D points
        
        Args:
            points_3d: 3D points
            P: Camera projection matrix
            points_2d: Observed 2D points
            
        Returns:
            Mean reprojection error
        """
        # Project 3D points
        points_3d_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        projected = (P @ points_3d_homo.T).T
        
        # Convert to Euclidean coordinates
        projected = projected[:, :2] / projected[:, 2:3]
        
        # Compute errors
        errors = np.linalg.norm(projected - points_2d, axis=1)
        
        mean_error = np.mean(errors)
        logger.info(f"Mean reprojection error: {mean_error:.2f} pixels")
        
        return mean_error