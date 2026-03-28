"""
Camera pose estimation using essential matrix decomposition
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import logging
from .utils import compute_fundamental_matrix, compute_essential_matrix, decompose_essential_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraEstimator:
    """
    Estimate camera poses from feature correspondences
    """
    
    def __init__(self, K: np.ndarray):
        """
        Initialize camera estimator with intrinsic parameters
        
        Args:
            K: Camera intrinsic matrix (3x3)
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)
        logger.info("Initialized CameraEstimator")
    
    def estimate_relative_pose(self, points1: np.ndarray, points2: np.ndarray,
                               matches: List) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Estimate relative camera pose between two views
        
        Args:
            points1: Points in first image
            points2: Points in second image
            matches: Matches between images
            
        Returns:
            Rotation matrix, translation vector, and inlier matches
        """
        if len(matches) < 8:
            logger.error("Not enough matches for pose estimation")
            return None, None, matches
        
        # Extract matched points
        pts1 = np.float32([points1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([points2[m.trainIdx].pt for m in matches])
        
        # Compute fundamental matrix
        F = compute_fundamental_matrix(pts1, pts2)
        
        # Compute essential matrix
        E = compute_essential_matrix(F, self.K)
        
        # Decompose essential matrix
        poses = decompose_essential_matrix(E)
        
        # Find the correct pose by checking cheirality (points in front of camera)
        best_pose = None
        best_inliers = 0
        best_points_3d = None
        best_R = None
        best_t = None
        
        for R, t in poses:
            # Build projection matrices
            P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = self.K @ np.hstack([R, t])
            
            # Triangulate points
            points_3d = []
            inliers = []
            
            for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
                # Triangulate point
                X = self._triangulate_point(P1, P2, pt1, pt2)
                
                # Check if point is in front of both cameras
                if X[2] > 0 and (R @ X + t.flatten())[2] > 0:
                    points_3d.append(X)
                    inliers.append(matches[i])
            
            if len(points_3d) > best_inliers:
                best_inliers = len(points_3d)
                best_pose = (R, t)
                best_points_3d = points_3d
                best_R, best_t = R, t
        
        if best_pose is None:
            logger.error("Failed to find valid camera pose")
            return None, None, matches
        
        logger.info(f"Found pose with {best_inliers} points in front of cameras")
        
        return best_R, best_t, [m for i, m in enumerate(matches) if i < best_inliers]
    
    def _triangulate_point(self, P1: np.ndarray, P2: np.ndarray,
                           pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
        """
        Triangulate a single 3D point using DLT
        """
        # Build constraint matrix
        A = np.zeros((4, 4))
        A[0] = pt1[0] * P1[2] - P1[0]
        A[1] = pt1[1] * P1[2] - P1[1]
        A[2] = pt2[0] * P2[2] - P2[0]
        A[3] = pt2[1] * P2[2] - P2[1]
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        
        return X[:3]
    
    def estimate_absolute_pose(self, points_2d: np.ndarray, points_3d: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate absolute camera pose using PnP (Perspective-n-Point)
        
        Args:
            points_2d: 2D points in image
            points_3d: Corresponding 3D points
            
        Returns:
            Rotation vector and translation vector
        """
        if len(points_2d) < 4:
            logger.error("Not enough points for PnP")
            return None
        
        # Use PnP with RANSAC
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, self.K, None,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99
        )
        
        if rvec is None or tvec is None:
            logger.error("PnP failed to estimate pose")
            return None
        
        logger.info(f"PnP pose estimation successful with {len(inliers)} inliers")
        
        return rvec, tvec