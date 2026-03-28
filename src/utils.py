"""
Utility functions for the SfM pipeline
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict
import os

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize points for DLT algorithm
    Points are normalized to have zero mean and sqrt(2) average distance from origin
    """
    if points.shape[0] == 0:
        return points, np.eye(3)
    
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    
    # Shift points to have zero mean
    shifted_points = points - centroid
    
    # Calculate average distance from origin
    distances = np.sqrt(np.sum(shifted_points ** 2, axis=1))
    avg_distance = np.mean(distances)
    
    # Scale factor to make average distance sqrt(2)
    scale = np.sqrt(2) / avg_distance
    
    # Create normalization matrix
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    
    # Normalize points
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    normalized_points = (T @ points_homo.T).T
    
    return normalized_points[:, :2], T

def compute_fundamental_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Compute fundamental matrix using normalized 8-point algorithm
    """
    # Normalize points
    pts1_norm, T1 = normalize_points(points1)
    pts2_norm, T2 = normalize_points(points2)
    
    # Build the constraint matrix A
    n_points = len(pts1_norm)
    A = np.zeros((n_points, 9))
    
    for i in range(n_points):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve for F using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank 2 constraint
    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[-1] = 0
    F = Uf @ np.diag(Sf) @ Vtf
    
    # Denormalize
    F = T2.T @ F @ T1
    
    return F / np.linalg.norm(F)

def compute_essential_matrix(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute essential matrix from fundamental matrix and camera intrinsics
    """
    E = K.T @ F @ K
    # Enforce essential matrix properties
    U, S, Vt = np.linalg.svd(E)
    S = np.diag([1, 1, 0])
    E = U @ S @ Vt
    return E / np.linalg.norm(E)

def decompose_essential_matrix(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Decompose essential matrix into possible rotation and translation combinations
    """
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper rotation matrix
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    
    # Rotation matrices
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Translation vector (third column of U)
    t = U[:, 2].reshape(3, 1)
    
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]

def triangulate_point(P1: np.ndarray, P2: np.ndarray, 
                      pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    """
    Triangulate a single 3D point from two camera views using DLT
    """
    # Build the constraint matrix A for DLT
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