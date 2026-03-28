"""
Feature extraction using SIFT (Scale-Invariant Feature Transform)
"""

import cv2
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extract and manage features from images using SIFT
    """
    
    def __init__(self, n_features: int = 5000, contrast_threshold: float = 0.04):
        """
        Initialize SIFT feature extractor
        
        Args:
            n_features: Maximum number of features to detect
            contrast_threshold: Threshold for filtering weak features
        """
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=10,
            sigma=1.6
        )
        logger.info(f"Initialized SIFT with {n_features} max features")
    
    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract SIFT features and descriptors from an image
        
        Args:
            image: Input image (RGB or grayscale)
            
        Returns:
            keypoints: List of keypoints
            descriptors: Feature descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect and compute features
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        logger.info(f"Extracted {len(keypoints)} features")
        
        return keypoints, descriptors
    
    def filter_features_by_response(self, keypoints: List[cv2.KeyPoint], 
                                    descriptors: np.ndarray, 
                                    keep_ratio: float = 0.5) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Keep only the strongest features based on response
        
        Args:
            keypoints: List of keypoints
            descriptors: Feature descriptors
            keep_ratio: Ratio of features to keep
            
        Returns:
            Filtered keypoints and descriptors
        """
        if len(keypoints) == 0:
            return keypoints, descriptors
        
        # Sort by response strength
        responses = [kp.response for kp in keypoints]
        sorted_indices = np.argsort(responses)[::-1]
        
        keep_count = int(len(keypoints) * keep_ratio)
        keep_indices = sorted_indices[:keep_count]
        
        filtered_keypoints = [keypoints[i] for i in keep_indices]
        filtered_descriptors = descriptors[keep_indices]
        
        logger.info(f"Filtered to {len(filtered_keypoints)} strongest features")
        
        return filtered_keypoints, filtered_descriptors
    
    def visualize_features(self, image: np.ndarray, keypoints: List[cv2.KeyPoint],
                          output_path: str = None) -> np.ndarray:
        """
        Draw keypoints on the image for visualization
        
        Args:
            image: Input image
            keypoints: List of keypoints to draw
            output_path: Optional path to save the visualization
            
        Returns:
            Image with keypoints drawn
        """
        vis_image = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            color=(0, 255, 0)
        )
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Saved feature visualization to {output_path}")
        
        return vis_image