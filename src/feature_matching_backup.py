"""
Feature matching using FLANN and ratio test
Fixed for OpenCV compatibility
"""

import cv2
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureMatcher:
    """
    Match features between image pairs using FLANN and Lowe's ratio test
    """
    
    def __init__(self, ratio_threshold: float = 0.75):
        """
        Initialize feature matcher
        
        Args:
            ratio_threshold: Threshold for Lowe's ratio test
        """
        self.ratio_threshold = ratio_threshold
        
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        logger.info("Initialized FLANN matcher")
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Match features between two descriptor sets
        
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            
        Returns:
            List of matches
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            logger.warning("Empty descriptors provided")
            return []
        
        try:
            # Use k-NN matching with k=2
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)
            
            logger.info(f"Found {len(good_matches)} good matches out of {len(matches)}")
            return good_matches
            
        except Exception as e:
            logger.error(f"Error in feature matching: {e}")
            return []
    
    def filter_matches_with_ransac(self, kp1, kp2, matches, 
                                  reprojection_threshold=3.0):
        """
        Filter matches using RANSAC with fundamental matrix
        Compatible with multiple OpenCV versions
        
        Args:
            kp1: Keypoints in first image
            kp2: Keypoints in second image
            matches: Initial matches
            reprojection_threshold: Threshold for RANSAC
            
        Returns:
            Filtered matches and fundamental matrix
        """
        if len(matches) < 8:
            logger.warning(f"Not enough matches for RANSAC: {len(matches)}")
            return matches, None
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Try different OpenCV versions
        F = None
        mask = None
        
        # Method 1: OpenCV 4.5+ with keyword arguments
        try:
            F, mask = cv2.findFundamentalMat(
                pts1, pts2, 
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=reprojection_threshold,
                confidence=0.99
            )
        except:
            # Method 2: OpenCV 4.x with positional arguments
            try:
                F, mask = cv2.findFundamentalMat(
                    pts1, pts2,
                    cv2.FM_RANSAC,
                    reprojection_threshold,
                    0.99
                )
            except:
                # Method 3: OpenCV 3.x
                try:
                    F, mask = cv2.findFundamentalMat(
                        pts1, pts2,
                        cv2.FM_RANSAC,
                        reprojection_threshold
                    )
                except:
                    # Method 4: Fallback
                    try:
                        result = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
                        if isinstance(result, tuple):
                            F, mask = result[0], result[1]
                        else:
                            F, mask = result, None
                    except Exception as e:
                        logger.error(f"All RANSAC attempts failed: {e}")
                        return matches, None
        
        if F is None:
            logger.warning("Failed to compute fundamental matrix")
            return matches, None
        
        # Filter matches based on mask
        if mask is not None and len(mask) > 0:
            mask = mask.ravel().astype(bool)
            filtered_matches = [matches[i] for i in range(len(matches)) if mask[i]]
            logger.info(f"RANSAC kept {len(filtered_matches)} out of {len(matches)} matches")
            return filtered_matches, F
        
        return matches, F
    
    def visualize_matches(self, img1, img2, kp1, kp2, matches, output_path=None):
        """
        Visualize matches between two images
        
        Args:
            img1: First image
            img2: Second image
            kp1: Keypoints in first image
            kp2: Keypoints in second image
            matches: Matches to visualize
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        try:
            vis_image = cv2.drawMatches(
                img1, kp1, img2, kp2, matches[:100], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0)
            )
            
            if output_path:
                cv2.imwrite(output_path, vis_image)
                logger.info(f"Saved match visualization to {output_path}")
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error visualizing matches: {e}")
            return None