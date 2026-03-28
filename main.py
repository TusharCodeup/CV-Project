#!/usr/bin/env python3
"""
Fixed Structure from Motion Pipeline - Compatible with all OpenCV versions
"""

import os
import sys
import argparse
import logging
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from src
from src.feature_extraction import FeatureExtractor
from src.feature_matching import FeatureMatcher
from src.camera_estimation import CameraEstimator
from src.triangulation import Triangulator
from src.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedStructureFromMotion:
    """
    Fixed Structure from Motion pipeline - Works with OpenCV 4.12.0
    """
    
    def __init__(self, intrinsic_matrix=None, image_size=None):
        """
        Initialize SfM pipeline
        """
        if intrinsic_matrix is None and image_size is not None:
            w, h = image_size
            fx = fy = max(w, h)
            cx, cy = w / 2, h / 2
            self.K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
            logger.info(f"Using estimated intrinsic matrix")
        elif intrinsic_matrix is not None:
            self.K = intrinsic_matrix
        else:
            raise ValueError("Either intrinsic_matrix or image_size must be provided")
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(n_features=2000)
        self.feature_matcher = FeatureMatcher(ratio_threshold=0.75)
        self.camera_estimator = CameraEstimator(self.K)
        self.triangulator = Triangulator()
        self.visualizer = Visualizer()
        
        # Storage
        self.images = []
        self.image_paths = []
        self.keypoints = []
        self.descriptors = []
        self.camera_poses = []
        self.points_3d = []
        
        logger.info("Initialized Fixed Structure from Motion pipeline")
    
    def load_images(self, image_dir):
        """Load images from directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        image_files = sorted(image_files)
        
        if len(image_files) < 2:
            raise ValueError(f"Need at least 2 images, found {len(image_files)}")
        
        logger.info(f"Loading {len(image_files)} images from {image_dir}")
        
        for img_path in tqdm(image_files, desc="Loading images"):
            img = cv2.imread(str(img_path))
            if img is not None:
                self.images.append(img)
                self.image_paths.append(str(img_path))
                logger.info(f"Loaded: {img_path.name}")
            else:
                logger.warning(f"Failed to load: {img_path}")
        
        logger.info(f"Successfully loaded {len(self.images)} images")
    
    def extract_features(self):
        """Extract features from all images"""
        logger.info("Extracting features from images...")
        
        for i, img in enumerate(tqdm(self.images, desc="Extracting features")):
            kp, desc = self.feature_extractor.extract_features(img)
            if len(kp) > 0:
                kp, desc = self.feature_extractor.filter_features_by_response(kp, desc, keep_ratio=0.7)
            
            self.keypoints.append(kp)
            self.descriptors.append(desc)
            
            logger.info(f"Image {i}: {len(kp)} features extracted")
    
    def match_features(self):
        """Match features between consecutive images"""
        logger.info("Matching features between images...")
        
        self.matches = []
        self.match_inliers = []
        
        for i in range(len(self.images) - 1):
            logger.info(f"Matching image {i} and {i+1}")
            
            # Match features
            matches = self.feature_matcher.match_features(
                self.descriptors[i], self.descriptors[i+1]
            )
            
            if len(matches) > 0:
                # Filter with RANSAC
                filtered_matches, F = self.feature_matcher.filter_matches_with_ransac(
                    self.keypoints[i], self.keypoints[i+1], matches
                )
                self.matches.append(filtered_matches)
                self.match_inliers.append(F is not None)
                logger.info(f"  Found {len(filtered_matches)} good matches")
            else:
                self.matches.append([])
                self.match_inliers.append(False)
                logger.warning(f"  No matches found")
    
    def estimate_camera_poses(self):
        """Estimate camera poses for all images"""
        logger.info("Estimating camera poses...")
        
        # Start with first camera at origin
        self.camera_poses = [(np.eye(3), np.zeros((3, 1)))]
        
        # Estimate relative poses for consecutive pairs
        for i in range(len(self.images) - 1):
            if i >= len(self.matches) or not self.match_inliers[i] or len(self.matches[i]) < 8:
                logger.warning(f"Skipping pair {i}-{i+1} - insufficient matches")
                continue
            
            # Get matches
            matches = self.matches[i]
            
            # Extract matched points
            pts_i = np.array([self.keypoints[i][m.queryIdx].pt for m in matches])
            pts_ip1 = np.array([self.keypoints[i+1][m.trainIdx].pt for m in matches])
            
            # Estimate relative pose
            R_rel, t_rel, filtered_matches = self.camera_estimator.estimate_relative_pose(
                self.keypoints[i], self.keypoints[i+1], matches
            )
            
            if R_rel is not None:
                # Update matches with inliers
                self.matches[i] = filtered_matches
                
                # Compute absolute pose for next camera
                R_prev, t_prev = self.camera_poses[i]
                R_abs = R_rel @ R_prev
                t_abs = R_rel @ t_prev + t_rel
                
                self.camera_poses.append((R_abs, t_abs))
                logger.info(f"Camera {i+1} pose estimated")
            else:
                logger.error(f"Failed to estimate pose for camera {i+1}")
                break
    
    def triangulate_points(self):
        """Triangulate 3D points from all views"""
        logger.info("Triangulating 3D points...")
        
        # Build projection matrices
        P_matrices = []
        for R, t in self.camera_poses:
            P = self.K @ np.hstack([R, t])
            P_matrices.append(P)
        
        # Triangulate points from each consecutive pair
        all_points_3d = []
        
        for i in range(len(self.images) - 1):
            if i >= len(self.matches) or len(self.matches[i]) < 10:
                continue
            
            matches = self.matches[i]
            
            # Triangulate
            try:
                points_3d = self.triangulator.triangulate_pair(
                    P_matrices[i], P_matrices[i+1],
                    self.keypoints[i], self.keypoints[i+1],
                    matches
                )
                all_points_3d.extend(points_3d)
                logger.info(f"Pair {i}-{i+1}: {len(points_3d)} points triangulated")
            except Exception as e:
                logger.warning(f"Triangulation failed for pair {i}-{i+1}: {e}")
        
        # Filter points
        if len(all_points_3d) > 0:
            self.points_3d = self.triangulator.filter_points_by_depth(all_points_3d, min_depth=0.5, max_depth=50.0)
        else:
            self.points_3d = []
        
        logger.info(f"Total triangulated points: {len(self.points_3d)}")
    
    def save_results(self, output_dir="outputs"):
        """Save reconstruction results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save point cloud as PLY
        if len(self.points_3d) > 0:
            ply_path = os.path.join(output_dir, "point_cloud.ply")
            self.save_ply(ply_path, self.points_3d)
            logger.info(f"Saved point cloud to {ply_path}")
        
        # Save camera poses
        poses_data = []
        for i, (R, t) in enumerate(self.camera_poses):
            poses_data.append({
                "camera_id": i,
                "rotation": R.tolist(),
                "translation": t.flatten().tolist()
            })
        
        poses_path = os.path.join(output_dir, "camera_poses.json")
        with open(poses_path, 'w') as f:
            json.dump(poses_data, f, indent=2)
        logger.info(f"Saved camera poses to {poses_path}")
        
        # Save reconstruction statistics
        stats = {
            "num_images": len(self.images),
            "num_cameras": len(self.camera_poses),
            "num_points_3d": len(self.points_3d),
            "reconstruction_complete": len(self.points_3d) > 0
        }
        
        stats_path = os.path.join(output_dir, "reconstruction_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")
    
    def save_ply(self, filename, points_3d):
        """Save point cloud in PLY format"""
        points = np.array(points_3d)
        
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for point in points:
                r = np.random.randint(100, 255)
                g = np.random.randint(100, 255)
                b = np.random.randint(100, 255)
                f.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
    
    def visualize_results(self, output_dir="outputs"):
        """Visualize reconstruction results"""
        if len(self.points_3d) == 0:
            logger.warning("No 3D points to visualize")
            return
        
        # Get camera positions
        camera_positions = [t.flatten() for _, t in self.camera_poses]
        
        # Visualize point cloud
        self.visualizer.visualize_point_cloud(
            self.points_3d,
            camera_positions,
            title="Structure from Motion - 3D Reconstruction",
            output_path=os.path.join(output_dir, "3d_reconstruction.png")
        )
    
    def run_pipeline(self, image_dir, output_dir="outputs"):
        """Run the complete SfM pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Fixed Structure from Motion Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load images
            self.load_images(image_dir)
            
            # Step 2: Extract features
            self.extract_features()
            
            # Step 3: Match features
            self.match_features()
            
            # Step 4: Estimate camera poses
            self.estimate_camera_poses()
            
            # Step 5: Triangulate points
            self.triangulate_points()
            
            # Step 6: Save results
            self.save_results(output_dir)
            
            # Step 7: Visualize
            self.visualize_results(output_dir)
            
            logger.info("=" * 60)
            logger.info("Structure from Motion Pipeline Completed!")
            logger.info(f"Results saved to: {output_dir}")
            logger.info("=" * 60)
            
            return self.points_3d, self.camera_poses
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return [], []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fixed Structure from Motion Pipeline")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        logger.error(f"Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Get image size from first image
    image_files = list(Path(args.image_dir).glob('*.jpg')) + list(Path(args.image_dir).glob('*.png'))
    if len(image_files) > 0:
        img = cv2.imread(str(image_files[0]))
        if img is not None:
            height, width = img.shape[:2]
            logger.info(f"Detected image size: {width}x{height}")
        else:
            width, height = 800, 600
    else:
        width, height = 800, 600
    
    # Initialize and run SfM
    sfm = FixedStructureFromMotion(image_size=(width, height))
    points_3d, camera_poses = sfm.run_pipeline(args.image_dir, args.output_dir)
    
    logger.info(f"Reconstruction complete. Generated {len(points_3d)} 3D points.")

if __name__ == "__main__":
    main()