#!/usr/bin/env python3
"""
Temporal Feature Generation Script for Suspicious Activity Detection

This script generates temporal features from video files for training the Suspicious Activity Detection model.
It processes each video to extract exactly 30 frames and generates a 60-dimensional feature vector for each frame,
resulting in a (30, 60) array per video.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Import dataset builder and feature extractor
from dataset_training import SPHARDatasetBuilder, YOLOv8FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generate_temporal_features.log')
    ]
)
logger = logging.getLogger(__name__)

class TemporalFeatureExtractor:
    """Extracts temporal features from videos using YOLOv8."""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: Optional[str] = None):
        """Initialize the feature extractor.
        
        Args:
            model_path: Path to YOLOv8 model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = YOLOv8FeatureExtractor(model_path=model_path, device=self.device)
        self.feature_size = self.feature_extractor.num_features  # 60 features per frame
        self.sequence_length = 30  # Fixed number of frames per video
        
        logger.info(f"Initialized TemporalFeatureExtractor on {self.device}")
        logger.info(f"Feature size: {self.feature_size} (per frame)")
        logger.info(f"Sequence length: {self.sequence_length} frames")
    
    def process_video(self, video_path: str) -> Optional[np.ndarray]:
        """Process a single video file to extract temporal features.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            NumPy array of shape (sequence_length, feature_size) or None if processing fails
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error(f"Video has no frames: {video_path}")
                cap.release()
                return None
            
            # Calculate frame indices to sample
            frame_indices = self._get_frame_indices(total_frames)
            
            # Initialize features array
            features = np.zeros((self.sequence_length, self.feature_size), dtype=np.float32)
            
            # Process each frame
            for i, frame_idx in enumerate(frame_indices):
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Could not read frame {frame_idx} from {video_path}")
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract features
                frame_features = self.feature_extractor.extract_features(frame_rgb)
                
                # Reshape to (feature_size,)
                if frame_features.size == self.feature_size:
                    features[i] = frame_features
                else:
                    logger.warning(f"Unexpected feature size: {frame_features.size}, expected {self.feature_size}")
            
            cap.release()
            return features
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}", exc_info=True)
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            return None
    
    def _get_frame_indices(self, total_frames: int) -> List[int]:
        """Get frame indices to sample from the video.
        
        Args:
            total_frames: Total number of frames in the video
            
        Returns:
            List of frame indices to sample
        """
        if total_frames <= self.sequence_length:
            # If video is shorter than sequence_length, repeat the last frame
            indices = list(range(total_frames))
            indices += [indices[-1]] * (self.sequence_length - len(indices))
            return indices[:self.sequence_length]
        else:
            # Sample frames evenly
            return np.linspace(0, total_frames - 1, self.sequence_length, dtype=int).tolist()

def process_dataset_split(split_name: str, video_paths: List[str], labels: List[int],
                         extractor: TemporalFeatureExtractor, output_dir: Path) -> bool:
    """Process a dataset split and save features to disk.
    
    Args:
        split_name: Name of the split ('train', 'val', 'test')
        video_paths: List of video file paths
        labels: List of corresponding labels
        extractor: Feature extractor instance
        output_dir: Directory to save features
        
    Returns:
        True if processing was successful, False otherwise
    """
    logger.info(f"Processing {split_name} split with {len(video_paths)} videos")
    
    # Create output directory
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize arrays
    all_features = []
    all_labels = []
    
    # Process each video
    for i, (video_path, label) in enumerate(tqdm(zip(video_paths, labels), total=len(video_paths), desc=f"Processing {split_name}")):
        features = extractor.process_video(video_path)
        if features is not None:
            all_features.append(features)
            all_labels.append(label)
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(video_paths)} {split_name} videos")
    
    if not all_features:
        logger.error(f"No features were extracted for {split_name} split")
        return False
    
    # Convert to numpy arrays
    features_array = np.array(all_features, dtype=np.float32)  # (n_samples, sequence_length, feature_size)
    labels_array = np.array(all_labels, dtype=np.int32)
    
    # Save features and labels
    features_path = output_dir / f"{split_name}_features.npy"
    labels_path = output_dir / f"{split_name}_labels.npy"
    
    # Ensure features are float32 and labels are int64
    features_array = features_array.astype(np.float32)
    labels_array = labels_array.astype(np.int64)
    
    np.save(features_path, features_array)
    np.save(labels_path, labels_array)
    
    logger.info(f"Saved {len(features_array)} {split_name} samples to {features_path}")
    logger.info(f"Feature shape: {features_array.shape}")
    logger.info(f"Label distribution: {np.bincount(labels_array)}")
    
    return True

def main():
    """Main function to generate temporal features from videos."""
    parser = argparse.ArgumentParser(description='Generate temporal features from videos')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing the SPHAR dataset')
    parser.add_argument('--output-dir', type=str, default='features_final',
                       help='Directory to save the generated features')
    parser.add_argument('--model-path', type=str, default='yolov8n.pt',
                       help='Path to YOLOv8 model weights')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run the model on (cuda/cpu), auto-detected if not specified')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset builder
    try:
        dataset_builder = SPHARDatasetBuilder(args.data_dir)
        splits = dataset_builder.build_dataset(test_size=0.2, val_size=0.2, random_state=42)
    except Exception as e:
        logger.error(f"Failed to initialize dataset builder: {str(e)}")
        return 1
    
    # Initialize feature extractor
    try:
        extractor = TemporalFeatureExtractor(model_path=args.model_path, device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize feature extractor: {str(e)}")
        return 1
    
    # Process each split
    success = True
    for split_name in ['train', 'val', 'test']:
        if split_name in splits:
            video_paths, labels = splits[split_name]
            if not process_dataset_split(split_name, video_paths, labels, extractor, output_dir):
                success = False
                logger.error(f"Failed to process {split_name} split")
    
    if success:
        logger.info("Feature extraction completed successfully!")
        logger.info(f"Features saved to: {output_dir.absolute()}")
        return 0
    else:
        logger.error("Feature extraction completed with errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
