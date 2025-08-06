#!/usr/bin/env python3
"""
Prepare LSTM Training Data from Raw Keypoint Sequences

This script organizes raw keypoint sequences into train/val/test splits
and creates the necessary metadata for LSTM training.
"""

import os
import shutil
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from typing import Tuple, List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prepare_lstm_data.log')
    ]
)
logger = logging.getLogger(__name__)

def engineer_advanced_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Enhance raw keypoint data with advanced kinematic and postural features.
    
    Args:
        keypoints: Input keypoints of shape (seq_len, 17, 3) where last dim is (x, y, confidence)
        
    Returns:
        Enhanced features of shape (seq_len, 102)
    """
    seq_len = keypoints.shape[0]
    enhanced_features = []
    
    # Calculate velocities (from frame 1 to 29)
    velocities = np.diff(keypoints[:, :, :2], axis=0, prepend=np.zeros((1, 17, 2)))
    
    # Calculate accelerations
    accelerations = np.diff(velocities, axis=0, prepend=np.zeros((1, 17, 2)))
    
    for i in range(seq_len):
        frame_kps = keypoints[i]  # (17, 3)
        
        # Hip center as the reference point for normalization
        left_hip = frame_kps[11, :2]  # Left hip keypoint
        right_hip = frame_kps[12, :2]  # Right hip keypoint
        hip_center = (left_hip + right_hip) / 2
        
        # 1. Normalized, relative keypoint positions (34 features)
        relative_kps = (frame_kps[:, :2] - hip_center).flatten()
        
        # 2. Keypoint velocities (34 features)
        frame_velocities = velocities[i].flatten()
        
        # 3. Keypoint accelerations (34 features)
        frame_accelerations = accelerations[i].flatten()
        
        # Combine all features for this frame
        frame_features = np.concatenate([
            relative_kps,    # 34 features
            frame_velocities,  # 34 features
            frame_accelerations  # 34 features
        ])  # Total: 102 features
        
        enhanced_features.append(frame_features)
    
    return np.array(enhanced_features, dtype=np.float32)


class LSTMDatasetPreparer:
    """Prepares LSTM training data from raw keypoint sequences."""
    
    def __init__(self, raw_data_dir: str, output_dir: str, 
                 test_size: float = 0.2, val_size: float = 0.2,
                 random_state: int = 42):
        """Initialize the dataset preparer.
        
        Args:
            raw_data_dir: Directory containing raw keypoint sequences.
            output_dir: Base directory to save processed dataset.
            test_size: Fraction of data to use for testing.
            val_size: Fraction of training data to use for validation.
            random_state: Random seed for reproducibility.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Create output directories
        self.splits = ['train', 'val', 'test']
        self.classes = ['normal', 'suspicious']
        
        # Setup output directory structure
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create the necessary output directory structure."""
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split in self.splits:
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Create class subdirectories
            for cls in self.classes:
                (split_dir / cls).mkdir(exist_ok=True, parents=True)
    
    def _get_video_paths(self) -> Tuple[Dict[str, List[Path]], Dict[str, int]]:
        """Get paths to all video keypoint files.
        
        Returns:
            Tuple of (class_to_paths, class_counts) where:
            - class_to_paths: Maps class names to lists of keypoint file paths
            - class_counts: Maps class names to their respective counts
        """
        class_to_paths = {cls: [] for cls in self.classes}
        suspicious_keywords = ['fight', 'assault', 'shooting', 'stealing', 'vandalism', 
                             'robbery', 'abuse', 'arrest', 'arson', 'shoplifting',
                             'explosion', 'suspicious', 'throwing', 'punching', 'kicking',
                             'struggling', 'chasing', 'hitting', 'attacking', 'threatening']
        
        # Find all .npy files in the raw directory
        for npy_file in self.raw_data_dir.rglob('*.npy'):
            # Convert filename to lowercase for case-insensitive matching
            filename = npy_file.name.lower()
            
            # Check for suspicious activities in filename
            is_suspicious = any(keyword in filename for keyword in suspicious_keywords)
            
            if is_suspicious:
                class_to_paths['suspicious'].append(npy_file)
                logger.debug(f"Classified as suspicious: {npy_file}")
            else:
                class_to_paths['normal'].append(npy_file)
                logger.debug(f"Classified as normal: {npy_file}")
        
        # Log class distribution
        class_counts = {cls: len(paths) for cls, paths in class_to_paths.items()}
        logger.info(f"Found {sum(class_counts.values())} total sequences")
        for cls, count in class_counts.items():
            logger.info(f"  {cls}: {count} sequences")
            
        return class_to_paths, class_counts
    
    def _save_split_info(self, split_data: Dict[str, List[Tuple[Path, int]]]) -> None:
        """Save information about the dataset splits.
        
        Args:
            split_data: Dictionary mapping split names to lists of (path, label) tuples.
        """
        split_info = {}
        
        for split, samples in split_data.items():
            split_info[split] = {
                'num_samples': len(samples),
                'class_distribution': {
                    cls: sum(1 for _, label in samples if label == i)
                    for i, cls in enumerate(self.classes)
                },
                'samples': [
                    {
                        'path': str(path),  # Store full source path
                        'label': int(label),
                        'class': self.classes[label]
                    }
                    for path, label in samples
                ]
            }
        
        # Save metadata
        metadata = {
            'dataset_info': {
                'name': 'suspicious_activity_lstm',
                'num_classes': len(self.classes),
                'class_names': self.classes,
                'splits': list(split_info.keys()),
                'total_samples': sum(len(samples) for samples in split_data.values())
            },
            'splits': split_info,
            'input_shape': {
                'sequence_length': 30,  # Fixed number of frames
                'num_features': 51      # 17 keypoints * 3 (x, y, confidence)
            },
            'processing_info': {
                'test_size': self.test_size,
                'val_size': self.val_size,
                'random_state': self.random_state
            }
        }
        
        # Save to file
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved dataset metadata to {metadata_path}")
    
    def prepare_dataset(self) -> None:
        """Prepare the dataset by splitting and organizing the data."""
        # Get all video paths by class
        class_to_paths, class_counts = self._get_video_paths()
        
        # Initialize data structures
        all_paths = []
        all_labels = []
        
        # Assign numerical labels and collect all samples
        for label, cls in enumerate(self.classes):
            paths = class_to_paths[cls]
            all_paths.extend(paths)
            all_labels.extend([label] * len(paths))
        
        # Split into train+val and test sets
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_paths, all_labels,
            test_size=self.test_size,
            stratify=all_labels,
            random_state=self.random_state
        )
        
        # Split train into train and validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=self.val_size / (1 - self.test_size),  # Adjust for initial split
            stratify=train_val_labels,
            random_state=self.random_state
        )
        
        # Create split data structure
        split_data = {
            'train': list(zip(train_paths, train_labels)),
            'val': list(zip(val_paths, val_labels)),
            'test': list(zip(test_paths, test_labels))
        }
        
        # Copy files to their respective directories
        for split, samples in split_data.items():
            logger.info(f"Processing {split} split with {len(samples)} samples")
            
            for src_path, label in samples:
                # Determine destination path
                cls = self.classes[label]
                dst_dir = self.output_dir / split / cls
                dst_path = dst_dir / src_path.name
                
                # Copy the file
                shutil.copy2(src_path, dst_path)
                
                # Verify the file was copied correctly
                if not dst_path.exists():
                    logger.error(f"Failed to copy {src_path} to {dst_path}")
                    continue
                
                # Verify the data can be loaded
                try:
                    data = np.load(dst_path)
                    if data.size == 0:
                        logger.error(f"Empty file: {dst_path}")
                        dst_path.unlink()  # Remove invalid file
                except Exception as e:
                    logger.error(f"Error loading {dst_path}: {e}")
                    dst_path.unlink()  # Remove corrupted file
        
        # Save dataset metadata
        self._save_split_info(split_data)
        
        logger.info("\nDataset preparation complete!")
        logger.info(f"Dataset saved to: {self.output_dir}")
        logger.info("\nClass distribution:")
        
        # Print final class distribution
        for split in self.splits:
            split_dir = self.output_dir / split
            total = 0
            dist = {}
            
            for cls in self.classes:
                count = len(list((split_dir / cls).glob('*.npy')))
                dist[cls] = count
                total += count
            
            logger.info(f"{split.upper()} ({total} samples): {dist}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare LSTM training data from raw keypoint sequences')
    parser.add_argument('--raw-dir', type=str, required=True,
                       help='Directory containing raw keypoint sequences')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save processed dataset')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Fraction of training data to use for validation')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create and run dataset preparer
    preparer = LSTMDatasetPreparer(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    preparer.prepare_dataset()


if __name__ == "__main__":
    main()
