"""
Threshold Calibration for Suspicious Activity Detection

This script helps find the optimal classification threshold for the LSTM model
by evaluating performance metrics across different threshold values.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_curve, 
    precision_recall_curve, average_precision_score, auc, confusion_matrix, classification_report
)
import seaborn as sns
from tqdm import tqdm
import argparse
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import custom modules
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from lstm_model import SuspiciousActivityLSTM
from train_lstm import TrainingConfig

# Default configuration
DEFAULT_CONFIG = {
    'features_dir': 'data/pose_features_processed',
    'model_path': 'runs/exp_best/checkpoints/best_model.pt',
    'output_dir': 'threshold_results',
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'threshold_range': (0.01, 0.99),
    'threshold_step': 0.01,
    'model_config': {
        'input_size': 102,  # 17 keypoints * 3 (x, y, confidence) * 2 (if using velocities)
        'hidden_size': 256,  # Must match the checkpoint's hidden size
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': False,
        'use_attention': True,  # Enable attention as per the checkpoint
        'num_attention_heads': 4  # Number of attention heads in the checkpoint
    }
}

@dataclass
class ThresholdMetrics:
    """Container for threshold evaluation metrics."""
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float = None
    tpr: float = None
    roc_auc: float = None
    pr_auc: float = None
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'threshold': self.threshold,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'fpr': self.fpr,
            'tpr': self.tpr,
            'roc_auc': self.roc_auc,
            'pr_auc': self.pr_auc
        }

class ThresholdCalibrator:
    """Class for finding optimal classification threshold."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """Initialize with a trained model."""
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        logger.info(f"Initialized ThresholdCalibrator on device: {device}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_config: Optional[Dict] = None, device: str = 'cpu') -> 'ThresholdCalibrator':
        """Load a model and config from a checkpoint file robustly."""
        import inspect
        
        logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get the config from the checkpoint, use provided config as fallback
        saved_config = checkpoint.get('config', {})
        if model_config:
            saved_config.update(model_config)
        
        # Get the list of expected arguments from the model's __init__ method
        model_signature = inspect.signature(SuspiciousActivityLSTM.__init__)
        expected_args = list(model_signature.parameters.keys())
        
        # Filter the loaded config to only include arguments the model class expects
        filtered_config = {
            key: value for key, value in saved_config.items() 
            if key in expected_args and key != 'self'
        }
        
        logger.info(f"Initializing model with config: {filtered_config}")
        model = SuspiciousActivityLSTM(**filtered_config)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return cls(model, device)
    
    def predict_proba(self, dataloader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions on a dataset."""
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for features, labels in tqdm(dataloader, desc="Generating predictions"):
                features = features.to(self.device)
                outputs = self.model(features)
                probs = torch.sigmoid(outputs).squeeze()
                
                y_true.extend(labels.cpu().numpy().flatten())
                y_scores.extend(probs.cpu().numpy().flatten())
        
        return np.array(y_true), np.array(y_scores)
    
    def evaluate_thresholds(self, y_true: np.ndarray, y_scores: np.ndarray, 
                          threshold_range: Tuple[float, float] = (0.01, 0.99),
                          step: float = 0.01) -> Dict[float, ThresholdMetrics]:
        """Evaluate metrics across a range of thresholds."""
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        results = {}
        
        for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
            y_pred = (y_scores >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results[threshold] = ThresholdMetrics(
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=f1
            )
        
        # Calculate ROC and PR curve metrics
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Get decision thresholds for ROC curve
        _, _, roc_thresholds = roc_curve(y_true, y_scores)
        
        # For each threshold in our grid, find the closest ROC/PR points
        for threshold, metrics in results.items():
            # For ROC curve metrics, find the closest threshold in ROC space
            if len(roc_thresholds) > 0:
                idx = np.argmin(np.abs(roc_thresholds - threshold))
                metrics.fpr = fpr[min(idx, len(fpr)-1)]
                metrics.tpr = tpr[min(idx, len(tpr)-1)]
            else:
                metrics.fpr = 0.0
                metrics.tpr = 0.0
                
            metrics.roc_auc = roc_auc
            metrics.pr_auc = pr_auc
        
        return results
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray, 
                             threshold_range: Tuple[float, float] = (0.01, 0.99),
                             step: float = 0.01) -> Tuple[float, Dict]:
        """Find the threshold that maximizes F1 score."""
        results = self.evaluate_thresholds(y_true, y_scores, threshold_range, step)
        best_metrics = max(results.values(), key=lambda x: x.f1)
        
        return best_metrics.threshold, {k: v.to_dict() for k, v in results.items()}
    
    @staticmethod
    def plot_metrics(metrics_dict: Union[Dict[float, ThresholdMetrics], Dict[float, Dict]], output_path: str = None) -> None:
        """Plot precision, recall, and F1 across thresholds.
        
        Args:
            metrics_dict: Dictionary mapping thresholds to either ThresholdMetrics objects or dictionaries
            output_path: Optional path to save the plot
        """
        thresholds = np.array(list(metrics_dict.keys()))
        
        # Convert to list of dictionaries if needed
        metrics = []
        for m in metrics_dict.values():
            if isinstance(m, dict):
                metrics.append(m)
            else:
                metrics.append(m.to_dict() if hasattr(m, 'to_dict') else m.__dict__)
        
        # Prepare data
        data = {
            'Threshold': thresholds,
            'Precision': [m.get('precision', 0) for m in metrics],
            'Recall': [m.get('recall', 0) for m in metrics],
            'F1': [m.get('f1', 0) for m in metrics]
        }
        
        # Helper function to get metric value safely
        def get_metric(metric_obj, key):
            if isinstance(metric_obj, dict):
                return metric_obj.get(key, 0.0)
            return getattr(metric_obj, key, 0.0)
            
        # Find best threshold (max F1)
        best_threshold = max(metrics_dict.keys(), 
                           key=lambda t: get_metric(metrics_dict[t], 'f1'))
        best_metrics = metrics_dict[best_threshold]
        
        # Get metric values for display
        best_f1 = get_metric(best_metrics, 'f1')
        best_precision = get_metric(best_metrics, 'precision')
        best_recall = get_metric(best_metrics, 'recall')
        
        # Plot metrics
        plt.figure(figsize=(12, 8))
        
        # Plot precision, recall, F1
        plt.plot(thresholds, data['Precision'], 'b-', 
                label=f'Precision (max={best_precision:.3f} @ {best_threshold:.3f})')
        plt.plot(thresholds, data['Recall'], 'g-', 
                label=f'Recall (max={best_recall:.3f} @ {best_threshold:.3f})')
        plt.plot(thresholds, data['F1'], 'r-', 
                label=f'F1 (max={best_f1:.3f} @ {best_threshold:.3f})')
        
        # Plot best threshold
        plt.axvline(x=best_threshold, color='k', linestyle='--', 
                   label=f'Best Threshold: {best_threshold:.3f} (F1={best_f1:.3f})')
        
        plt.title('Metrics vs. Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_roc_pr_curves(y_true: np.ndarray, y_scores: np.ndarray, output_path: str = None) -> None:
        """Plot ROC and Precision-Recall curves."""
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot ROC
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")
        
        # Plot PR
        ax2.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.2f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved ROC/PR curves to {output_path}")
        else:
            plt.show()
        plt.close()

def load_validation_data(features_dir: str, batch_size: int = 32) -> torch.utils.data.DataLoader:
    """
    Load validation data from a directory structure with 'normal' and 'suspicious' subdirectories.
    
    Expected directory structure:
    features_dir/
    ├── val/
    │   ├── normal/     # Contains .npy files for normal activities
    │   └── suspicious/ # Contains .npy files for suspicious activities
    """
    try:
        val_dir = os.path.join(features_dir, 'val')
        normal_dir = os.path.join(val_dir, 'normal')
        suspicious_dir = os.path.join(val_dir, 'suspicious')
        
        # Check directories exist
        for d in [val_dir, normal_dir, suspicious_dir]:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Directory not found: {d}")
        
        features_list = []
        labels_list = []
        
        # Load normal samples (label 0)
        normal_count = 0
        for file in os.listdir(normal_dir):
            if file.endswith('.npy'):
                file_path = os.path.join(normal_dir, file)
                try:
                    # Load features from .npy file
                    features = np.load(file_path, allow_pickle=True)
                    
                    # Handle both dictionary and array formats
                    if isinstance(features, dict):
                        if 'features' in features:
                            features = features['features']
                        else:
                            logger.warning(f"No 'features' key in {file}, skipping...")
                            continue
                    
                    # Ensure features is a numpy array
                    if not isinstance(features, np.ndarray):
                        logger.warning(f"Unexpected data type in {file}, skipping...")
                        continue
                    
                    features_list.append(features)
                    labels_list.append(0)  # 0 for normal
                    normal_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error loading {file}: {str(e)}")
        
        # Load suspicious samples (label 1)
        suspicious_count = 0
        for file in os.listdir(suspicious_dir):
            if file.endswith('.npy'):
                file_path = os.path.join(suspicious_dir, file)
                try:
                    # Load features from .npy file
                    features = np.load(file_path, allow_pickle=True)
                    
                    # Handle both dictionary and array formats
                    if isinstance(features, dict):
                        if 'features' in features:
                            features = features['features']
                        else:
                            logger.warning(f"No 'features' key in {file}, skipping...")
                            continue
                    
                    # Ensure features is a numpy array
                    if not isinstance(features, np.ndarray):
                        logger.warning(f"Unexpected data type in {file}, skipping...")
                        continue
                    
                    features_list.append(features)
                    labels_list.append(1)  # 1 for suspicious
                    suspicious_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error loading {file}: {str(e)}")
        
        if not features_list:
            raise ValueError(f"No valid feature files found in {normal_dir} or {suspicious_dir}")
        
        logger.info(f"Loaded {normal_count} normal and {suspicious_count} suspicious samples")
        
        # Convert to tensors
        def engineer_features(keypoints):
            """Convert raw keypoints to 102-dimensional features."""
            seq_len = keypoints.shape[0]
            features = []
            
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
                
                # Combine all features for this frame (102 features total)
                frame_features = np.concatenate([
                    relative_kps,      # 34 features
                    frame_velocities,  # 34 features
                    frame_accelerations # 34 features
                ])
                features.append(frame_features)
            
            return np.array(features, dtype=np.float32)
        
        # Process features to ensure correct dimensions
        processed_features = []
        for feat in features_list:
            # Ensure features are in the correct shape (seq_len, 51)
            if len(feat.shape) == 1:
                # If features are 1D, reshape assuming they're already flattened (seq_len*51)
                seq_len = len(feat) // 51
                if seq_len * 51 != len(feat):
                    logger.warning(f"Feature length {len(feat)} not divisible by 51, skipping")
                    continue
                feat = feat.reshape(seq_len, 17, 3)  # Reshape to (seq_len, 17, 3)
            elif len(feat.shape) == 2:
                # If features are 2D, reshape to (seq_len, 17, 3)
                if feat.shape[1] == 51:  # Flattened version
                    feat = feat.reshape(-1, 17, 3)
                elif feat.shape[1] == 17:  # Already in keypoint format
                    feat = np.dstack([feat, np.ones((feat.shape[0], 17, 1))])  # Add confidence=1
                else:
                    logger.warning(f"Unexpected feature shape {feat.shape}")
                    continue
            elif len(feat.shape) == 3 and feat.shape[2] == 3:  # Already in (seq_len, 17, 3)
                pass
            else:
                logger.warning(f"Unexpected feature shape {feat.shape}, skipping")
                continue
            
            # Apply feature engineering to get 102-D features
            feat_engineered = engineer_features(feat)
            
            # Pad or truncate sequence to fixed length if needed
            seq_len = 30  # Match the sequence length used in training
            if len(feat_engineered) < seq_len:
                # Pad with zeros
                pad = np.zeros((seq_len - len(feat_engineered), 102))
                feat_engineered = np.vstack([feat_engineered, pad])
            elif len(feat_engineered) > seq_len:
                # Truncate to seq_len
                feat_engineered = feat_engineered[:seq_len]
                
            processed_features.append(feat_engineered)
        
        if not processed_features:
            raise ValueError("No valid features after processing")
            
        features_array = np.array(processed_features)
        logger.info(f"Processed features shape: {features_array.shape}")
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features_array)
        labels_tensor = torch.FloatTensor(np.array(labels_list)).unsqueeze(1)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Created DataLoader with {len(dataset)} total samples")
        return dataloader
    
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        raise

def save_results(results: Dict, output_dir: str) -> None:
    """Save threshold calibration results to disk."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, 'threshold_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save optimal threshold
        best_threshold = max(results['threshold_metrics'].values(), 
                           key=lambda x: x['f1'])['threshold']
        
        threshold_path = os.path.join(output_dir, 'optimal_threshold.txt')
        with open(threshold_path, 'w') as f:
            f.write(f"{best_threshold:.4f}")
        logger.info(f"Optimal threshold ({best_threshold:.4f}) saved to {threshold_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Threshold Calibration for Suspicious Activity Detection')
    parser.add_argument('--features-dir', type=str, default=DEFAULT_CONFIG['features_dir'],
                       help='Directory containing validation features')
    parser.add_argument('--model-path', type=str, default=DEFAULT_CONFIG['model_path'],
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                       help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'],
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default=DEFAULT_CONFIG['device'],
                       help='Device to run on (cuda, mps, cpu)')
    parser.add_argument('--min-threshold', type=float, default=DEFAULT_CONFIG['threshold_range'][0],
                       help='Minimum threshold value to evaluate')
    parser.add_argument('--max-threshold', type=float, default=DEFAULT_CONFIG['threshold_range'][1],
                       help='Maximum threshold value to evaluate')
    parser.add_argument('--threshold-step', type=float, default=0.01,
                       help='Step size for threshold evaluation')
    
    return parser.parse_args()

def main():
    """Main function for threshold calibration."""
    args = parse_args()
    
    try:
        logger.info("Starting threshold calibration...")
        logger.info(f"Using device: {args.device}")
        
        # Load data
        dataloader = load_validation_data(args.features_dir, args.batch_size)
        
        # Initialize calibrator with trained model
        calibrator = ThresholdCalibrator.from_checkpoint(
            args.model_path,
            model_config=DEFAULT_CONFIG['model_config'],
            device=args.device
        )
        
        # Get model predictions
        logger.info("Generating predictions...")
        y_true, y_scores = calibrator.predict_proba(dataloader)
        
        # Find optimal threshold
        logger.info("Finding optimal threshold...")
        threshold_range = (args.min_threshold, args.max_threshold)
        best_threshold, threshold_metrics = calibrator.find_optimal_threshold(
            y_true, y_scores, threshold_range, args.threshold_step
        )
        
        # Generate plots
        logger.info("Generating plots...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save metrics plot
        metrics_plot_path = os.path.join(args.output_dir, 'threshold_metrics.png')
        calibrator.plot_metrics(threshold_metrics, metrics_plot_path)
        
        # Save ROC/PR curves
        curves_plot_path = os.path.join(args.output_dir, 'roc_pr_curves.png')
        calibrator.plot_roc_pr_curves(y_true, y_scores, curves_plot_path)
        
        # Save results
        results = {
            'best_threshold': best_threshold,
            'threshold_metrics': {str(k): v for k, v in threshold_metrics.items()},
            'config': {
                'features_dir': args.features_dir,
                'model_path': args.model_path,
                'threshold_range': [args.min_threshold, args.max_threshold],
                'threshold_step': args.threshold_step,
                'device': args.device
            }
        }
        save_results(results, args.output_dir)
        
        logger.info(f"Threshold calibration complete. Optimal threshold: {best_threshold:.4f}")
        logger.info(f"Results saved to: {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        logger.error(f"Error during threshold calibration: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
