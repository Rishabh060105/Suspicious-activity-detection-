import os
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve, 
    average_precision_score, accuracy_score, f1_score
)

# Add parent directory to path to import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_model import SuspiciousActivityLSTM, ModelTrainer
from prepare_lstm_data import engineer_advanced_features

class TrainingConfig:
    """Configuration class for training parameters."""
    def __init__(self):
        # Data parameters
        self.batch_size = 32
        self.num_workers = 4
        self.pin_memory = True
        self.sequence_length = 30  # Fixed sequence length for LSTM
        
        # Model parameters
        self.hidden_size = 256  # Increased hidden size for better capacity
        self.num_layers = 2     # Two LSTM layers for better feature learning
        self.dropout = 0.3      # Keep dropout to prevent overfitting
        self.use_attention = True  # Enable attention mechanism
        self.num_attention_heads = 4  # Multi-head attention with 4 heads
        
        # Training parameters
        self.num_epochs = 100
        self.learning_rate = 3e-4  # Slightly higher learning rate
        self.weight_decay = 1e-4
        self.max_grad_norm = 1.0
        self.patience = 15  # Increased patience for more stable training
        
        # Paths
        self.features_dir = 'data/pose_features_processed_20250801_1900'  # Directory containing processed pose features
        self.output_dir = 'runs/exp_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        
        # Model parameters - will be set based on data
        # Input size after feature engineering: 102 features per timestep
        # (34 relative positions + 34 velocities + 34 accelerations)
        self.input_size = 102
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Initialize device - store as string for JSON serialization
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def device(self):
        """Get the torch device."""
        return torch.device(self._device)
    
    @device.setter
    def device(self, value):
        """Set the device from string or torch.device."""
        if isinstance(value, torch.device):
            self._device = str(value)
        else:
            self._device = str(torch.device(value))
    
    def save(self, path):
        """Save configuration to JSON file."""
        # Create a copy of the dict to avoid modifying the original
        config_dict = self.__dict__.copy()
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path):
        """Load configuration from JSON file."""
        config = cls()
        with open(path, 'r') as f:
            loaded_dict = json.load(f)
            # Update all attributes except device
            for key, value in loaded_dict.items():
                if key != '_device':
                    setattr(config, key, value)
        return config

def load_features(features_dir: str):
    """
    Load pre-extracted features and labels from the processed directory structure.
    
    The directory structure should be:
    features_dir/
    ├── train/
    │   ├── normal/
    │   └── suspicious/
    ├── val/
    │   ├── normal/
    │   └── suspicious/
    ├── test/
    │   ├── normal/
    │   └── suspicious/
    └── dataset_metadata.json
    """
    try:
        logging.info("Loading features and labels...")
        
        def load_split(split_dir):
            """Load features and labels for a single split (train/val/test)."""
            normal_dir = os.path.join(split_dir, 'normal')
            suspicious_dir = os.path.join(split_dir, 'suspicious')
            
            # Load normal samples (label 0)
            normal_samples = []
            for fname in os.listdir(normal_dir):
                if fname.endswith('.npy'):
                    path = os.path.join(normal_dir, fname)
                    data = np.load(path, allow_pickle=True)
                    # Ensure data is at least 2D (seq_len, features)
                    if len(data.shape) == 1:
                        data = np.expand_dims(data, 0)
                    normal_samples.append(data)
            
            # Load suspicious samples (label 1)
            suspicious_samples = []
            for fname in os.listdir(suspicious_dir):
                if fname.endswith('.npy'):
                    path = os.path.join(suspicious_dir, fname)
                    data = np.load(path, allow_pickle=True)
                    # Ensure data is at least 2D (seq_len, features)
                    if len(data.shape) == 1:
                        data = np.expand_dims(data, 0)
                    suspicious_samples.append(data)
            
            # Combine and create labels
            if normal_samples and suspicious_samples:
                X = np.vstack([np.stack(normal_samples), np.stack(suspicious_samples)])
                y = np.concatenate([
                    np.zeros(len(normal_samples)),  # Normal samples (0)
                    np.ones(len(suspicious_samples))  # Suspicious samples (1)
                ])
                return X, y
            elif normal_samples:
                return np.stack(normal_samples), np.zeros(len(normal_samples))
            elif suspicious_samples:
                return np.stack(suspicious_samples), np.ones(len(suspicious_samples))
            else:
                return np.array([]), np.array([])
        
        # Load each split
        X_train, y_train = load_split(os.path.join(features_dir, 'train'))
        X_val, y_val = load_split(os.path.join(features_dir, 'val'))
        X_test, y_test = load_split(os.path.join(features_dir, 'test'))
        
        # Log dataset statistics
        logging.info(f"Loaded {len(X_train)} training samples")
        logging.info(f"Loaded {len(X_val)} validation samples")
        logging.info(f"Loaded {len(X_test)} test samples")
        
        # Ensure features and labels have the same length
        assert len(X_train) == len(y_train), "Mismatch in training features and labels"
        assert len(X_val) == len(y_val), "Mismatch in validation features and labels"
        assert len(X_test) == len(y_test), "Mismatch in test features and labels"
        
        # Log class distribution
        logging.info(f"Training class distribution - Normal: {np.sum(y_train == 0)}, Suspicious: {np.sum(y_train == 1)}")
        logging.info(f"Validation class distribution - Normal: {np.sum(y_val == 0)}, Suspicious: {np.sum(y_val == 1)}")
        logging.info(f"Test class distribution - Normal: {np.sum(y_test == 0)}, Suspicious: {np.sum(y_test == 1)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    except Exception as e:
        logging.error(f"Error loading features: {e}")
        raise
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Train: {len(X_train)} samples, shape: {X_train.shape}")
        print(f"  Val: {len(X_val)} samples, shape: {X_val.shape}")
        print(f" Test: {len(X_test)} samples, shape: {X_test.shape}")
        
        # Add debug info for first sample
        if len(X_train) > 0:
            print("\nDebug - First training sample:")
            print(f"Type: {type(X_train[0])}")
            if hasattr(X_train[0], 'shape'):
                print(f"Shape: {X_train[0].shape}")
            print(f"Content: {X_train[0]}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    except Exception as e:
        print(f"Error loading features: {e}")
        import traceback
        traceback.print_exc()
        raise

class PoseSequenceDataset(Dataset):
    """Dataset class for loading and processing pose sequence data with advanced feature engineering."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 30):
        """
        Initialize the dataset.
        
        Args:
            features: Numpy array of shape (num_samples, seq_len, num_features)
                     where num_features is 51 (17 keypoints * 3 values)
            labels: Numpy array of shape (num_samples,)
            sequence_length: Fixed sequence length for LSTM input
        """
        self.features = features
        self.labels = labels.astype(np.float32)
        self.sequence_length = sequence_length
        
        # Validate shapes
        assert len(self.features) == len(self.labels), "Features and labels must have the same length"
        if len(self.features) > 0:
            assert len(self.features[0].shape) == 2, "Each sample should be 2D (seq_len, features)"
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence and label
        sequence = self.features[idx]  # Shape: (seq_len, 51)
        label = self.labels[idx]
        
        # Ensure sequence is numpy array
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence)
        
        # Reshape to (seq_len, 17, 3) for feature engineering
        seq_len = sequence.shape[0]
        keypoints = sequence.reshape(seq_len, 17, 3)  # (seq_len, 17, 3)
        
        # Apply advanced feature engineering
        enhanced_sequence = engineer_advanced_features(keypoints)  # (seq_len, 102)
        
        # Pad or trim sequence to fixed length
        if enhanced_sequence.shape[0] < self.sequence_length:
            # Pad with zeros if sequence is shorter
            pad_width = ((0, self.sequence_length - enhanced_sequence.shape[0]), (0, 0))
            enhanced_sequence = np.pad(enhanced_sequence, pad_width, mode='constant')
        else:
            # Trim sequence if longer
            enhanced_sequence = enhanced_sequence[:self.sequence_length]
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(enhanced_sequence)  # (seq_len, 102)
        label_tensor = torch.FloatTensor([label])  # Keep as [1,] tensor
        
        return sequence_tensor, label_tensor

def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray, 
                      config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        X_train: Training features of shape (num_samples, seq_len, num_features)
        y_train: Training labels of shape (num_samples,)
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        config: Training configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("\nCreating data loaders...")
    
    try:
        # Create datasets
        print("\nCreating PoseSequenceDatasets...")
        train_dataset = PoseSequenceDataset(X_train, y_train, config.sequence_length)
        val_dataset = PoseSequenceDataset(X_val, y_val, config.sequence_length)
        test_dataset = PoseSequenceDataset(X_test, y_test, config.sequence_length)
        
        # Calculate class weights for imbalanced dataset
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        sample_weights = class_weights[y_train.astype(int)]
        
        # Create samplers
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        print("\nCreating DataLoaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(config.batch_size, len(train_dataset)),
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True  # Drop last incomplete batch for stable training
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(config.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(config.batch_size, len(test_dataset)),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        # Verify loaders
        print("\nVerifying DataLoaders...")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Check first batch
        try:
            train_batch = next(iter(train_loader))
            print(f"\nFirst training batch:")
            print(f"  Features shape: {train_batch[0].shape} (batch_size, seq_len, num_features)")
            print(f"  Labels shape: {train_batch[1].shape} (batch_size, 1)")
            
            # Print class distribution in first batch
            unique, counts = np.unique(train_batch[1].numpy(), return_counts=True)
            print("  Class distribution in first batch:")
            for cls, cnt in zip(unique, counts):
                print(f"    Class {int(cls)}: {cnt} samples")
                
        except Exception as e:
            print(f"  Error checking first training batch: {e}")
            raise
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"\nError in create_dataloaders: {e}")
        print("Debug info:")
        print(f"  X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
        print(f"  y_train shape: {y_train.shape if hasattr(y_train, 'shape') else 'N/A'}")
        if 'train_dataset' in locals():
            print(f"  Train dataset size: {len(train_dataset)}")
        raise

def train_model(train_loader, val_loader, config):
    """Initializes and trains the LSTM model using the ModelTrainer."""
    print("\nInitializing model...")
    
    # Get input size from the first batch
    sample_batch = next(iter(train_loader))[0]
    if len(sample_batch.shape) == 3:  # (batch, seq_len, features)
        input_size = sample_batch.shape[2]
    else:
        input_size = sample_batch.shape[1] if len(sample_batch.shape) > 1 else sample_batch.shape[0]
    
    print(f"Input feature size: {input_size}")
    
    # Initialize model
    model = SuspiciousActivityLSTM(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_attention=config.use_attention,
        num_attention_heads=config.num_attention_heads
    )
    
    # Initialize trainer
    trainer = ModelTrainer(model, device=config.device)
    
    print(f"\nStarting training for up to {config.num_epochs} epochs...")
    
    # The ModelTrainer's own .train() method handles everything
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        early_stopping_rounds=config.patience,
        checkpoint_dir=config.checkpoint_dir
    )
    
    # The trainer automatically loads the best model at the end
    return trainer.model, history, {}  # Return empty dict for best_metrics as trainer handles it

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def evaluate_model(model, test_loader, config, history=None):
    """Evaluate the model on test set and save results."""
    print("\nEvaluating on test set...")
    
    # Initialize trainer for evaluation
    trainer = ModelTrainer(model, device=config.device)
    
    # Evaluate
    test_metrics = trainer.evaluate(test_loader, mode='test')
    
    # Print metrics
    print("\nTest Set Performance:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    
    # Save classification report
    report = classification_report(
        test_metrics['targets'],
        test_metrics['predictions'],
        target_names=['Normal', 'Suspicious'],
        output_dict=True
    )
    
    # Save metrics to file
    metrics_path = os.path.join(config.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        metrics_to_save = {
            'test_metrics': convert_numpy_types(test_metrics),
            'classification_report': convert_numpy_types(report)
        }
        json.dump(metrics_to_save, f, indent=4, ensure_ascii=False)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_metrics['targets'], test_metrics['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Suspicious'],
                yticklabels=['Normal', 'Suspicious'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(config.figures_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_metrics['targets'], test_metrics['probabilities'])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.figures_dir, 'roc_curve.png'))
    plt.close()
    
    # Plot training curves if history is provided
    if history:
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot metrics
        plt.subplot(1, 2, 2)
        plt.plot(history['val_accuracy'], label='Accuracy')
        plt.plot(history['val_f1'], label='F1 Score')
        plt.plot(history['val_precision'], label='Precision')
        plt.plot(history['val_recall'], label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.figures_dir, 'training_curves.png'))
        plt.close()
    
    return test_metrics

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize configuration
    config = TrainingConfig()
    print(f"Using device: {config.device}")
    
    # Save configuration
    config_path = os.path.join(config.output_dir, 'config.json')
    config.save(config_path)
    print(f"Configuration saved to {config_path}")
    
    try:
        # Load features and labels
        logging.info("Loading features and labels...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_features(config.features_dir)
        
        # Log dataset statistics and shapes
        logging.info(f"Training set: {len(X_train)} samples, shape: {X_train.shape}")
        logging.info(f"Validation set: {len(X_val)} samples, shape: {X_val.shape}")
        logging.info(f"Test set: {len(X_test)} samples, shape: {X_test.shape}")
        
        # Debug: Print first sample shape
        if len(X_train) > 0:
            logging.info(f"First training sample shape: {X_train[0].shape if hasattr(X_train[0], 'shape') else 'No shape attribute'}")
            logging.info(f"First training sample type: {type(X_train[0])}")
            logging.info(f"First training sample content: {X_train[0]}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test, config
        )
        
        logging.info("Starting training...")
        
        # Get input size from the first batch
        sample_batch = next(iter(train_loader))[0]
        if len(sample_batch.shape) == 3:  # (batch, seq_len, features)
            config.input_size = sample_batch.shape[2]
        else:
            config.input_size = sample_batch.shape[1] if len(sample_batch.shape) > 1 else sample_batch.shape[0]
        
        logging.info(f"Input feature size: {config.input_size}")
        
        # Train model
        model, history, best_metrics = train_model(train_loader, val_loader, config)
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, config, history)
        
        # Save final metrics (convert all NumPy types to native Python types)
        final_metrics = {
            'best_validation': convert_numpy_types(best_metrics),
            'test_metrics': convert_numpy_types(test_metrics),
            'training_history': convert_numpy_types(history)
        }
        
        metrics_path = os.path.join(config.output_dir, 'final_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4, ensure_ascii=False)
        print(f"\nFinal metrics saved to {metrics_path}")
        
        print("\nTraining completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()
