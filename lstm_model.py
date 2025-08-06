import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    This loss function is designed to address class imbalance by down-weighting
    well-classified examples (pt > 0.5) and focusing on hard, misclassified examples.
    
    Args:
        alpha: Weighting factor (0-1) for the positive class. Default: 0.25
        gamma: Focusing parameter (γ ≥ 0). Higher values increase the effect of
               modulating factor (1-pt)^γ. Default: 2.0
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # Calculate binary cross entropy loss without reduction
        bce_loss = self.bce_loss(inputs, targets)
        
        # Calculate p_t and the focal loss
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss

# Set up logging
logger = logging.getLogger(__name__)

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism for better gradient flow.
    """
    def __init__(self, hidden_size: int):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5  # 1/sqrt(d_k)
        
        # Learnable parameters
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply scaled dot-product attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            context: Context vector (batch_size, hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.query(x)  # (batch_size, seq_len, hidden_size)
        K = self.key(x)    # (batch_size, seq_len, hidden_size)
        V = self.value(x)  # (batch_size, seq_len, hidden_size)
        
        # Calculate attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (batch_size, seq_len, seq_len)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights, V)  # (batch_size, seq_len, hidden_size)
        
        # Average over sequence length
        context = context.mean(dim=1)  # (batch_size, hidden_size)
        
        return context, attention_weights.mean(dim=1)  # Average attention weights across heads

class SuspiciousActivityLSTM(nn.Module):
    """
    LSTM model for binary sequence classification of pose sequences.
    Handles variable-length sequences with proper padding and masking.
    """

    def __init__(self, 
                 input_size: int = 51,  # 17 keypoints * 3 (x, y, confidence)
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.3,
                 use_attention: bool = False,
                 num_attention_heads: int = 0):
        """
        Initialize the LSTM model for pose sequence classification.
        
        Args:
            input_size: Number of input features per timestep (default: 51 for 17 keypoints * 3)
            hidden_size: Number of LSTM units (default: 128)
            num_layers: Number of LSTM layers (default: 1)
            dropout: Dropout probability (default: 0.3)
            use_attention: Whether to use attention mechanism (default: False)
            num_attention_heads: Number of attention heads if use_attention is True (default: 0)
        """
        super(SuspiciousActivityLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout  # Store dropout rate as instance variable
        self.bidirectional = False  # Store bidirectional flag as instance variable
        self.use_attention = use_attention
        
        # Input dropout for regularization
        self.input_dropout = nn.Dropout(dropout * 0.5)  # Lighter dropout on input
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0  # Only apply dropout between LSTM layers if num_layers > 1
        )
        
        # Optional attention layer
        if use_attention and num_attention_heads > 0:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                batch_first=True
            )
        else:
            self.attention = None
        
        # Classifier head - outputs raw logits (no sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Output raw logits
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                elif 'attention' in name and len(param.shape) > 1:
                    nn.init.xavier_uniform_(param.data)
                elif 'classifier' in name and len(param.shape) > 1:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Initialize forget gate bias to 1 for LSTM
                if 'lstm' in name:
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size,) with class probabilities
        """
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Expected input with 3 dimensions (batch, seq_len, features), got {x.dim()} dimensions")
            
        batch_size, seq_len, input_size = x.size()
        
        if input_size != self.input_size:
            raise ValueError(f"Input feature size mismatch: expected {self.input_size}, got {input_size}")
        
        # Log input shape once for debugging
        if not hasattr(self, '_logged_input_shape'):
            logger.info(f"Model input shape: {tuple(x.shape)}")
            self._logged_input_shape = True
        
        # Apply input dropout
        x = self.input_dropout(x)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Apply attention if enabled
        if self.attention is not None:
            # Use self-attention over the sequence
            attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Use the last timestep's output
            context = attn_output[:, -1, :]
        else:
            # Default: use the last hidden state
            context = lstm_out[:, -1, :]
        
        # Classify - output shape will be [batch_size, 1]
        output = self.classifier(context)
        
        # Return output with shape [batch_size, 1] to match target shape
        return output

class ModelTrainer:
    """
    Enhanced trainer class for sequence classification with improved training stability.
    Handles class imbalance, learning rate scheduling, and early stopping.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 class_weights: Optional[torch.Tensor] = None,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 max_grad_norm: float = 1.0,
                 patience: int = 7):
        """
        Initialize the ModelTrainer.
        
        Args:
            model: The PyTorch model to train (should be an instance of SuspiciousActivityLSTM)
            device: Device to run the model on ('cpu' or 'cuda')
            class_weights: Optional tensor of shape [num_classes] containing class weights.
                         If None, will use inverse class frequency as weights.
            learning_rate: Initial learning rate (default: 1e-3)
            weight_decay: Weight decay for L2 regularization (default: 1e-4)
            max_grad_norm: Maximum gradient norm for gradient clipping (default: 1.0)
            patience: Number of epochs to wait before early stopping (default: 7)
        """
        self.model = model
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        
        # Initialize class weights and loss function
        self.class_weights = class_weights
        self.criterion = None
        self.update_loss_function()
        
        # Initialize optimizer with weight decay (L2 regularization)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True  # Use AMSGrad variant of Adam for better convergence
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,  # Reduce learning rate by half when plateauing
            patience=3,  # Number of epochs with no improvement after which learning rate will be reduced
            min_lr=1e-6  # Minimum learning rate
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
        # Initialize history with all required keys
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }

    def update_loss_function(self) -> None:
        """
        Update the loss function with current class weights.
        Uses Focal Loss for better handling of class imbalance.
        """
        if self.class_weights is not None:
            # For binary classification, we use Focal Loss with class weights
            # Focal Loss helps with class imbalance by reducing the loss for well-classified examples
            self.criterion = FocalLoss(
                alpha=self.class_weights[1],  # Weight for positive class
                gamma=2.0,  # Focusing parameter
                reduction='mean'
            )
        else:
            # Default to BCEWithLogitsLoss if no class weights are provided
            self.criterion = nn.BCEWithLogitsLoss()
    
    def set_class_weights(self, targets: torch.Tensor):
        """
        Calculate and set class weights based on the target distribution.
        Uses inverse class frequency to handle class imbalance.
        
        Args:
            targets: Tensor containing class labels (0 or 1). Will be flattened if not 1D.
        """
        if targets is not None and len(targets) > 0:
            # Move to CPU if necessary and ensure it's a 1D tensor
            if targets.is_cuda:
                targets = targets.cpu()
            
            # Flatten the targets tensor if it's not 1D
            if len(targets.shape) > 1:
                targets = targets.view(-1)
            
            # Ensure targets are in the correct format for bincount
            targets = targets.long()
            
            # Calculate class frequencies
            class_counts = torch.bincount(targets)
            
            # Handle case where we might only have one class
            if len(class_counts) == 1:
                # If only one class, create equal weights for both classes
                # This is a fallback and might indicate an issue with the data
                logger.warning(f"Only one class found in targets: {class_counts.tolist()}")
                weights = torch.ones(2, dtype=torch.float32)
            else:
                # Calculate weights inversely proportional to class frequencies
                # Add smoothing to avoid division by zero
                weights = 1.0 / (class_counts.float() + 1e-6)
                
                # Normalize weights to sum to 1
                weights = weights / weights.sum()
            
            logger.info(f"Class counts: {class_counts.tolist()}")
            logger.info(f"Class weights: {weights.tolist()}")
            
            # Store weights as a parameter for the loss function
            self.class_weights = weights.to(self.device)
            
            # Update the loss function with the new weights
            self.update_loss_function()
            
            logger.info(f"Class weights set to: {self.class_weights.tolist()}")
        else:
            # If no targets provided, reset to default
            self.class_weights = None
            self.update_loss_function()
            logger.warning("No targets provided to set_class_weights, using default loss")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader providing training batches
            epoch: Current epoch number (for logging)
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        
        # Calculate class weights on first epoch if not already set
        if epoch == 1 and self.class_weights is None:
            logger.info("Calculating class weights from training data...")
            temp_targets = []
            for _, targets in dataloader:
                temp_targets.append(targets)
            if temp_targets:
                all_targets_tensor = torch.cat(temp_targets)
                self.set_class_weights(all_targets_tensor)
        
        # Training loop
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            # Move data to device
            sequences = sequences.to(self.device)
            targets = targets.to(self.device).float()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Calculate predictions and probabilities
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Update metrics
                epoch_loss += loss.item() * sequences.size(0)  # Multiply by batch size
                all_preds.append(preds.detach().cpu())
                all_targets.append(targets.detach().cpu())
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(
                    f'Train Epoch: {epoch} '
                    f'[{batch_idx * len(sequences)}/{len(dataloader.dataset)} '
                    f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(dataloader.dataset)
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        
        # Log metrics
        logger.info(
            f'\nTrain Epoch: {epoch} Average Loss: {avg_loss:.6f}\n'
            f'Accuracy: {accuracy:.4f}, F1: {f1:.4f}, '
            f'Precision: {precision:.4f}, Recall: {recall:.4f}\n'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predictions': all_preds,
            'targets': all_targets
        }

    def evaluate(self, dataloader: DataLoader, mode: str = 'val', threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate the model on the given data loader.
        
        Args:
            dataloader: DataLoader providing evaluation batches
            mode: Evaluation mode ('val' or 'test') for logging
            threshold: Decision threshold for binary classification (default: 0.5)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        all_probs = []
        all_preds = []
        all_targets = []
        
        # Disable gradient calculation for evaluation
        with torch.no_grad():
            for sequences, targets in dataloader:
                # Move data to device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device).float()
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Get probabilities and predictions
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > threshold).float()
                
                # Update metrics
                batch_size = sequences.size(0)
                epoch_loss += loss.item() * batch_size
                
                # Store results
                all_probs.append(probs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_targets.append(targets.detach().cpu())
        
        # Concatenate all results
        all_probs = torch.cat(all_probs).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        # Calculate metrics
        avg_loss = epoch_loss / len(dataloader.dataset)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        
        # Calculate additional metrics
        try:
            roc_auc = roc_auc_score(all_targets, all_probs)
            avg_precision = average_precision_score(all_targets, all_probs)
        except ValueError as e:
            logger.warning(f"Could not calculate ROC AUC/PR AUC: {e}")
            roc_auc = 0.0
            avg_precision = 0.0
        
        # Log metrics
        logger.info(
            f'\n{mode.capitalize()} Evaluation - Loss: {avg_loss:.6f}\n'
            f'Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, '
            f'Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {avg_precision:.4f}\n'
        )
        
        # Store results in history if in validation mode
        if mode == 'val':
            self.history['val_loss'].append(avg_loss)
            self.history['val_accuracy'].append(accuracy)
            self.history['val_f1'].append(f1)
            self.history['val_precision'].append(precision)
            self.history['val_recall'].append(recall)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'probabilities': all_probs,
            'predictions': all_preds,
            'targets': all_targets
        }

    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        early_stopping_rounds: Optional[int] = 10,
        checkpoint_dir: str = 'checkpoints',
        model_name: str = 'best_model.pth'
    ) -> Dict[str, List[float]]:
        """
        Train the model with early stopping and model checkpointing.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Maximum number of training epochs
            early_stopping_rounds: Number of epochs to wait before early stopping
            checkpoint_dir: Directory to save model checkpoints
            model_name: Name of the model checkpoint file
            
        Returns:
            Dictionary containing training history
        """
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, model_name)
        
        # Initialize early stopping
        if early_stopping_rounds is not None:
            early_stopping = EarlyStopping(
                patience=early_stopping_rounds,
                verbose=True,
                path=best_model_path,
                trace_func=logger.info
            )
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate on validation set if provided
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, mode='val')
                
                # Update learning rate based on validation loss
                self.scheduler.step(val_metrics['loss'])
                
                # Check for early stopping
                if early_stopping_rounds is not None:
                    early_stopping(val_metrics['loss'], self)
                    if early_stopping.early_stop:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
            
            # Log epoch metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Log validation metrics if available
            if val_metrics:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}"
                )
        
        # Load the best model weights
        if val_loader is not None and os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            self.load_model(best_model_path)
        
        return self.history
    
    def save_model(self, path: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the model checkpoint with optional metrics.
        
        Args:
            path: Path to save the model checkpoint
            metrics: Optional dictionary of metrics to save with the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'epoch': getattr(self, 'current_epoch', 0),
            'best_val_loss': getattr(self, 'best_val_loss', float('inf')),
            'class_weights': self.class_weights,
            'metrics': metrics,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'bidirectional': self.model.bidirectional,
                'use_attention': self.model.use_attention
            }
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            path: Path to the model checkpoint
            
        Returns:
            Dictionary containing saved metrics and configuration
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        # Verify model architecture matches
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            for key, value in model_config.items():
                if hasattr(self.model, key) and getattr(self.model, key) != value:
                    logger.warning(
                        f"Model parameter mismatch: {key} (loaded: {value}, "
                        f"current: {getattr(self.model, key)})"
                    )
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load class weights if available
        if 'class_weights' in checkpoint and checkpoint['class_weights'] is not None:
            self.class_weights = checkpoint['class_weights'].to(self.device)
            self.update_loss_function()
        
        # Load best validation loss if available
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Model loaded from {path}")
        
        # Return saved metrics and config
        return {
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('model_config', {})
        }


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    
    Adapted from: https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(
        self, 
        patience: int = 7, 
        verbose: bool = False, 
        delta: float = 0,
        path: str = 'checkpoint.pt',
        trace_func=print
    ):
        """
        Args:
            patience: How long to wait after last time validation loss improved
            verbose: If True, prints a message for each validation loss improvement
            delta: Minimum change in the monitored quantity to qualify as an improvement
            path: Path for the checkpoint to be saved to
            trace_func: Trace print function (default: print)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model_trainer):
        """
        Call this when you want to check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            model_trainer: The model trainer instance to save
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_trainer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_trainer)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model_trainer):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        # Save the model
        model_trainer.save_model(self.path)
        self.val_loss_min = val_loss

