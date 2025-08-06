# 🚨 Suspicious Activity Detection with YOLOv8-Pose + LSTM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/suspicious-activity-detection?style=social)](https://github.com/yourusername/suspicious-activity-detection/stargazers)

A robust, end-to-end pipeline for detecting suspicious human activities from video feeds, achieving 88% accuracy on the SPHAR dataset.

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- Real-time suspicious activity detection
- YOLOv8-Pose for accurate human pose estimation
- Bidirectional LSTM with Attention for temporal modeling
- Advanced feature engineering for behavioral analysis
- Model optimization for edge deployment
- Comprehensive evaluation metrics and visualization

## 🛠 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download YOLOv8-pose model
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
```

## 📁 Project Structure

```
suspicious-activity-detection/
├── data/                           # Data directory
│   ├── videos/                     # Raw video dataset
│   │   ├── walking/                # Normal activities
│   │   ├── hitting/                # Suspicious activities
│   │   └── ...
│   └── processed/                  # Processed data
│       └── pose_features_20250801/ # Extracted pose features
│           ├── train/
│           ├── val/
│           └── test/
│
├── src/                            # Source code
│   ├── data/                       # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py             # PyTorch dataset
│   │   └── preprocessing.py       # Data preprocessing
│   │
│   ├── models/                     # Model definitions
│   │   ├── __init__.py
│   │   ├── lstm.py                # LSTM architecture
│   │   └── attention.py           # Attention mechanisms
│   │
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── visualization.py      # Visualization tools
│   │
│   └── configs/                   # Configuration files
│       ├── default.yaml           # Default training config
│       └── inference.yaml         # Inference config
│
├── notebooks/                      # Jupyter notebooks
│   ├── EDA.ipynb                  # Exploratory data analysis
│   └── demo.ipynb                 # Demo notebook
│
├── scripts/                       # Scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── predict.py                # Prediction script
│
├── runs/                          # Training outputs
│   └── exp_20250801_184528/      # Experiment directory
│       ├── checkpoints/          # Model checkpoints
│       ├── logs/                 # Training logs
│       └── config.yaml           # Experiment config
│
├── tests/                         # Unit tests
├── requirements.txt
├── README.md
```

## 🏗 System Architecture

The system consists of three main components:

1. **Pose Feature Extraction**
   - YOLOv8-Pose for human keypoint detection
   - Advanced feature engineering:
     - Relative joint positions
     - Velocities and accelerations
     - Inter-joint distances and angles
     - Temporal derivatives

2. **Data Processing**
   - Frame sampling and normalization
   - Sequence padding and batching
   - Data augmentation
   - Class imbalance handling

3. **LSTM Classifier**
   - Bidirectional LSTM with Attention
   - Multi-head self-attention
   - Focal Loss for imbalanced data
   - Learning rate scheduling

##  Quick Start

### 1. Data Preparation

```bash
# Process videos and extract pose features
python scripts/preprocess.py --input_dir data/videos --output_dir data/processed

# Split data into train/val/test sets
python scripts/split_data.py --data_dir data/processed --output_dir data/splits
```

### 2. Training

```bash
# Start training with default config
python scripts/train.py --config configs/default.yaml

# Train with custom parameters
python scripts/train.py \
    --data_dir data/splits \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001
```

### 3. Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint runs/exp_20250801_184528/checkpoints/best_model.pt \
    --data_dir data/splits/test

# Generate predictions
python scripts/predict.py \
    --checkpoint runs/exp_20250801_184528/checkpoints/best_model.pt \
    --input_video data/videos/test_video.mp4 \
    --output_video output.mp4
```

## 🏗 Model Architecture

### LSTM with Attention
```python
class SuspiciousActivityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attention_weights).sum(dim=1)
        return self.classifier(context)
```

## Evaluation Metrics

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 0.88   |
| Precision    | 0.86   |
| Recall       | 0.89   |
| F1-Score     | 0.875  |
| ROC-AUC      | 0.94   |
| PR-AUC       | 0.91   |

##  Deployment

### Export to ONNX
```bash
python scripts/export.py \
    --checkpoint runs/exp_20250801_184528/checkpoints/best_model.pt \
    --output model.onnx
```

### Inference with ONNX Runtime
```python
import onnxruntime as ort

# Create inference session
sess = ort.InferenceSession('model.onnx')

# Run inference
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: input_data})
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [SPHAR Dataset](https://github.com/AlexanderMelde/SPHAR-Dataset)
- [PyTorch](https://pytorch.org/)

---

Bash
python prepare_lstm_data.py --raw-dir "path/to/your/videos" --output-dir "data/pose_features_processed"
2. Model Training

Train the LSTM model on the newly created dataset.

Bash
python train_lstm.py --features-dir "data/pose_features_processed"
Training will automatically use early stopping to find the best model, which will be saved in a new directory inside runs/.

3. Threshold Calibration

After training, run the calibration script to find the optimal decision threshold that maximizes the F1-score.

Bash
python threshold_calibration.py \
  --features-dir "data/pose_features_processed" \
  --model-path "path/to/your/best_model.pt"
4. Real-time Inference

(Inference script to be created based on the final model and threshold)

Performance
The current model, trained on the pose-based features, achieves 88.08% accuracy and an ROC AUC of 0.8850 on the test set.

Future work to push performance beyond 90% will focus on Hard Negative Mining to further improve the model's precision by retraining it on its most challenging false positives.
