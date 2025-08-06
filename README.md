Real-Time Suspicious Activity Detection with YOLOv8-Pose + LSTM
This project implements a robust, end-to-end pipeline for detecting suspicious human activities from video feeds, achieving 88% accuracy on the SPHAR dataset.

Project Structure
suspicious-activity-detection/
├── README.md
├── requirements.txt
├── prepare_lstm_data.py        # Extracts pose features and splits data
├── train_lstm.py               # Main training script
├── lstm_model.py               # Defines the LSTM architecture and trainer
├── threshold_calibration.py    # Script to find the optimal decision threshold
├── data/
│   ├── videos/                 # SPHAR dataset source videos
│   │   ├── walking/
│   │   └── hitting/
│   │   └── ...
│   └── pose_features_processed_20250801_1900/ # Final processed data
│       ├── train_features.npy
│       ├── train_labels.npy
│       └── ...
└── runs/
    └── exp_20250801_184528/      # Example output from a training run
        ├── checkpoints/
        │   └── best_model.pt   # The final trained model
        ├── final_metrics.json  # Final performance metrics
        └── ...
System Architecture
The system consists of three main stages:

Pose Feature Extraction: Uses YOLOv8-Pose to extract human keypoints from video frames. It then engineers an advanced feature set including the relative positions, velocities, and accelerations of body joints to create a rich, behavioral signal.

Data Loading: A memory-efficient custom PyTorch Dataset loads the engineered features on-the-fly and a WeightedRandomSampler is used to handle the significant class imbalance during training.

LSTM Model: A powerful, multi-layer Bidirectional LSTM with Attention, trained with FocalLoss to focus on hard-to-classify examples, classifies the temporal sequences.

Installation
Bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the required YOLOv8-pose model
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
Usage
1. Feature Extraction & Data Preparation

Run the preparation script to process your videos into the final feature set and create the train/val/test splits.

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
