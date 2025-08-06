
# Real-time Suspicious Activity Detection with YOLOv8 + LSTM

## Project Structure

```
suspicious-activity-detection/
├── README.md                    # This file
├── requirements.txt            # Python dependencies
├── main.py                     # Main entry point
├── feature_extractor.py        # YOLOv8 feature extraction
├── lstm_model.py               # Bidirectional LSTM model
├── video_processor.py          # Real-time video processing
├── dataset_training.py         # SPHAR dataset training
├── threshold_calibration.py    # ROC/PR curve calibration
├── model_conversion.py         # Model format conversion
├── deployment/                 # Deployment files (created after conversion)
│   ├── model.torchscript      # TorchScript model
│   ├── model.onnx             # ONNX model
│   ├── deployment_info.json   # Deployment instructions
│   └── inference_example.py   # Example inference script
└── videos/                     # SPHAR dataset (your data)
    ├── carcrash/
    ├── falling/
    ├── hitting/
    ├── igniting/
    ├── kicking/
    ├── luggage/
    ├── murdering/
    ├── neutral/
    ├── panicking/
    ├── running/
    ├── sitting/
    ├── stealing/
    ├── vandalizing/
    └── walking/
```

## System Architecture

The system consists of four main components:

1. **YOLOv8 Feature Extractor**: Extracts bounding boxes, confidences, and class IDs from video frames
2. **Bidirectional LSTM**: Processes temporal sequences for activity classification
3. **Video Processor**: Handles real-time frame capture and sliding window inference
4. **Threshold Calibrator**: Optimizes detection thresholds using ROC and PR curves

## Hardware Requirements

### Development (macOS M1/M2)
- macOS Sonoma 14.3+
- Apple Silicon (M1/M2/M3)
- 16GB+ RAM recommended
- Python 3.10+

### Production (Raspberry Pi)
- Raspberry Pi 4 Model B (4GB+ RAM recommended)
- Raspberry Pi OS 64-bit (Bookworm)
- MicroSD card (32GB+ Class 10)
- Pi Camera Module or USB webcam

## Installation

### macOS M1/M2 Setup

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch with MPS support (CPU fallback for YOLOv8)
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Download YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Raspberry Pi Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install python3-pip python3-venv libopenblas-dev libopenmpi-dev libomp-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch for ARM64
pip install torch>=1.8.0 torchvision>=0.9.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

## Usage

### 1. Dataset Preparation

Download and extract the SPHAR dataset to the `videos/` directory:

```bash
# The dataset should be organized as shown in the project structure above
# Each folder contains videos of the corresponding activity type
```

### 2. Training

Train the LSTM model on the SPHAR dataset:

```bash
# Basic training
python main.py train

# Advanced training options
python main.py train --epochs 100 --batch-size 16 --device cpu
```

Training will:
- Automatically classify activities as suspicious vs normal
- Split data into train/validation/test sets
- Save the best model as `best_model.pth`
- Log training progress and metrics

### 3. Threshold Calibration

Calibrate optimal detection thresholds:

```bash
python main.py calibrate --model-path best_model.pth
```

This will:
- Generate ROC and Precision-Recall curves
- Compute optimal thresholds using Youden's Index and F1-score
- Provide recommendations for different operational priorities
- Save results to `threshold_calibration.json`

### 4. Model Conversion

Convert the model for deployment:

```bash
python main.py convert --model-path best_model.pth --output-dir deployment
```

This creates:
- TorchScript model (`model.torchscript`)
- ONNX model (`model.onnx`)
- Quantized model for ARM64 (`model_quantized.torchscript`)
- Deployment instructions and example code

### 5. Real-time Inference

Run real-time suspicious activity detection:

```bash
# Use webcam
python main.py inference --source 0

# Use video file
python main.py inference --source path/to/video.mp4

# Advanced options
python main.py inference \
    --source 0 \
    --threshold 0.7 \
    --sequence-length 16 \
    --save-results
```

Output format:
```json
{
  "frame_idx": 245,
  "suspicious_score": 0.856,
  "timestamp": 1678901234.567,
  "is_suspicious": true
}
```

## Model Architecture

### YOLOv8 Feature Extraction
- Model: YOLOv8n (nano) for speed
- Output: Bounding boxes (x, y, w, h), confidence, class_id
- Features per frame: 60 (10 detections × 6 features)
- Normalization: Coordinates normalized to [0,1]

### Bidirectional LSTM
- Architecture: 2-layer Bi-LSTM
- Hidden units: 256 per direction (512 total)
- Dropout: 0.3
- Attention: Additive attention mechanism
- Output: Single probability (suspicious/normal)

### Sequence Processing
- Sequence length: 16 frames (configurable)
- Frame rate: 30 FPS
- Sliding window: Real-time inference
- Buffer size: 60 frames maximum

## Performance Optimization

### For Raspberry Pi

1. **Use quantized models**:
```python
# Load quantized model
model = torch.jit.load('deployment/model_quantized.torchscript')
```

2. **Reduce sequence length**:
```bash
python main.py inference --sequence-length 8
```

3. **Lower frame rate**:
```python
# In video_processor.py, adjust target_fps
video_processor = VideoProcessor(target_fps=15)
```

### For macOS M1

1. **Use CPU device** (YOLOv8 MPS compatibility issues):
```bash
python main.py inference --device cpu
```

2. **Enable optimizations**:
```python
# In your code
torch.set_num_threads(4)  # Adjust based on your CPU
```

## Threshold Selection Guidelines

Based on operational priorities:

- **Balanced Performance**: Use ROC threshold (Youden's Index)
- **High Precision** (fewer false alarms): Use PR threshold (max F1-score)
- **High Recall** (catch all incidents): Use lower threshold
- **Conservative** (minimize false positives): Use higher threshold

Example threshold values:
- Security cameras: 0.3-0.5 (prioritize recall)
- Smart home: 0.6-0.8 (minimize false alarms)
- Research: 0.5 (balanced)

## GPIO Integration (Raspberry Pi)

Enable GPIO alerts for suspicious activity:

```python
# Uncomment in video_processor.py
def trigger_gpio_alert(self):
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)  # LED/Buzzer pin
    GPIO.output(18, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(18, GPIO.LOW)
```

## Troubleshooting

### Common Issues

1. **YOLOv8 MPS errors on macOS**:
   - Solution: Use `--device cpu`

2. **PyTorch installation on Raspberry Pi**:
   - Use ARM64 wheels: `--index-url https://download.pytorch.org/whl/cpu`

3. **Out of memory errors**:
   - Reduce batch size: `--batch-size 4`
   - Reduce sequence length: `--sequence-length 8`

4. **Slow inference**:
   - Use quantized models
   - Reduce frame rate
   - Use smaller YOLOv8 model (yolov8n)

### Performance Monitoring

Monitor system performance:
```python
# In your inference loop
stats = video_processor.get_statistics()
print(f"Processing rate: {stats['processing_rate']:.2f}")
print(f"Suspicious rate: {stats['suspicious_rate']:.2f}")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{suspicious_activity_detection_2025,
  title={Real-time Suspicious Activity Detection using YOLOv8 and Bidirectional LSTM},
  author={AI Assistant},
  year={2025},
  howpublished={\url{https://github.com/username/suspicious-activity-detection}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [SPHAR Dataset](https://github.com/AlexanderMelde/SPHAR-Dataset)
- PyTorch Team for ARM64 support
