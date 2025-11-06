# Electronics Detection System

A real-time camera-based system that detects **portable travel electronics** at airports and train stations, displaying their estimated power consumption to help travelers understand their personal device energy footprint.

## 🎯 Project Focus

This system is designed for **travel electronics detection** in:
- ✈️ **Airport Security Checkpoints** - Electronics in screening trays
- 🔌 **Charging Stations** - Devices being charged at gates/waiting areas
- 🚂 **Train Station Waiting Areas** - Travelers using personal devices
- 🎒 **Baggage Screening Areas** - Portable electronics in luggage

## Features

- ✅ **High-performance display** - 60 FPS with intelligent frame skipping
- ✅ **Real-time object detection** using YOLOv8
- ✅ **Custom visual overlays** with device-specific icons and colors
- ✅ **Device information display** (type, power consumption, confidence, traveler prevalence)
- ✅ **Configurable detection zone** with interactive controls
- ✅ **Pre-flight system validation** - Automated dependency and permission checks
- ✅ **Cross-platform** (Windows, Mac, Linux)
- ✅ **Offline operation** (runs locally after initial model download)
- ✅ **Adaptive detection** - Toggle between speed and accuracy modes

## Supported Travel Electronics

Currently detecting **6 device types** (16% coverage of 38 travel electronics):

| Device | Prevalence | Power | Status |
|--------|------------|-------|--------|
| 📱 Cell Phone | 99% travelers | 5-10W | ✅ |
| 💻 Laptop | 70% travelers | 45-65W | ✅ |
| 🖱️ Mouse | 40% travelers | 1-3W | ✅ |
| 💇 Hair Dryer | 30% travelers | 800-1200W | ✅ |
| ⌨️ Keyboard | 25% travelers | 2-5W | ✅ |
| 📡 Remote | 10% travelers | 0.5W | ✅ |

**Next Phase (High Priority):**
- 📱 Tablet (40% travelers)
- 🎧 Wireless Earbuds (60% travelers)
- 🔋 Power Bank (50% travelers)
- 🔌 Phone Charger (90% travelers)
- 🎧 Wireless Headphones (45% travelers)
- ⌚ Smartwatch (35% travelers)

See `TRAVEL_ELECTRONICS_CATALOG.md` for full device list (38 devices).

## Quick Start

### Easy Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/GAM3RG33K/electronics_detector
cd electronics_detector

# 2. Make scripts executable
chmod +x setup-py.sh run.sh

# 3. Run setup (creates virtual environment and installs dependencies)
./setup-py.sh

# 4. Activate virtual environment (follow the output from setup-py.sh)
source .py_env/bin/activate

# 5. Run the detector
./run.sh
```

**First Run Notes:**
- May download YOLOv8n model file (~7 MB) automatically
- Will request camera permission (required)
- Takes maximum FPS supported by your camera (typically 30-60 fps)
- On macOS: May need to grant Terminal camera access in System Preferences

### Manual Installation

#### Prerequisites
- Python 3.8 or higher
- A webcam
- Internet connection (for first-time model download only)

#### Setup Steps

1. **Install Python** (if not already installed):
   - Windows: Download from [python.org](https://www.python.org/downloads/)
   - Mac: `brew install python` or download from python.org
   - Linux: Usually pre-installed, or `sudo apt install python3 python3-pip`

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install with system packages flag:
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

3. **First run** (downloads YOLO model, ~6MB):
   ```bash
   python electronics_detector.py
   ```

## Usage

### Running the Application

**Quick run (Unix/Mac):**
```bash
./run.sh
```

**Direct Python:**
```bash
python electronics_detector.py
```

**Windows:**
```bash
run.bat
```

### Keyboard Controls

- **Q** - Quit the application
- **Z** - Toggle detection zone visibility
- **+** or **=** - Expand detection zone
- **-** - Shrink detection zone
- **D** - Toggle detection interval (high accuracy ↔ balanced ↔ high speed)

### Detection Modes

Press **D** to cycle through detection speeds:
1. **Every 1 frame** - High accuracy (~30fps display)
2. **Every 2 frames** - Balanced (~50fps display) *[Default]*
3. **Every 3 frames** - High speed (~60fps display)

### Detection Zone

The detection zone is the area where objects will be detected. By default, it covers the center 80% of the camera view. Objects outside this zone are ignored.

You can:
- Toggle visibility with **Z**
- Resize with **+** and **-** keys
- Modify in code by editing the `zone` dictionary in `ElectronicsDetector.__init__()`

## Training Custom Models

To add more device types (tablets, earbuds, power banks, etc.), you can train custom models:

### 🎯 Recommended: Google Colab (FREE GPU)

**Training time: 2-3 hours** (vs 24+ hours on local CPU)

Google Colab provides free GPU access, making training much faster than on most local machines.

#### Quick Start with Colab:

1. **Download Dataset**:
   - Visit [Roboflow Universe](https://universe.roboflow.com/sanctum/electronics-j0cxl)
   - Click "Download Dataset" → Format: YOLOv8 → Download ZIP

2. **Upload to Google Drive**:
   - Create a folder in Google Drive (e.g., `Electronics.v1i.yolov8`)
   - Upload and extract the dataset ZIP

3. **Open Notebook**:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click "File" → "Upload notebook"
   - Upload `train_colab.ipynb` from this project

4. **Enable GPU**:
   - In Colab: Runtime → Change runtime type → Hardware accelerator → **GPU (T4)**
   - Click "Save"

5. **Run the Notebook**:
   - Follow the step-by-step instructions in the notebook
   - Update the `DATASET_PATH` to match your Google Drive folder
   - Run each cell in sequence (Shift+Enter)

6. **Download Results**:
   - The notebook will automatically download your trained model
   - Extract the ZIP and use `weights/best.pt` in your detector

#### Direct Colab Link:
```
Open train_colab.ipynb in Google Colab
```

---

### 💻 Alternative: Local Training

If you prefer to train locally or have a powerful GPU:

```bash
# Download a dataset (e.g., from Roboflow Universe)
# https://universe.roboflow.com/sanctum/electronics-j0cxl

# Train the model
python train_electronics.py --dataset /path/to/dataset

# Optional parameters:
python train_electronics.py \
  --dataset /path/to/dataset \
  --model yolov8s.pt \
  --epochs 50 \
  --batch 8 \
  --device 0  # Use 'mps' for Apple Silicon, 'cpu' for CPU-only

# For Apple Silicon (M1/M2/M3) - disable validation due to MPS issues
python train_electronics.py \
  --dataset /path/to/dataset \
  --device mps \
  --no-val
```

**Local Training Times (100 epochs):**
- 🖥️ **CPU**: ~24 hours
- 🍎 **Apple M2 Pro (MPS)**: ~40 hours (with `--no-val`)
- 🎮 **NVIDIA RTX 3060**: ~2-3 hours
- 🚀 **NVIDIA RTX 4090**: ~1 hour

See `train_electronics.py --help` for all options.

### Training Results

After training completes, update `electronics_detector.py` to use your trained model:
```python
# Replace this line:
self.model = YOLO('yolov8n.pt')

# With your trained model:
self.model = YOLO('path/to/best.pt')
# Example: self.model = YOLO('electronics_training/colab_run/weights/best.pt')
```

## Customization

### Adjust Detection Zone in Code

Edit the `zone` dictionary (values from 0.0 to 1.0):

```python
self.zone = {
    'x1': 0.1,  # Left boundary (10% from left edge)
    'y1': 0.1,  # Top boundary (10% from top)
    'x2': 0.9,  # Right boundary (90% from left edge)
    'y2': 0.9   # Bottom boundary (90% from top)
}
```

### Add More Electronics

Add to the `electronics_map` dictionary:

```python
'your_device': {
    'color': (B, G, R),          # BGR color for overlay
    'icon': '🔌',                # Unicode icon
    'power': 'XX-YYW',           # Power consumption
    'type': 'Device Type',       # Category
    'prevalence': 'XX%'          # % of travelers
}
```

### Adjust Camera Settings

Modify camera resolution and FPS in the `run()` method:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Height
cap.set(cv2.CAP_PROP_FPS, 60)             # FPS target
```

### Detection Sensitivity

Adjust the confidence threshold in the `run()` method:

```python
results = self.model(frame, conf=0.3, verbose=False)
# Lower conf = more detections (more false positives)
# Higher conf = fewer detections (more accurate)
```

## Performance Tips

### For 60fps Target:

1. **Use detection interval mode** - Press 'D' to cycle through speed modes
2. **Use smaller resolution** - 1280x720 instead of 1920x1080
3. **Use YOLOv8n (nano)** - Already configured for speed
4. **Reduce confidence threshold** - Faster processing with `conf=0.4`
5. **GPU acceleration** - Install CUDA-enabled PyTorch for NVIDIA GPUs

### GPU Support (Optional, for better performance):

Install CUDA-enabled PyTorch:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

### Camera not opening
- **macOS**: Run `tccutil reset Camera` then grant permission when prompted
- Check if another application is using the camera
- Try different camera indices: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- On Linux, check camera permissions

### Low FPS
- Press **D** to switch to high-speed mode (every 3 frames)
- Reduce camera resolution
- Close other applications
- Use GPU acceleration (see above)
- The YOLOv8n model is already optimized for speed

### Pre-flight checks failing
- **NumPy 2.x error**: Run `pip uninstall numpy -y && pip install "numpy<2"`
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Camera permission denied**: Follow platform-specific instructions shown by the checker

### Model download fails
- Check internet connection
- The model (~6MB) downloads automatically on first run
- Manual download: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

### No detections
- Ensure objects are within the detection zone (green box)
- Move closer to the camera
- Adjust lighting conditions
- Lower the confidence threshold in code
- Press 'Z' to show detection zone

## Technical Details

- **Detection Model**: YOLOv8n (Nano) - optimized for real-time inference
- **Framework**: OpenCV for camera and display, Ultralytics for detection
- **Platform**: Cross-platform (Windows/Mac/Linux with Python 3.8+)
- **Offline**: Yes (after initial model download)
- **Display FPS**: 60fps (achieved via frame skipping)
- **Detection FPS**: Configurable (30fps at interval=2, 20fps at interval=3)
- **Resolution**: 1280x720 (optimal balance)
- **Detection Latency**: <100ms between updates

## Project Structure

```
energy-footprint-py/
├── electronics_detector.py           # Main detection system (~650 lines)
├── train_electronics.py              # Training script for local/custom training
├── train_colab.ipynb                 # Google Colab training notebook (GPU)
├── requirements.txt                  # Python dependencies
├── CONTEXT.md                        # Project documentation (source of truth)
├── TRAVEL_ELECTRONICS_CATALOG.md    # Full device catalog (38 devices)
├── README.md                         # This file
├── yolov8n.pt                       # YOLOv8 nano model (~6MB)
├── setup-py.sh                      # Automated setup script
├── run.sh                           # Quick launch script (Unix/Mac)
└── run.bat                          # Quick launch script (Windows)
```

## License

This is a demonstration project focused on travel electronics detection. YOLOv8 is licensed under AGPL-3.0.

## Credits

- YOLOv8 by Ultralytics
- OpenCV for computer vision
- Roboflow for dataset hosting and tools

## Contributing

See `CONTEXT.md` for development guidelines and project architecture.

---

**Repository**: https://github.com/GAM3RG33K/electronics_detector