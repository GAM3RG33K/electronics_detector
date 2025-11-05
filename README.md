# Electronics Detection System

A real-time camera-based system that detects electronic devices and displays custom overlays with device information.

## Features

- ✅ **High-quality camera feed** with optimized settings
- ✅ **Real-time object detection** using YOLOv8
- ✅ **Custom visual overlays** with device-specific icons and colors
- ✅ **Device information display** (type, power consumption, confidence)
- ✅ **Configurable detection zone** (limit detection to specific area)
- ✅ **Cross-platform** (Windows, Mac, Linux)
- ✅ **Offline operation** (runs locally after initial model download)
- ✅ **FPS monitoring** with real-time performance stats

## Supported Electronics

The system can detect:
- 💻 Laptops
- 📱 Cell phones
- 📺 TVs
- ⌨️ Keyboards
- 🖱️ Mice
- 📡 Remotes
- 📟 Microwaves
- 🔥 Ovens
- 🍞 Toasters
- ❄️ Refrigerators

## Installation

### Prerequisites
- Python 3.8 or higher
- A webcam
- Internet connection (for first-time model download only)

### Setup

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

```bash
python electronics_detector.py
```

### Keyboard Controls

- **Q** - Quit the application
- **Z** - Toggle detection zone visibility
- **+** or **=** - Expand detection zone
- **-** - Shrink detection zone

### Detection Zone

The detection zone is the area where objects will be detected. By default, it covers the center 80% of the camera view. Objects outside this zone are ignored.

You can:
- Toggle visibility with **Z**
- Resize with **+** and **-** keys
- Modify in code by editing the `zone` dictionary in `ElectronicsDetector.__init__()`

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
    'color': (R, G, B),      # BGR color for overlay
    'icon': '🔌',            # Unicode icon
    'power': 'XX-YYW',       # Power consumption
    'type': 'Device Type'    # Category
}
```

### Adjust Camera Settings

Modify camera resolution and FPS in the `run()` method:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height
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

1. **Use smaller resolution**: 1280x720 instead of 1920x1080
2. **Use YOLOv8n (nano)**: Already configured for speed
3. **Reduce confidence threshold**: Faster processing with `conf=0.4`
4. **GPU acceleration**: Install CUDA-enabled PyTorch for NVIDIA GPUs

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
- Check if another application is using the camera
- Try different camera indices: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- On Linux, check camera permissions

### Low FPS
- Reduce camera resolution
- Close other applications
- Use GPU acceleration (see above)
- The YOLOv8n model is already optimized for speed

### Model download fails
- Check internet connection
- The model (~6MB) downloads automatically on first run
- Manual download: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

### No detections
- Ensure objects are within the detection zone (green box)
- Move closer to the camera
- Adjust lighting conditions
- Lower the confidence threshold in code

## Technical Details

- **Detection Model**: YOLOv8n (Nano) - optimized for real-time inference
- **Framework**: OpenCV for camera and display, Ultralytics for detection
- **Platform**: Cross-platform (Windows/Mac/Linux)
- **Offline**: Yes (after initial model download)
- **FPS**: 30-60fps depending on hardware and resolution

## License

This is a demonstration project. YOLOv8 is licensed under AGPL-3.0.

## Credits

- YOLOv8 by Ultralytics
- OpenCV for computer vision
