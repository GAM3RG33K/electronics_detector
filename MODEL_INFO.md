# YOLOv8 Segmentation Model Information

## 📦 Model Details

**Current Model:** YOLOv8n-seg (Nano Segmentation)
**File:** `yolov8n-seg.pt`
**Size:** ~7MB
**Type:** Instance Segmentation

## 🔽 Automatic Download

The model will **automatically download** on first run when you execute:

```bash
python electronics_detector.py
```

The Ultralytics library will download it from:
- **Official Source:** https://github.com/ultralytics/assets/releases/

## 📍 Download Location

The model will be saved to:
- **macOS/Linux:** `~/.cache/torch/hub/ultralytics_yolov8_main/`
- **Windows:** `C:\Users\<YourUsername>\.cache\torch\hub\ultralytics_yolov8_main\`

Or in your current directory as `yolov8n-seg.pt`

## 🔧 Manual Download (Optional)

If you want to download manually:

```bash
# Using wget
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt

# Using curl
curl -L https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt -o yolov8n-seg.pt
```

Or visit: https://docs.ultralytics.com/models/yolov8/#segmentation

## ✨ What Changed

### Previous: YOLOv8n (Detection)
- Provided bounding boxes only
- Required complex contour detection
- Approximate device shapes

### Current: YOLOv8n-seg (Segmentation)
- Provides pixel-perfect masks
- Exact device boundaries
- No contour detection needed
- Same speed, better accuracy

## 🎯 Benefits

✅ **Precise Device Contours** - Pixel-perfect device shapes
✅ **Better Thermal Overlay** - Masks follow exact device boundaries  
✅ **No False Contours** - No background noise in masks
✅ **Same Performance** - Minimal speed difference
✅ **Same Classes** - Works with all COCO classes (cell phone, laptop, etc.)

## 🔄 Reverting to Detection Model

If you need to revert to the detection model:

```python
# In electronics_detector.py line 304:
self.model = YOLO('yolov8n.pt')  # Detection model
```

## 📊 Model Comparison

| Feature | YOLOv8n | YOLOv8n-seg |
|---------|---------|-------------|
| Output | Bounding boxes | Boxes + Masks |
| Size | ~6MB | ~7MB |
| Speed | ~30ms | ~35ms |
| Accuracy | Good | Excellent |
| Device Shapes | Approximate | Exact |

## 🆘 Troubleshooting

### Model Download Fails
```bash
# Check internet connection
ping github.com

# Try manual download
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
```

### Model Not Found
```bash
# Ensure model is in project directory or cache
ls -la yolov8n-seg.pt
```

### Permission Issues
```bash
# Check cache directory permissions
ls -la ~/.cache/torch/hub/
```

## 📚 Additional Resources

- **Ultralytics Docs:** https://docs.ultralytics.com/
- **YOLOv8 Segmentation:** https://docs.ultralytics.com/tasks/segment/
- **Model Zoo:** https://github.com/ultralytics/ultralytics

