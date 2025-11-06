# 🚀 Google Colab Training Guide

Complete guide to train your electronics detection model using **FREE GPU** on Google Colab.

---

## Why Use Google Colab?

| Method | Time (100 epochs) | Cost | GPU |
|--------|------------------|------|-----|
| **Google Colab** | **2-3 hours** | Free | ✅ Tesla T4 |
| Local CPU | ~24 hours | Free | ❌ |
| Local M2 Pro (MPS) | ~40 hours | Free | ⚠️ Limited |
| Cloud GPU (AWS/Azure) | 2-3 hours | ~$1-3/hour | ✅ |

**Winner:** Google Colab provides free GPU access with no setup required!

---

## Step-by-Step Guide

### 1. Prepare Your Dataset

#### Option A: Download from Roboflow (Recommended)

1. Visit [Roboflow Universe - Electronics Dataset](https://universe.roboflow.com/sanctum/electronics-j0cxl)
2. Click "Download Dataset"
3. Select format: **YOLOv8**
4. Click "Download ZIP"
5. Save the file (e.g., `Electronics.v1i.yolov8.zip`)

#### Option B: Use Your Own Dataset

Ensure your dataset follows this structure:
```
Electronics.v1i.yolov8/
├── data.yaml          # Dataset configuration
├── train/
│   ├── images/        # Training images
│   └── labels/        # Training labels (.txt)
├── valid/
│   ├── images/        # Validation images
│   └── labels/        # Validation labels
└── test/ (optional)
    ├── images/
    └── labels/
```

---

### 2. Upload Dataset to Google Drive

1. Go to [Google Drive](https://drive.google.com/)
2. Create a new folder (e.g., "ML_Datasets")
3. Upload your dataset ZIP file
4. Right-click → "Extract" to unzip it
5. Note the folder path (e.g., `/content/drive/MyDrive/ML_Datasets/Electronics.v1i.yolov8`)

**💡 Tip:** Uploading to Google Drive means you won't lose your dataset if Colab disconnects!

---

### 3. Open the Training Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Upload notebook"
3. Select `train_colab.ipynb` from this project
4. The notebook will open in a new tab

**Alternative:** You can also drag and drop the `.ipynb` file into Colab.

---

### 4. Enable GPU

🚨 **CRITICAL STEP** - Don't skip this!

1. In Colab, click "Runtime" in the menu
2. Select "Change runtime type"
3. Under "Hardware accelerator", select **GPU**
4. Choose **T4** if prompted (it's the free tier GPU)
5. Click "Save"

You should see "Connected to Python 3 Google Compute Engine backend (GPU)" at the top right.

---

### 5. Run the Notebook

**Follow these steps in order:**

#### Cell 1: Check GPU
Run the first cell to verify GPU is available:
```python
!nvidia-smi
```
You should see Tesla T4 GPU information.

#### Cell 2: Install Dependencies
This installs YOLOv8 and other required packages (~1 minute).

#### Cell 3: Mount Google Drive
Authorize access to your Google Drive:
- Click the link
- Sign in with your Google account
- Copy the authorization code
- Paste it back in Colab

#### Cell 4: Set Dataset Path
**IMPORTANT:** Update the path to match your dataset location:
```python
DATASET_PATH = "/content/drive/MyDrive/ML_Datasets/Electronics.v1i.yolov8"
```

#### Cell 5: Configure Training
Adjust settings if needed:
- `epochs`: 100 (recommended), reduce to 50 for faster testing
- `batch`: 16 (reduce to 8 if out of memory)
- `model`: 'yolov8n.pt' (nano, fastest)

#### Cell 6: Start Training 🏋️
This is the main training cell. It will take **2-3 hours**.

⚠️ **Keep the tab open!** Colab may disconnect if you close the tab or your computer sleeps.

**Progress indicators:**
- Epoch counter (1/100, 2/100, etc.)
- Loss values (should decrease over time)
- mAP metrics (should increase over time)

#### Cell 7-9: View Results
After training:
- Cell 7: Validation metrics
- Cell 8: Test on sample images
- Cell 9: View training curves

#### Cell 10: Download Model
Downloads a ZIP file containing:
- `weights/best.pt` - Best performing model
- `weights/last.pt` - Final epoch model
- `results.png` - Training curves
- `confusion_matrix.png` - Performance visualization

#### Cell 11 (Optional): Save to Google Drive
Saves everything to your Google Drive for permanent storage.

---

### 6. Use Your Trained Model

1. **Extract the downloaded ZIP**
2. **Find `weights/best.pt`**
3. **Copy it to your project folder**
4. **Update `electronics_detector.py`:**

```python
# Find this line (around line 40):
self.model = YOLO('yolov8n.pt')

# Replace with:
self.model = YOLO('path/to/best.pt')
# Example: self.model = YOLO('models/electronics_best.pt')
```

5. **Run your detector:**
```bash
python electronics_detector.py
```

---

## Troubleshooting

### "No GPU available"
- Runtime → Change runtime type → GPU → Save
- Restart the runtime (Runtime → Restart runtime)
- Check if you've exceeded free tier limits (Colab limits GPU hours)

### "Out of memory"
Reduce batch size in Cell 5:
```python
'batch': 8  # or even 4
```

### "Dataset not found"
- Verify the DATASET_PATH in Cell 4
- Make sure Google Drive is mounted (Cell 3)
- Check that data.yaml exists in the folder

### "Disconnected from runtime"
Colab may disconnect after ~12 hours or if idle:
- Training progress is saved every 10 epochs
- Resume by running all cells again
- Consider using Colab Pro for longer sessions

### "Validation errors on MPS"
This only applies to local training on Apple Silicon - Colab uses NVIDIA GPUs, so this won't be an issue.

---

## Performance Tips

### Faster Training
- Use fewer epochs (50 instead of 100)
- Use smaller batch size (8 instead of 16)
- Use a smaller dataset

### Better Accuracy
- Train for more epochs (150-200)
- Use larger model (yolov8s.pt or yolov8m.pt)
- Add more diverse training data
- Use data augmentation (already enabled)

### Save GPU Hours
- Test with 10 epochs first to ensure everything works
- Close the tab when not training to free up resources
- Use Colab Pro if you need more GPU time

---

## Understanding Results

### Key Metrics

- **mAP50**: Mean Average Precision at 50% IoU
  - Higher is better (0.0 to 1.0)
  - 0.5+ is decent, 0.7+ is good, 0.9+ is excellent

- **mAP50-95**: Mean Average Precision at 50-95% IoU
  - More strict metric
  - Usually lower than mAP50

- **Precision**: How many detections were correct
  - High precision = few false positives

- **Recall**: How many actual objects were detected
  - High recall = few false negatives

### Training Curves

Check `results.png` for:
- **Decreasing loss** - Model is learning
- **Increasing mAP** - Model is improving
- **Stable after ~50 epochs** - Model has converged

---

## Next Steps

After training:

1. ✅ Test the model locally with your detector
2. ✅ Collect more training data for weak classes
3. ✅ Fine-tune with domain-specific images
4. ✅ Share your trained model with the community

---

## FAQ

**Q: How long does Colab let me use the GPU?**
A: Free tier typically allows 12-15 hours per session, with a weekly limit.

**Q: Can I use Colab for commercial projects?**
A: Free tier is for educational/research use. Consider Colab Pro for commercial use.

**Q: What if training gets interrupted?**
A: Checkpoints are saved every 10 epochs. You can resume by loading `last.pt`.

**Q: Can I train on multiple GPUs?**
A: Free tier provides 1 GPU. Colab Pro+ offers more resources.

**Q: How do I resume training?**
A: Change the model path in Cell 6 from `'yolov8n.pt'` to `'path/to/last.pt'`.

---

## Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Roboflow Universe](https://universe.roboflow.com/)
- [Project Repository](https://github.com/GAM3RG33K/electronics_detector)

---

**Happy Training! 🚀**

If you encounter issues not covered here, please open an issue on GitHub.

