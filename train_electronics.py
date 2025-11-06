"""
Training script for electronics detection using Roboflow dataset
Trains YOLOv8 model offline on local dataset
"""

from ultralytics import YOLO
import os
from pathlib import Path
import argparse
import sys

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model on electronics dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_electronics.py --dataset /path/to/electronics-j0cxl
  python train_electronics.py --dataset ~/Downloads/electronics-j0cxl --epochs 50
  python train_electronics.py --dataset ./data --batch 8 --model yolov8s.pt
  python train_electronics.py --dataset ./data --device mps --no-val  # MPS without validation
        """
    )
    
    # Required argument
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset folder containing data.yaml'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='Base model to use (default: yolov8n.pt)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16, reduce if out of memory)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size (default: 640)'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='electronics_training',
        help='Project folder name (default: electronics_training)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='roboflow_v1',
        help='Experiment name (default: roboflow_v1)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use: 0, 1, 2, etc. for GPU or "cpu" for CPU-only (default: 0)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker threads (default: 8)'
    )
    
    parser.add_argument(
        '--val',
        action='store_true',
        default=True,
        help='Run validation during training (default: True, use --no-val to disable)'
    )
    
    parser.add_argument(
        '--no-val',
        dest='val',
        action='store_false',
        help='Disable validation during training (useful for MPS device compatibility)'
    )
    
    return parser.parse_args()

def train_model(args):
    """Train YOLOv8 model on electronics dataset"""
    
    # Setup paths
    dataset_path = os.path.expanduser(args.dataset)  # Expand ~ if present
    data_yaml = os.path.join(dataset_path, "data.yaml")
    
    print("="*60)
    print("🚀 Starting Electronics Detection Training")
    print("="*60)
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset folder not found at {dataset_path}")
        sys.exit(1)
    
    if not os.path.exists(data_yaml):
        print(f"❌ ERROR: data.yaml not found at {data_yaml}")
        print("Please ensure the dataset folder contains data.yaml")
        sys.exit(1)
    
    print(f"\n📁 Dataset: {dataset_path}")
    print(f"📊 Config: {data_yaml}")
    print(f"🎯 Base Model: {args.model}")
    print(f"⚙️  Epochs: {args.epochs}")
    print(f"📐 Image Size: {args.imgsz}")
    print(f"📦 Batch Size: {args.batch}")
    print(f"💻 Device: {args.device}")
    print(f"👷 Workers: {args.workers}")
    print(f"✅ Validation: {'Enabled' if args.val else 'Disabled (--no-val)'}")
    
    # Load pre-trained model (transfer learning)
    print(f"\n🔄 Loading pre-trained {args.model}...")
    model = YOLO(args.model)
    
    # Train the model
    print("\n🏋️  Starting training...")
    print("This may take 1-4 hours depending on your hardware\n")
    
    # Convert device string to appropriate format
    device_str = args.device.lower()
    if device_str in ['cpu', 'mps', 'cuda']:
        device = device_str
    else:
        device = int(args.device)  # GPU index like 0, 1, 2, etc.
    
    results = model.train(
        data=data_yaml,              # Path to data.yaml
        epochs=args.epochs,          # Number of epochs
        imgsz=args.imgsz,            # Image size
        batch=args.batch,            # Batch size
        name=args.name,              # Experiment name
        project=args.project,        # Project folder name
        
        # Performance settings
        patience=20,                 # Early stopping patience
        save=True,                   # Save checkpoints
        save_period=10,              # Save every N epochs
        
        # Optimization
        optimizer='auto',            # Auto-select optimizer
        lr0=0.01,                    # Initial learning rate
        lrf=0.01,                    # Final learning rate
        
        # Augmentation (helps model generalize)
        hsv_h=0.015,                # HSV-Hue augmentation
        hsv_s=0.7,                  # HSV-Saturation augmentation
        hsv_v=0.4,                  # HSV-Value augmentation
        degrees=0.0,                # Rotation (+/- deg)
        translate=0.1,              # Translation (+/- fraction)
        scale=0.5,                  # Scaling (+/- gain)
        shear=0.0,                  # Shear (+/- deg)
        perspective=0.0,            # Perspective (+/- fraction)
        flipud=0.0,                 # Flip up-down (probability)
        fliplr=0.5,                 # Flip left-right (probability)
        mosaic=1.0,                 # Mosaic augmentation (probability)
        
        # Device
        device=device,              # Use specified device
        
        # Validation
        val=args.val,               # Run validation (disable with --no-val for MPS compatibility)
        
        # Other
        workers=args.workers,       # Number of worker threads
        exist_ok=True,              # Overwrite existing project
        pretrained=True,            # Use pretrained weights
        verbose=True,               # Verbose output
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    
    # Print results location
    save_dir = Path(args.project) / args.name
    print(f"\n📊 Results saved to: {save_dir}")
    print(f"🎯 Best weights: {save_dir}/weights/best.pt")
    print(f"🔄 Last weights: {save_dir}/weights/last.pt")
    
    # Validation
    print("\n🔍 Running validation on test set...")
    metrics = model.val()
    
    print(f"\n📈 Performance Metrics:")
    print(f"   mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    
    return save_dir, dataset_path

def test_trained_model(weights_path, dataset_path):
    """Test the trained model on sample images"""
    
    print("\n" + "="*60)
    print("🧪 Testing Trained Model")
    print("="*60)
    
    model = YOLO(weights_path)
    
    # Test on validation images
    test_images_path = os.path.join(dataset_path, "valid/images")
    
    if os.path.exists(test_images_path):
        # Try different image extensions
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            test_images.extend(list(Path(test_images_path).glob(ext)))
        
        test_images = test_images[:5]  # Test first 5 images
        
        if not test_images:
            print("⚠️  No test images found in valid/images folder")
            return
        
        print(f"\n📸 Testing on {len(test_images)} sample images...")
        
        for img_path in test_images:
            print(f"   Testing: {img_path.name}")
            results = model(str(img_path))
            
            # Print detections
            for result in results:
                boxes = result.boxes
                print(f"      Detected {len(boxes)} objects")
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    print(f"        - {name}: {conf:.2%}")
    else:
        print(f"⚠️  Test images path not found: {test_images_path}")
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Train the model
    save_dir, dataset_path = train_model(args)
    
    # Test the trained model
    best_weights = save_dir / "weights/best.pt"
    if best_weights.exists():
        test_trained_model(str(best_weights), dataset_path)
    
    print("\n" + "="*60)
    print("🎉 All Done! Your model is ready to use.")
    print("="*60)
    print(f"\nTo use your trained model, replace 'yolov8n.pt' with:")
    print(f"  '{best_weights}'")
    print("\nIn electronics_detector.py, update the model path:")
    print(f"  self.model = YOLO('{best_weights}')")