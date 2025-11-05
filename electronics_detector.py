"""
Real-time Electronics Detection System
Detects electronic devices in camera feed with custom overlays and information display
"""

import cv2
import numpy as np
import sys
import platform
import subprocess
from ultralytics import YOLO
import time
from collections import defaultdict
import os

class SystemChecker:
    """Pre-flight system checks before running detector"""
    
    def __init__(self):
        self.platform = platform.system()
        self.platform_release = platform.release()
        self.python_version = sys.version_info
        self.issues = []
        self.warnings = []
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)
    
    def print_check(self, name, status, message=""):
        """Print check result"""
        symbol = "✓" if status else "✗"
        color_code = "\033[92m" if status else "\033[91m"
        reset_code = "\033[0m"
        
        status_text = f"{color_code}{symbol}{reset_code} {name}"
        if message:
            print(f"{status_text}: {message}")
        else:
            print(status_text)
        
        return status
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        self.print_header("Python Version Check")
        
        major, minor = self.python_version.major, self.python_version.minor
        version_str = f"{major}.{minor}.{self.python_version.micro}"
        
        if major == 3 and minor >= 8:
            self.print_check("Python Version", True, f"{version_str} (Compatible)")
            return True
        else:
            self.print_check("Python Version", False, f"{version_str} (Requires 3.8+)")
            self.issues.append("Python 3.8 or higher is required")
            return False
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        self.print_header("Dependencies Check")
        
        all_good = True
        required = {
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'torch': 'torch',
            'ultralytics': 'ultralytics'
        }
        
        for module, package in required.items():
            try:
                if module == 'cv2':
                    import cv2
                    version = cv2.__version__
                elif module == 'numpy':
                    import numpy as np
                    version = np.__version__
                elif module == 'torch':
                    import torch
                    version = torch.__version__
                elif module == 'ultralytics':
                    import ultralytics
                    version = ultralytics.__version__
                
                self.print_check(f"{package}", True, f"v{version}")
            except ImportError:
                self.print_check(f"{package}", False, "Not installed")
                self.issues.append(f"Missing package: {package}")
                all_good = False
        
        return all_good
    
    def check_numpy_compatibility(self):
        """Check NumPy version compatibility with PyTorch"""
        self.print_header("NumPy Compatibility Check")
        
        try:
            import numpy as np
            import torch
            
            numpy_version = np.__version__
            numpy_major = int(numpy_version.split('.')[0])
            
            if numpy_major >= 2:
                self.print_check("NumPy Compatibility", False, 
                                f"v{numpy_version} (PyTorch needs NumPy 1.x)")
                self.issues.append("NumPy 2.x is incompatible with current PyTorch")
                self.issues.append("Run: pip uninstall numpy -y && pip install 'numpy<2'")
                return False
            else:
                self.print_check("NumPy Compatibility", True, f"v{numpy_version}")
                return True
        except Exception as e:
            self.print_check("NumPy Compatibility", False, str(e))
            return False
    
    def check_camera_availability(self):
        """Check if camera is available"""
        self.print_header("Camera Availability Check")
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.print_check("Camera Access", False, "Cannot open camera")
            
            if self.platform == "Darwin":  # macOS
                self.issues.append("Camera permission denied (macOS detected)")
                return False
            else:
                self.issues.append("Camera not accessible")
                return False
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            self.print_check("Camera Access", False, "Cannot read frames")
            self.issues.append("Camera hardware issue or busy")
            return False
        
        self.print_check("Camera Access", True, "Camera is accessible")
        return True
    
    def check_camera_permissions_macos(self):
        """Check and provide instructions for macOS camera permissions"""
        if self.platform != "Darwin":
            return True
        
        self.print_header("macOS Camera Permissions")
        
        # Try to detect if permissions are granted
        cap = cv2.VideoCapture(0)
        is_open = cap.isOpened()
        
        if is_open:
            ret, _ = cap.read()
            cap.release()
            if ret:
                self.print_check("Camera Permissions", True, "Access granted")
                return True
        
        cap.release()
        
        # Permission issue detected
        self.print_check("Camera Permissions", False, "Access denied")
        
        print("\n📋 MANUAL STEPS REQUIRED (macOS):")
        print("\n  Option 1 - Reset Camera Permissions (Recommended):")
        print("    1. Open Terminal")
        print("    2. Run: tccutil reset Camera")
        print("    3. Run this script again")
        print("    4. Grant permission when prompted")
        
        print("\n  Option 2 - Grant Permission Manually:")
        print("    1. Open System Preferences")
        print("    2. Go to: Security & Privacy → Privacy → Camera")
        print("    3. Find and enable:")
        
        # Detect what's running the script
        if 'TERM_PROGRAM' in os.environ:
            term_program = os.environ.get('TERM_PROGRAM', 'Terminal')
            print(f"       - {term_program}")
        else:
            print("       - Terminal")
            print("       - Python")
        
        print("\n  Option 3 - Automated Reset (will ask for password):")
        response = input("\n  Would you like to reset camera permissions now? (y/n): ").lower().strip()
        
        if response == 'y':
            try:
                print("\n  Attempting to reset camera permissions...")
                result = subprocess.run(['tccutil', 'reset', 'Camera'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("  ✓ Permissions reset successfully!")
                    print("  ℹ️  Please run this script again and grant permission when prompted.")
                    return False
                else:
                    print(f"  ✗ Failed to reset: {result.stderr}")
                    self.issues.append("Manual permission reset required")
                    return False
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.issues.append("Manual permission reset required")
                return False
        else:
            self.issues.append("Camera permissions need to be granted manually")
            return False
    
    def check_model_file(self):
        """Check if YOLO model exists"""
        self.print_header("YOLO Model Check")
        
        model_path = 'yolov8n.pt'
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            self.print_check("Model File", True, f"{model_path} ({size_mb:.1f}MB)")
            return True
        else:
            self.print_check("Model File", False, "Will download on first run (~6MB)")
            self.warnings.append("YOLOv8n model will be downloaded automatically")
            return True  # Not a critical issue
    
    def run_all_checks(self):
        """Run all pre-flight checks"""
        print("\n" + "🔍 "*20)
        print("  PRE-FLIGHT SYSTEM CHECKS")
        print("🔍 "*20)
        
        checks = [
            self.check_python_version(),
            self.check_dependencies(),
            self.check_numpy_compatibility(),
            self.check_model_file(),
        ]
        
        # Platform-specific checks
        if self.platform == "Darwin":
            checks.append(self.check_camera_permissions_macos())
        else:
            checks.append(self.check_camera_availability())
        
        # Summary
        self.print_header("Pre-Flight Summary")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.issues:
            print("\n❌ Issues Found:")
            for issue in self.issues:
                print(f"  - {issue}")
            print("\n❌ Cannot proceed until all issues are resolved.\n")
            return False
        
        if not all(checks):
            print("\n❌ Some checks failed. Please resolve issues above.\n")
            return False
        
        print("\n✅ All checks passed! Ready to start detector.\n")
        return True
    
    def print_system_info(self):
        """Print detailed system information"""
        self.print_header("System Information")
        print(f"  Platform: {self.platform} {self.platform_release}")
        print(f"  Python: {sys.version.split()[0]}")
        
        try:
            import cv2
            print(f"  OpenCV: {cv2.__version__}")
        except:
            print(f"  OpenCV: Not installed")
        
        try:
            import numpy as np
            print(f"  NumPy: {np.__version__}")
        except:
            print(f"  NumPy: Not installed")
        
        try:
            import torch
            print(f"  PyTorch: {torch.__version__}")
            print(f"  CUDA Available: {torch.cuda.is_available()}")
        except:
            print(f"  PyTorch: Not installed")


class ElectronicsDetector:
    def __init__(self):
        print("\n📦 Initializing Electronics Detector...")
        
        # Load YOLOv8 model (will auto-download on first run)
        print("  Loading YOLO model...")
        try:
            self.model = YOLO('yolov8n.pt')  # Nano model for speed
            print("  ✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load YOLO model: {e}")
            raise
        
        # Travel electronic devices we care about and their properties
        # Focus: Portable devices travelers carry to airports/train stations
        self.electronics_map = {
            # Personal Devices
            'cell phone': {'color': (100, 255, 100), 'icon': '📱', 'power': '5-10W', 'type': 'Mobile Device', 'prevalence': '99%'},
            'laptop': {'color': (255, 100, 100), 'icon': '💻', 'power': '45-65W', 'type': 'Computer', 'prevalence': '70%'},
            
            # Accessories
            'keyboard': {'color': (255, 255, 100), 'icon': '⌨️', 'power': '2-5W', 'type': 'Peripheral', 'prevalence': '25%'},
            'mouse': {'color': (255, 100, 255), 'icon': '🖱️', 'power': '1-3W', 'type': 'Peripheral', 'prevalence': '40%'},
            'remote': {'color': (100, 255, 255), 'icon': '📡', 'power': '0.5W', 'type': 'Controller', 'prevalence': '10%'},
            
            # Personal Care (Travel)
            'hair drier': {'color': (255, 150, 200), 'icon': '💇', 'power': '800-1200W', 'type': 'Personal Care', 'prevalence': '30%'},
        }
        
        # Note: Non-travel items removed (TV, Microwave, Oven, Toaster, Refrigerator)
        # Next phase: Add tablet, earbuds, power bank, charger, headphones, smartwatch
        
        print(f"  ✓ Configured to detect {len(self.electronics_map)} device types")
        
        # Detection zone (relative to frame size, 0.0 to 1.0)
        # Default: center 80% of the frame
        self.zone = {
            'x1': 0.1,  # 10% from left
            'y1': 0.1,  # 10% from top
            'x2': 0.9,  # 90% from left (80% width)
            'y2': 0.9   # 90% from top (80% height)
        }
        
        self.fps_history = []
        self.max_fps_samples = 30
        
        # Performance settings for 60fps display
        self.detection_interval = 2  # Run YOLO every N frames (higher = faster display, less frequent detection)
        self.last_detections = []  # Cache last detection results
        
        print("  ✓ Initialization complete!")
        print(f"  ⚡ Performance: Running detection every {self.detection_interval} frames for 60fps display\n")
        
    def point_in_zone(self, x, y, frame_width, frame_height):
        """Check if a point is within the detection zone"""
        zone_x1 = int(self.zone['x1'] * frame_width)
        zone_y1 = int(self.zone['y1'] * frame_height)
        zone_x2 = int(self.zone['x2'] * frame_width)
        zone_y2 = int(self.zone['y2'] * frame_height)
        
        return zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2
    
    def draw_detection_zone(self, frame):
        """Draw the detection zone boundary"""
        height, width = frame.shape[:2]
        x1 = int(self.zone['x1'] * width)
        y1 = int(self.zone['y1'] * height)
        x2 = int(self.zone['x2'] * width)
        y2 = int(self.zone['y2'] * height)
        
        # Draw semi-transparent overlay outside zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw zone border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "DETECTION ZONE", (x1 + 10, y1 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def draw_custom_overlay(self, frame, box, label, confidence):
        """Draw custom overlay for detected electronics"""
        x1, y1, x2, y2 = map(int, box)
        
        if label not in self.electronics_map:
            return
        
        props = self.electronics_map[label]
        color = props['color']
        
        # Draw rounded rectangle outline
        thickness = 3
        corner_length = 20
        
        # Corners
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
        
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
        
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
        
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
        
        # Info panel background
        panel_height = 90
        panel_y = max(y1 - panel_height - 10, 0)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, panel_y), (x1 + 250, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border for info panel
        cv2.rectangle(frame, (x1, panel_y), (x1 + 250, panel_y + panel_height), color, 2)
        
        # Text information
        text_x = x1 + 10
        text_y = panel_y + 25
        
        cv2.putText(frame, f"{label.upper()}", (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Type: {props['type']}", (text_x, text_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Power: {props['power']}", (text_x, text_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (text_x, text_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Center dot
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        cv2.circle(frame, (center_x, center_y), 8, color, 2)
    
    def run(self):
        """Main detection loop"""
        print("🎥 Opening camera...")
        
        # Open camera (0 is usually the default camera)
        cap = cv2.VideoCapture(0)
        
        # Try to set high quality settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Get actual camera settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"  ✓ Camera: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        if not cap.isOpened():
            print("  ✗ Error: Could not open camera")
            return
        
        print("\n" + "="*60)
        print("  🎮 CONTROLS")
        print("="*60)
        print("  Q       - Quit")
        print("  Z       - Toggle detection zone visibility")
        print("  + / =   - Expand detection zone")
        print("  -       - Shrink detection zone")
        print("  D       - Toggle detection interval (speed vs accuracy)")
        print("="*60)
        print("\n🚀 Starting detection loop...\n")
        
        show_zone = True
        frame_count = 0
        last_key_press = ""
        key_press_time = 0
        
        while True:
            start_time = time.time()
            frame_count += 1
            
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Could not read frame")
                break
            
            height, width = frame.shape[:2]
            
            # Run YOLO detection only every N frames (for 60fps display)
            detected_items = defaultdict(int)
            
            if frame_count % self.detection_interval == 0:
                # Run YOLO detection with lower confidence for better detection
                results = self.model(frame, conf=0.3, verbose=False)
                
                # Cache detection results
                self.last_detections = []
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Check if center is in detection zone
                        if not self.point_in_zone(center_x, center_y, width, height):
                            continue
                        
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        label = self.model.names[class_id]
                        confidence = float(box.conf[0]) * 100
                        
                        # Only cache electronics
                        if label in self.electronics_map:
                            self.last_detections.append({
                                'box': [x1, y1, x2, y2],
                                'label': label,
                                'confidence': confidence
                            })
            
            # Draw detection zone
            if show_zone:
                frame = self.draw_detection_zone(frame)
            
            # Draw cached detections (from last YOLO run)
            for detection in self.last_detections:
                detected_items[detection['label']] += 1
                self.draw_custom_overlay(frame, detection['box'], detection['label'], detection['confidence'])
            
            # Calculate FPS
            frame_time = time.time() - start_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(fps)
            if len(self.fps_history) > self.max_fps_samples:
                self.fps_history.pop(0)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            # Draw FPS and stats
            panel_height = 130 if last_key_press else 100
            cv2.rectangle(frame, (10, 10), (300, 10 + panel_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, 10 + panel_height), (0, 255, 0), 2)
            
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Detected: {sum(detected_items.values())} items", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Detection: Every {self.detection_interval} frames", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Show last key press feedback (fades after 2 seconds)
            if last_key_press and time.time() - key_press_time < 2:
                cv2.putText(frame, f"Key: {last_key_press}", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            
            # Display frame
            cv2.imshow('Electronics Detector', frame)
            
            # Handle keyboard input (wait 1ms for key)
            key = cv2.waitKey(1) & 0xFF
            
            if key != 255:  # A key was pressed (255 means no key)
                if key == ord('q') or key == ord('Q'):
                    last_key_press = "Q - Quit"
                    key_press_time = time.time()
                    print("\n[INFO] Quit requested by user")
                    break
                    
                elif key == ord('z') or key == ord('Z'):
                    show_zone = not show_zone
                    last_key_press = f"Z - Zone {'ON' if show_zone else 'OFF'}"
                    key_press_time = time.time()
                    print(f"[INFO] Detection zone: {'ON' if show_zone else 'OFF'}")
                    
                elif key == ord('+') or key == ord('='):
                    # Expand zone
                    margin = 0.05
                    self.zone['x1'] = max(0, self.zone['x1'] - margin)
                    self.zone['y1'] = max(0, self.zone['y1'] - margin)
                    self.zone['x2'] = min(1, self.zone['x2'] + margin)
                    self.zone['y2'] = min(1, self.zone['y2'] + margin)
                    last_key_press = "+ - Zone Expanded"
                    key_press_time = time.time()
                    print(f"[INFO] Zone expanded")
                    
                elif key == ord('-') or key == ord('_'):
                    # Shrink zone
                    margin = 0.05
                    if self.zone['x2'] - self.zone['x1'] > 0.2:  # Keep minimum size
                        self.zone['x1'] += margin
                        self.zone['y1'] += margin
                        self.zone['x2'] -= margin
                        self.zone['y2'] -= margin
                        last_key_press = "- - Zone Shrunk"
                        key_press_time = time.time()
                        print(f"[INFO] Zone shrunk")
                    else:
                        last_key_press = "- - Zone at minimum"
                        key_press_time = time.time()
                        print(f"[INFO] Zone already at minimum size")
                        
                elif key == ord('d') or key == ord('D'):
                    # Toggle detection interval (1, 2, 3 frames)
                    self.detection_interval = (self.detection_interval % 3) + 1
                    speed_desc = {1: "High accuracy", 2: "Balanced", 3: "High speed"}
                    last_key_press = f"D - {speed_desc[self.detection_interval]}"
                    key_press_time = time.time()
                    print(f"[INFO] Detection interval: Every {self.detection_interval} frames ({speed_desc[self.detection_interval]})")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Detection stopped cleanly.\n")


if __name__ == "__main__":
    print("\n" + "✈️ "*20)
    print("  TRAVEL ELECTRONICS ENERGY FOOTPRINT DETECTOR")
    print("  Airport & Train Station Deployment")
    print("✈️ "*20)
    
    try:
        # Step 1: Run system checks
        checker = SystemChecker()
        checker.print_system_info()
        
        if not checker.run_all_checks():
            print("⚠️  System checks failed. Please resolve issues and try again.\n")
            sys.exit(1)
        
        # Step 2: Initialize detector
        detector = ElectronicsDetector()
        
        # Step 3: Run detection
        detector.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        print("✓ Exiting gracefully...\n")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print("\n📋 Stack trace:")
        import traceback
        traceback.print_exc()
        print("\n💡 If this error persists, please check:")
        print("  1. All dependencies are installed correctly")
        print("  2. Camera is not being used by another application")
        print("  3. You have proper permissions")
        sys.exit(1)
