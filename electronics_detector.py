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

yolo_model_name = 'yolo11n-seg'
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
        
        model_path = f"{yolo_model_name}.pt"
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            self.print_check("Model File", True, f"{model_path} ({size_mb:.1f}MB)")
            return True
        else:
            self.print_check("Model File", False, "Will download on first run (~6MB)")
            self.warnings.append("YOLOv model will be downloaded automatically")
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
            self.model = YOLO(f"{yolo_model_name}.pt")  # Nano segmentation model for precise masks
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
            'phone': {'color': (120, 255, 120), 'icon': '📞', 'power': '3-8W', 'type': 'Mobile Device', 'prevalence': '95%'},
            'ipad': {'color': (200, 150, 255), 'icon': '📱', 'power': '8-15W', 'type': 'Tablet', 'prevalence': '40%'},
            
            # Accessories
            'keyboard': {'color': (255, 255, 100), 'icon': '⌨️', 'power': '2-5W', 'type': 'Peripheral', 'prevalence': '25%'},
            'mouse': {'color': (255, 100, 255), 'icon': '🖱️', 'power': '1-3W', 'type': 'Peripheral', 'prevalence': '40%'},
            'remote': {'color': (100, 255, 255), 'icon': '📡', 'power': '0.5W', 'type': 'Controller', 'prevalence': '10%'},
            'monitor': {'color': (150, 150, 150), 'icon': '🖥️', 'power': '15-30W', 'type': 'Display', 'prevalence': '20%'},
            
            # Audio Devices
            'headphones': {'color': (100, 100, 255), 'icon': '🎧', 'power': '1-5W', 'type': 'Audio', 'prevalence': '45%'},
            'head_phone': {'color': (120, 120, 255), 'icon': '🎵', 'power': '1-3W', 'type': 'Audio', 'prevalence': '35%'},
            'earphone': {'color': (150, 150, 255), 'icon': '🎶', 'power': '0.5-2W', 'type': 'Audio', 'prevalence': '50%'},
            'ear_piece': {'color': (180, 180, 255), 'icon': '📞', 'power': '1-3W', 'type': 'Audio', 'prevalence': '30%'},
            'airpods_buds': {'color': (200, 100, 200), 'icon': '🎧', 'power': '1-3W', 'type': 'Audio', 'prevalence': '60%'},
            'airpods_case': {'color': (220, 120, 220), 'icon': '🔋', 'power': '0-1W', 'type': 'Accessory', 'prevalence': '55%'},
            
            # Gaming Controllers
            'playstation_controller': {'color': (0, 100, 200), 'icon': '🎮', 'power': '5-8W', 'type': 'Gaming', 'prevalence': '15%'},
            'xbox_controller': {'color': (0, 150, 0), 'icon': '🕹️', 'power': '5-8W', 'type': 'Gaming', 'prevalence': '12%'},
            
            # Wearables
            'smartwatch': {'color': (255, 200, 100), 'icon': '⌚', 'power': '1-5W', 'type': 'Wearable', 'prevalence': '35%'},
            
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
        
        # Heatmap configuration
        self.conf_threshold = 0.30  # Configurable confidence threshold (70%)
        self.heatmap_alpha = 0.7    # 70% transparency for heatmap overlay (brighter devices)
        self.mask_method = 'rounded'  # 'rounded' or 'grabcut'
        self.gradient_cache = {}    # Cache generated gradients for performance
        self.show_heatmap = True    # Toggle heatmap vs boxes
        self.show_thermal_filter = True  # Toggle mock thermal filter overlay on entire frame
        self.dark_tint = 0.3        # Background darkness level (0.3 = 70% darker) - PERMANENT
        self.visualization_mode = 'anchor'  # 'anchor' (default pulsing) or 'mask' (segmentation)
        
        print("  ✓ Initialization complete!")
        print(f"  ⚡ Performance: Running detection every {self.detection_interval} frames for 60fps display")
        print(f"  🎯 Visualization: Pulsing Anchor Mode (default)")
        print(f"  🔍 Confidence threshold: {self.conf_threshold*100:.0f}%")
        print(f"  💡 Press 'M' to toggle between Anchor and Segmentation Mask modes\n")
        
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
    
    def get_heatmap_colormap(self, power_str):
        """
        Get color gradient scheme based on power consumption
        All colors are in warm spectrum (green → yellow → orange → red)
        Returns dictionary with colors array
        Colors are in BGR format for OpenCV
        """
        # Extract average power from range like "5-10W"
        try:
            if '-' in power_str:
                low, high = power_str.replace('W', '').split('-')
                avg_power = (float(low) + float(high)) / 2
            else:
                avg_power = float(power_str.replace('W', ''))
        except:
            avg_power = 5.0  # Default to low power
        
        # All colors in warm spectrum with varying intensity
        if avg_power <= 10:  # Very Low: Light Green → Yellow-Green (dim intensity)
            return {
                'colors': [
                    (0, 100, 50),      # Dim Green (edges)
                    (0, 150, 100),     # Light Green
                    (0, 200, 150),     # Yellow-Green
                    (0, 255, 200),     # Bright Yellow-Green (center)
                ],
                'intensity': 0.6    # Lower intensity for low power
            }
        elif avg_power <= 25:  # Low-Medium: Yellow-Green → Yellow (medium intensity)
            return {
                'colors': [
                    (0, 150, 100),     # Yellow-Green (edges)
                    (0, 200, 200),     # Yellow (edges)
                    (0, 220, 240),     # Yellow-Orange
                    (0, 180, 255),     # Orange (center)
                ],
                'intensity': 0.75   # Medium intensity
            }
        elif avg_power <= 50:  # Medium: Yellow → Orange (high intensity)
            return {
                'colors': [
                    (0, 200, 200),     # Yellow (edges)
                    (0, 220, 240),     # Yellow-Orange
                    (0, 180, 255),     # Orange
                    (0, 200, 255),     # Bright Orange (center)
                ],
                'intensity': 0.85   # High intensity
            }
        else:  # High: Orange → Red → Bright Red (maximum intensity)
            return {
                'colors': [
                    (0, 100, 200),     # Orange (edges)
                    (0, 50, 255),      # Orange-Red
                    (0, 0, 255),       # Red
                    (50, 50, 255),     # Bright Red (center)
                ],
                'intensity': 1.0    # Maximum intensity
            }
    
    def create_rounded_rectangle_mask(self, shape, corner_radius=20):
        """
        Create smooth rounded rectangle mask
        Fast method for device shape approximation
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create rounded rectangle using circles at corners
        cv2.rectangle(mask, (corner_radius, 0), (w-corner_radius, h), 255, -1)
        cv2.rectangle(mask, (0, corner_radius), (w, h-corner_radius), 255, -1)
        
        # Add rounded corners
        if corner_radius > 0:
            cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 255, -1)
            cv2.circle(mask, (w-corner_radius, corner_radius), corner_radius, 255, -1)
            cv2.circle(mask, (corner_radius, h-corner_radius), corner_radius, 255, -1)
            cv2.circle(mask, (w-corner_radius, h-corner_radius), corner_radius, 255, -1)
        
        return mask
    
    def create_radial_gradient_heatmap_fast(self, mask_shape, colormap_info):
        """
        Create radial gradient heatmap with thermal camera-style effect
        Heat radiates from center (hottest) to edges (coolest)
        Uses distance transform for performance
        """
        h, w = mask_shape
        
        # Create center point
        center_mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        cv2.circle(center_mask, (center_x, center_y), 2, 255, -1)
        
        # Distance transform: distance from center point
        dist_transform = cv2.distanceTransform(
            255 - center_mask, 
            cv2.DIST_L2, 
            5
        )
        
        # Normalize to 0-255 range
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
        
        # Invert so center is hot (255) and edges are cool (0)
        intensity_map = 255 - dist_transform.astype(np.uint8)
        
        # Apply power curve for realistic heat falloff (more concentrated at center)
        intensity_map = np.power(intensity_map / 255.0, 1.5) * 255
        intensity_map = intensity_map.astype(np.uint8)
        
        # Create color gradient using Look-Up Table
        colors = colormap_info['colors']
        num_colors = len(colors)
        
        # Build LUT for 256 intensity values
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            # Map intensity to color position with interpolation
            pos = (i / 255.0) * (num_colors - 1)
            idx_low = int(np.floor(pos))
            idx_high = min(idx_low + 1, num_colors - 1)
            blend = pos - idx_low
            
            # Interpolate between two colors
            color_low = np.array(colors[idx_low])
            color_high = np.array(colors[idx_high])
            interpolated = color_low * (1 - blend) + color_high * blend
            
            lut[i, 0] = interpolated.astype(np.uint8)
        
        # Apply color mapping
        heatmap = cv2.LUT(cv2.merge([intensity_map, intensity_map, intensity_map]), lut)
        
        return heatmap, intensity_map
    
    def add_glow_effect(self, device_roi, mask, colormap_info):
        """Add subtle glow around high-power device edges"""
        # Dilate mask slightly for glow region
        kernel = np.ones((5, 5), np.uint8)
        glow_mask = cv2.dilate(mask, kernel, iterations=1)

        # Get glow region (dilated - original)
        glow_region = cv2.subtract(glow_mask, mask)

        # Create glow color (brightest color from gradient)
        glow_color = colormap_info['colors'][-1]  # Hottest color
        glow_overlay = np.zeros_like(device_roi)
        glow_overlay[:] = glow_color

        # Apply glow with transparency
        glow_mask_3ch = cv2.merge([glow_region, glow_region, glow_region])
        glow_overlay = cv2.bitwise_and(glow_overlay, glow_mask_3ch)

        device_roi = cv2.addWeighted(device_roi, 1.0, glow_overlay, 0.3, 0)

        return device_roi

    def apply_mock_thermal_filter(self, frame):
        """
        Apply an authentic thermal camera filter effect to the entire frame
        Uses professional thermal imaging color scheme with enhanced contrast
        """
        # Convert to grayscale for intensity mapping
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to preserve edges while smoothing
        smoothed = cv2.bilateralFilter(gray, 9, 75, 75)

        # Use CLAHE for enhanced local contrast (like real thermal cameras)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)

        # Further contrast boost
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=10)

        # Create authentic thermal color mapping using a lookup table
        # Professional thermal scheme: black → deep blue → purple → orange → yellow → white
        lut = np.zeros((256, 1, 3), dtype=np.uint8)

        for i in range(256):
            intensity = i / 255.0

            if intensity < 0.2:  # Very cold (black to deep blue)
                # Black to deep blue
                t = intensity / 0.2
                r = int(t * 20)
                g = int(t * 15)
                b = int(t * 120)
            elif intensity < 0.4:  # Cold (deep blue to purple)
                # Deep blue to purple
                t = (intensity - 0.2) / 0.2
                r = int(20 + t * 80)
                g = int(15 + t * 25)
                b = int(120 + t * 60)
            elif intensity < 0.6:  # Moderate (purple to orange)
                # Purple to orange transition
                t = (intensity - 0.4) / 0.2
                r = int(100 + t * 120)
                g = int(40 + t * 60)
                b = int(180 - t * 150)
            elif intensity < 0.8:  # Warm (orange to yellow)
                # Orange to yellow
                t = (intensity - 0.6) / 0.2
                r = int(220 + t * 35)
                g = int(100 + t * 130)
                b = int(30 - t * 30)
            else:  # Hot (yellow to white)
                # Yellow to white
                t = (intensity - 0.8) / 0.2
                r = int(255)
                g = int(230 + t * 25)
                b = int(t * 200)

            # Clamp values to 0-255
            r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
            lut[i, 0] = [b, g, r]  # BGR format for OpenCV

        # Apply the thermal color mapping
        thermal_frame = cv2.LUT(cv2.merge([enhanced, enhanced, enhanced]), lut)

        # Blend with original frame for authentic thermal effect (65% thermal, 35% original)
        thermal_overlay = cv2.addWeighted(frame, 0.35, thermal_frame, 0.65, 0)

        return thermal_overlay

    def extract_segmentation_mask(self, seg_mask, box):
        """
        Extract and process segmentation mask from YOLO results
        Returns a clean binary mask for the detected device with precise boundaries
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = y2 - y1, x2 - x1
        
        if seg_mask is None or seg_mask.shape[0] == 0:
            # Fallback to rounded rectangle if no segmentation available
            corner_radius = min(15, min(h, w) // 4)
            return self.create_rounded_rectangle_mask((h, w), corner_radius=corner_radius)
        
        # YOLO segmentation mask is in full frame resolution
        # First crop to bounding box region, then resize if needed
        mask_h, mask_w = seg_mask.shape
        
        # Crop mask to bounding box region (with bounds checking)
        y1_crop = max(0, min(y1, mask_h))
        y2_crop = max(0, min(y2, mask_h))
        x1_crop = max(0, min(x1, mask_w))
        x2_crop = max(0, min(x2, mask_w))
        
        # Validate crop region is not empty
        if y2_crop <= y1_crop or x2_crop <= x1_crop:
            # Invalid crop region, fall back to rounded rectangle
            corner_radius = min(15, min(h, w) // 4)
            return self.create_rounded_rectangle_mask((h, w), corner_radius=corner_radius)
        
        cropped_mask = seg_mask[y1_crop:y2_crop, x1_crop:x2_crop]
        
        # Validate cropped mask is not empty
        if cropped_mask.size == 0 or cropped_mask.shape[0] == 0 or cropped_mask.shape[1] == 0:
            # Empty mask, fall back to rounded rectangle
            corner_radius = min(15, min(h, w) // 4)
            return self.create_rounded_rectangle_mask((h, w), corner_radius=corner_radius)
        
        # Resize to exact box dimensions if needed
        if cropped_mask.shape[0] != h or cropped_mask.shape[1] != w:
            mask_resized = cv2.resize(cropped_mask, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            mask_resized = cropped_mask
        
        # Convert to binary mask with threshold for clean edges
        _, binary_mask = cv2.threshold(mask_resized, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        
        # Minimal morphological operations to preserve exact shape
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Light smoothing to preserve edges while removing noise
        binary_mask = cv2.GaussianBlur(binary_mask, (3, 3), 0)
        _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        
        return binary_mask

    def draw_pulsing_anchor_contour_constrained(self, frame, box, label, confidence, frame_count, seg_mask=None):
        """
        Enhance thermal camera filter to show device heat naturally
        Modifies existing thermal intensity instead of adding artificial overlays
        Device appears as a hot object within the thermal camera view
        """
        x1, y1, x2, y2 = map(int, box)

        # Skip if confidence below threshold
        if confidence < self.conf_threshold * 100:
            return frame

        if label not in self.electronics_map:
            return frame

        props = self.electronics_map[label]
        
        # Extract segmentation mask (full frame coordinates)
        device_mask = self.extract_segmentation_mask(seg_mask, box)
        
        # Create full-frame heat mask
        frame_h, frame_w = frame.shape[:2]
        heat_mask = np.zeros((frame_h, frame_w), dtype=np.float32)
        
        # Place device mask in full frame coordinates with proper bounds checking
        mask_h, mask_w = device_mask.shape
        
        # Calculate actual placement bounds
        end_y = min(y1 + mask_h, frame_h)
        end_x = min(x1 + mask_w, frame_w)
        actual_h = end_y - y1
        actual_w = end_x - x1
        
        # Only place mask if we have valid bounds
        if actual_h > 0 and actual_w > 0:
            # Crop mask to fit within frame bounds
            cropped_mask = device_mask[:actual_h, :actual_w]
            heat_mask[y1:end_y, x1:end_x] = cropped_mask / 255.0
        
        # Check if mask is valid
        mask_coords = np.argwhere(heat_mask > 0)
        if len(mask_coords) == 0:
            return frame
        
        # Calculate device center for diffusion reference
        center_y = int(np.mean(mask_coords[:, 0]))
        center_x = int(np.mean(mask_coords[:, 1]))
        
        # Create heat intensity map
        heat_intensity = np.zeros((frame_h, frame_w), dtype=np.float32)
        
        # === Make entire device uniformly hot ===
        # Device interior: uniform high temperature (no gradient)
        device_interior = heat_mask > 0
        
        # Power-based base temperature (entire device at this level)
        power_multiplier = {
            '0.5W': 0.7, '1-3W': 0.75, '1-5W': 0.8, '2-5W': 0.85,
            '3-8W': 0.9, '5-8W': 0.92, '5-10W': 0.95, '8-15W': 0.97,
            '15-30W': 1.0, '45-65W': 1.0, '800-1200W': 1.0
        }
        
        base_temp = power_multiplier.get(props['power'], 0.9)
        
        # Subtle pulse effect (5% variation)
        pulse = 1.0 + 0.05 * np.sin(frame_count * 0.08)
        device_temp = base_temp * pulse
        
        # Set entire device to uniform high temperature
        heat_intensity[device_interior] = device_temp
        
        # === Add subtle heat diffusion beyond device edges ===
        # Create distance transform from device edges
        device_mask_uint8 = (heat_mask * 255).astype(np.uint8)
        
        # Much smaller diffusion - just a subtle glow at edges
        diffusion_size = {
            '0.5W': 5, '1-3W': 8, '1-5W': 10, '2-5W': 12,
            '3-8W': 15, '5-8W': 18, '5-10W': 20, '8-15W': 22,
            '15-30W': 25, '45-65W': 30, '800-1200W': 35
        }
        
        diffusion_radius = diffusion_size.get(props['power'], 15)
        
        # Calculate distance from device edge for smooth falloff
        dist_from_device = cv2.distanceTransform(255 - device_mask_uint8, cv2.DIST_L2, 5)
        
        # Create very subtle falloff outside device (only very close to edges)
        outside_device = (dist_from_device > 0) & (dist_from_device <= diffusion_radius) & (device_mask_uint8 == 0)
        if np.any(outside_device):
            # Much stronger falloff (power 3.5 instead of 2.0)
            falloff = np.clip(1.0 - (dist_from_device / diffusion_radius), 0, 1)
            falloff = np.power(falloff, 3.5)  # Very sharp falloff
            # Much lower intensity outside device (20% instead of 60%)
            heat_intensity[outside_device] = falloff[outside_device] * device_temp * 0.2
        
        # Ensure intensity is in valid range
        heat_intensity = np.clip(heat_intensity, 0, 1)
        
        # === Modify existing frame thermal intensity ===
        # Convert frame to grayscale for intensity manipulation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Boost intensity in hot areas
        boosted_gray = gray.astype(np.float32) + (heat_intensity * 80)  # Add heat
        boosted_gray = np.clip(boosted_gray, 0, 255).astype(np.uint8)
        
        # Apply thermal color mapping to boosted areas only
        # Use the existing thermal filter logic but with modified intensity
        enhanced = cv2.bilateralFilter(boosted_gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=10)
        
        # Create thermal LUT (same as apply_mock_thermal_filter)
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            intensity = i / 255.0
            if intensity < 0.2:
                t = intensity / 0.2
                r = int(t * 20)
                g = int(t * 15)
                b = int(t * 120)
            elif intensity < 0.4:
                t = (intensity - 0.2) / 0.2
                r = int(20 + t * 80)
                g = int(15 + t * 25)
                b = int(120 + t * 60)
            elif intensity < 0.6:
                t = (intensity - 0.4) / 0.2
                r = int(100 + t * 120)
                g = int(40 + t * 60)
                b = int(180 - t * 150)
            elif intensity < 0.8:
                t = (intensity - 0.6) / 0.2
                r = int(220 + t * 35)
                g = int(100 + t * 130)
                b = int(30 - t * 30)
            else:
                t = (intensity - 0.8) / 0.2
                r = int(255)
                g = int(230 + t * 25)
                b = int(t * 200)
            
            r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
            lut[i, 0] = [b, g, r]
        
        # Apply thermal colors to enhanced areas
        thermal_enhanced = cv2.LUT(cv2.merge([enhanced, enhanced, enhanced]), lut)
        
        # Create blend mask (thermal effect only where heat is)
        blend_mask = np.clip(heat_intensity * 2, 0, 1)  # Boost visibility
        
        # Blend thermal enhancement with original frame
        for c in range(3):
            frame[:, :, c] = (frame[:, :, c] * (1 - blend_mask) + 
                              thermal_enhanced[:, :, c] * blend_mask).astype(np.uint8)
        
        # Get thermal color scheme for text panel
        colormap_info = self.get_heatmap_colormap(props['power'])
        
        # Draw text panel
        self.draw_text_panel_beside_mask(
            frame, box, label, confidence, props, colormap_info
        )

        return frame

    def draw_heatmap_overlay(self, frame, box, label, confidence, frame_count):
        """
        Draw thermal radiation-style heatmap overlay on detected device
        Uses radial gradient to simulate heat distribution
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Skip if confidence below threshold
        if confidence < self.conf_threshold * 100:
            return frame
        
        if label not in self.electronics_map:
            return frame
        
        props = self.electronics_map[label]
        
        # Extract device region
        device_roi = frame[y1:y2, x1:x2].copy()
        h, w = device_roi.shape[:2]
        
        if h <= 10 or w <= 10:  # Skip very small detections
            return frame
        
        # Get or create cached gradient
        cache_key = f"{w}x{h}_{props['power']}"
        if cache_key not in self.gradient_cache:
            # Get color scheme for power level
            colormap_info = self.get_heatmap_colormap(props['power'])
            
            # Create device mask
            corner_radius = min(15, min(h, w) // 4)
            mask = self.create_rounded_rectangle_mask((h, w), corner_radius=corner_radius)
            
            # Create radial gradient heatmap
            heatmap, intensity_map = self.create_radial_gradient_heatmap_fast(
                (h, w), 
                colormap_info
            )
            
            self.gradient_cache[cache_key] = (mask, heatmap, colormap_info)
        else:
            mask, heatmap, colormap_info = self.gradient_cache[cache_key]
        
        # Apply mask to heatmap
        mask_3channel = cv2.merge([mask, mask, mask])
        heatmap_masked = cv2.bitwise_and(heatmap, mask_3channel)
        
        # Brighten the device area first (counteract dark tint)
        device_brightened = cv2.convertScaleAbs(device_roi, alpha=1.5, beta=30)
        
        # Blend brightened device with heatmap (70% heatmap for strong color)
        device_with_heatmap = cv2.addWeighted(
            device_brightened, 1 - self.heatmap_alpha,
            heatmap_masked, self.heatmap_alpha, 0
        )
        
        # Apply intensity scaling based on power level
        intensity_factor = colormap_info['intensity']
        device_with_heatmap = cv2.convertScaleAbs(device_with_heatmap, alpha=intensity_factor, beta=20)
        
        # Apply only to masked area
        mask_bool = mask > 0
        device_roi[mask_bool] = device_with_heatmap[mask_bool]
        
        # Add glow effect for high-intensity devices
        if colormap_info['intensity'] >= 0.85:
            device_roi = self.add_glow_effect(device_roi, mask, colormap_info)
        
        # Place back into frame
        frame[y1:y2, x1:x2] = device_roi
        
        # Draw text panel beside the mask
        self.draw_text_panel_beside_mask(
            frame, box, label, confidence, props, colormap_info
        )
        
        return frame
    
    def draw_pulsing_anchor(self, frame, box, label, confidence, frame_count):
        """
        Draw pulsing energy sphere with radial gradient fill
        Creates futuristic force field effect with glowing particles
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Skip if confidence below threshold
        if confidence < self.conf_threshold * 100:
            return frame
        
        if label not in self.electronics_map:
            return frame
        
        props = self.electronics_map[label]
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Get color gradient based on power consumption
        colormap_info = self.get_heatmap_colormap(props['power'])
        colors = colormap_info['colors']
        
        # Pulsing animation parameters
        pulse_speed = 0.08  # Slightly slower for smoother effect
        pulse_phase = (frame_count * pulse_speed) % (2 * np.pi)
        
        # Calculate base radius from bounding box size (reduced size)
        box_width = x2 - x1
        box_height = y2 - y1
        base_radius = min(box_width, box_height) // 3.5  # Reduced from 2.5 to 3.5
        base_radius = max(base_radius, 25)  # Reduced minimum radius from 40 to 25
        
        # Pulse effect: radius oscillates
        pulse_amplitude = base_radius * 0.25  # 25% size variation
        pulse_offset = np.sin(pulse_phase) * pulse_amplitude
        current_radius = int(base_radius + pulse_offset)
        
        # Create a separate overlay for the sphere
        sphere_size = current_radius * 2 + 20
        sphere_overlay = np.zeros((sphere_size, sphere_size, 3), dtype=np.uint8)
        sphere_center = (sphere_size // 2, sphere_size // 2)
        
        # Create radial gradient fill for energy sphere
        # Draw multiple concentric filled circles with decreasing opacity
        num_layers = 60  # More layers for smoother gradient
        
        for i in range(num_layers, 0, -1):
            layer_radius = int(current_radius * (i / num_layers))
            if layer_radius < 1:
                continue
            
            # Calculate color interpolation based on distance from center
            pos = (num_layers - i) / num_layers
            
            # Map to color gradient
            color_pos = pos * (len(colors) - 1)
            color_idx_low = int(np.floor(color_pos))
            color_idx_high = min(color_idx_low + 1, len(colors) - 1)
            blend = color_pos - color_idx_low
            
            # Interpolate between colors (brightest at center)
            color_low = np.array(colors[-(color_idx_low + 1)])  # Reverse for bright center
            color_high = np.array(colors[-(color_idx_high + 1)])
            interpolated_color = color_low * (1 - blend) + color_high * blend
            
            # Add intensity boost to center
            intensity_boost = 1.0 + (1.0 - pos) * 0.5  # Up to 50% brighter at center
            interpolated_color = np.clip(interpolated_color * intensity_boost, 0, 255)
            
            cv2.circle(sphere_overlay, sphere_center, layer_radius, 
                      tuple(map(int, interpolated_color)), -1)
        
        # Calculate position on main frame
        y1_sphere = max(0, center_y - sphere_size // 2)
        y2_sphere = min(frame.shape[0], center_y + sphere_size // 2)
        x1_sphere = max(0, center_x - sphere_size // 2)
        x2_sphere = min(frame.shape[1], center_x + sphere_size // 2)
        
        # Calculate corresponding region in sphere overlay
        y1_overlay = sphere_size // 2 - (center_y - y1_sphere)
        y2_overlay = sphere_size // 2 + (y2_sphere - center_y)
        x1_overlay = sphere_size // 2 - (center_x - x1_sphere)
        x2_overlay = sphere_size // 2 + (x2_sphere - center_x)
        
        # Extract regions
        frame_roi = frame[y1_sphere:y2_sphere, x1_sphere:x2_sphere]
        sphere_roi = sphere_overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
        
        # Blend sphere with frame using alpha based on intensity
        sphere_mask = np.any(sphere_roi > 0, axis=2).astype(np.float32)
        sphere_alpha = sphere_mask * 0.85  # 85% opacity for sphere
        
        # Apply blend
        for c in range(3):
            frame_roi[:, :, c] = (frame_roi[:, :, c] * (1 - sphere_alpha) + 
                                  sphere_roi[:, :, c] * sphere_alpha).astype(np.uint8)
        
        # Add outer glow effect (force field)
        glow_radius = int(current_radius * 1.2)
        glow_color = colors[-1]  # Brightest color
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), glow_radius, glow_color, 4)
        glow_alpha = 0.3 + 0.15 * np.sin(pulse_phase)  # Pulsing glow
        cv2.addWeighted(overlay, glow_alpha, frame, 1 - glow_alpha, 0, frame)
        
        # Add multiple pulsing force field rings
        for ring_offset in [1.3, 1.45]:
            ring_radius = int(current_radius * ring_offset + pulse_offset * 0.5)
            ring_alpha = 0.2 + 0.1 * np.sin(pulse_phase + ring_offset)
            overlay = frame.copy()
            cv2.circle(overlay, (center_x, center_y), ring_radius, colors[-1], 2)
            cv2.addWeighted(overlay, ring_alpha, frame, 1 - ring_alpha, 0, frame)
        
        # Add energy particles around the sphere
        num_particles = 8
        particle_distance = current_radius * 1.25
        for i in range(num_particles):
            angle = (i / num_particles) * 2 * np.pi + (pulse_phase * 0.5)  # Rotate slowly
            particle_x = int(center_x + np.cos(angle) * particle_distance)
            particle_y = int(center_y + np.sin(angle) * particle_distance)
            
            # Pulsing particle size
            particle_size = int(3 + 2 * np.sin(pulse_phase + i))
            particle_alpha = 0.4 + 0.2 * np.sin(pulse_phase + i)
            
            overlay = frame.copy()
            cv2.circle(overlay, (particle_x, particle_y), particle_size, colors[-1], -1)
            cv2.addWeighted(overlay, particle_alpha, frame, 1 - particle_alpha, 0, frame)
        
        # Draw bright center core
        core_radius = max(3, current_radius // 12)
        core_pulse = int(core_radius + 2 * np.sin(pulse_phase * 2))  # Faster pulse
        cv2.circle(frame, (center_x, center_y), core_pulse, (255, 255, 255), -1)
        
        # Add center glow
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), core_pulse * 2, (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw text panel beside the anchor
        self.draw_text_panel_beside_anchor(
            frame, (center_x, center_y), base_radius, label, confidence, props, colormap_info
        )
        
        return frame
    
    def draw_text_panel_beside_anchor(self, frame, center_pos, anchor_radius, label, confidence, props, colormap_info):
        """
        Draw text panel beside the pulsing anchor
        Auto-positions to avoid overlapping with anchor
        """
        center_x, center_y = center_pos
        frame_height, frame_width = frame.shape[:2]
        
        panel_width = 250
        panel_height = 85
        
        # Calculate safe distance from anchor
        safe_distance = int(anchor_radius * 2 + 15)
        
        # Auto-position: right side if space, else left side
        if center_x + safe_distance + panel_width < frame_width:
            # Position to the right
            panel_x = center_x + safe_distance
        elif center_x - safe_distance - panel_width > 0:
            # Position to the left
            panel_x = center_x - safe_distance - panel_width
        else:
            # Position above or below if no side space
            panel_x = max(center_x - panel_width // 2, 10)
        
        # Position panel vertically centered with anchor
        panel_y = max(min(center_y - panel_height // 2, 
                          frame_height - panel_height - 10), 10)
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), 
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Use gradient color for border (from colormap)
        border_color = colormap_info['colors'][-1]  # Brightest color from gradient
        
        # Border for info panel
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), 
                      border_color, 2)
        
        # Text content
        text_x = panel_x + 10
        text_y = panel_y + 28
        
        # Device name
        cv2.putText(frame, f"{label.upper()}", (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Power consumption
        cv2.putText(frame, f"Power: {props['power']}", (text_x, text_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Confidence
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (text_x, text_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    def draw_text_panel_beside_mask(self, frame, box, label, confidence, props, colormap_info):
        """
        Draw text panel beside the heatmap mask
        Auto-positions to right or left based on available space
        """
        x1, y1, x2, y2 = map(int, box)
        frame_height, frame_width = frame.shape[:2]
        
        panel_width = 250
        panel_height = 85  # Reduced height (no heat level text)
        
        # Auto-position: right side if space, else left side
        if x2 + panel_width + 10 < frame_width:
            # Position to the right
            panel_x = x2 + 10
        elif x1 - panel_width - 10 > 0:
            # Position to the left
            panel_x = x1 - panel_width - 10
        else:
            # Default to above if no side space
            panel_x = max(x1, 10)
        
        # Position panel vertically centered with mask
        mask_center_y = (y1 + y2) // 2
        panel_y = max(min(mask_center_y - panel_height // 2, 
                          frame_height - panel_height - 10), 10)
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), 
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Use gradient color for border (from colormap)
        border_color = colormap_info['colors'][-1]  # Brightest color from gradient
        
        # Border for info panel
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), 
                      border_color, 2)
        
        # Text content
        text_x = panel_x + 10
        text_y = panel_y + 28
        
        # Device name
        cv2.putText(frame, f"{label.upper()}", (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Power consumption (no heat level label)
        cv2.putText(frame, f"Power: {props['power']}", (text_x, text_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Confidence
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (text_x, text_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
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
        print("  D       - Toggle detection interval (speed vs accuracy)")
        print("  C       - Cycle confidence threshold (30%-80%)")
        print("  H       - Toggle anchor/box display mode")
        print("  T       - Toggle thermal filter overlay")
        print("  M       - Toggle visualization mode (anchor/mask)")
        print("="*60)
        print("\n🚀 Starting detection loop...\n")
        
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
                # Run YOLO segmentation detection with lower confidence for better detection
                results = self.model(frame, conf=0.3, verbose=False)
                
                # Cache detection results
                self.last_detections = []
                
                for result in results:
                    boxes = result.boxes
                    
                    # Extract segmentation masks if available
                    masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
                    
                    for idx, box in enumerate(boxes):
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
                        
                        # Extract segmentation mask for this detection
                        seg_mask = None
                        if masks is not None and idx < len(masks.data):
                            seg_mask = masks.data[idx].cpu().numpy()
                        
                        # Only cache electronics
                        if label in self.electronics_map:
                            self.last_detections.append({
                                'box': [x1, y1, x2, y2],
                                'label': label,
                                'confidence': confidence,
                                'mask': seg_mask
                            })
            
            # Apply permanent dark tint overlay for cinematic surveillance effect (70% darker)
            dark_overlay = frame.copy()
            dark_overlay = cv2.addWeighted(dark_overlay, self.dark_tint, np.zeros_like(dark_overlay), 1 - self.dark_tint, 0)
            frame = dark_overlay

            # Apply mock thermal filter overlay (simulates thermal camera effect on entire frame)
            if self.show_thermal_filter:
                frame = self.apply_mock_thermal_filter(frame)
            
            # Draw cached detections (from last YOLO run)
            for detection in self.last_detections:
                # Only count if above confidence threshold
                if detection['confidence'] >= self.conf_threshold * 100:
                    detected_items[detection['label']] += 1
                
                # Draw visualization based on mode
                if self.show_heatmap:
                    if self.visualization_mode == 'mask':
                        # Segmentation mask mode (detailed)
                        frame = self.draw_pulsing_anchor_contour_constrained(
                            frame,
                            detection['box'],
                            detection['label'],
                            detection['confidence'],
                            frame_count,
                            detection.get('mask', None)
                        )
                    else:
                        # Pulsing anchor mode (default)
                        frame = self.draw_pulsing_anchor(
                            frame,
                            detection['box'],
                            detection['label'],
                            detection['confidence'],
                            frame_count
                        )
                else:
                    # Fallback to old style boxes
                    self.draw_custom_overlay(frame, detection['box'], detection['label'], detection['confidence'])
            
            # Calculate FPS
            frame_time = time.time() - start_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(fps)
            if len(self.fps_history) > self.max_fps_samples:
                self.fps_history.pop(0)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            # Draw FPS and stats
            panel_height = 175 if last_key_press else 145
            cv2.rectangle(frame, (10, 10), (300, 10 + panel_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, 10 + panel_height), (0, 255, 0), 2)
            
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Detected: {sum(detected_items.values())} items", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Detection: Every {self.detection_interval} frames", (20, 92),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.putText(frame, f"Confidence: {self.conf_threshold*100:.0f}%", (20, 112),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Show visualization mode
            mode_display = "Anchor" if self.visualization_mode == 'anchor' else "Mask"
            cv2.putText(frame, f"Mode: {mode_display}", (20, 132),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Show last key press feedback (fades after 2 seconds)
            if last_key_press and time.time() - key_press_time < 2:
                cv2.putText(frame, f"Key: {last_key_press}", (20, 162),
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
                    
                elif key == ord('d') or key == ord('D'):
                    # Toggle detection interval (1, 2, 3 frames)
                    self.detection_interval = (self.detection_interval % 3) + 1
                    speed_desc = {1: "High accuracy", 2: "Balanced", 3: "High speed"}
                    last_key_press = f"D - {speed_desc[self.detection_interval]}"
                    key_press_time = time.time()
                    print(f"[INFO] Detection interval: Every {self.detection_interval} frames ({speed_desc[self.detection_interval]})")
                    
                elif key == ord('c') or key == ord('C'):
                    # Cycle confidence threshold: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
                    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                    try:
                        current_idx = thresholds.index(self.conf_threshold)
                        self.conf_threshold = thresholds[(current_idx + 1) % len(thresholds)]
                    except ValueError:
                        self.conf_threshold = 0.7  # Default if not in list
                    last_key_press = f"C - Confidence: {self.conf_threshold*100:.0f}%"
                    key_press_time = time.time()
                    print(f"[INFO] Confidence threshold: {self.conf_threshold*100:.0f}%")
                    
                elif key == ord('h') or key == ord('H'):
                    # Toggle anchor mode
                    self.show_heatmap = not self.show_heatmap
                    mode = "Anchor" if self.show_heatmap else "Boxes"
                    last_key_press = f"H - {mode} Mode"
                    key_press_time = time.time()
                    print(f"[INFO] Display mode: {mode}")
                    # Clear gradient cache when toggling
                    if self.show_heatmap:
                        self.gradient_cache.clear()

                elif key == ord('t') or key == ord('T'):
                    # Toggle thermal filter overlay
                    self.show_thermal_filter = not self.show_thermal_filter
                    thermal_status = "ON" if self.show_thermal_filter else "OFF"
                    last_key_press = f"T - Thermal Filter {thermal_status}"
                    key_press_time = time.time()
                    print(f"[INFO] Thermal filter: {thermal_status}")

                elif key == ord('m') or key == ord('M'):
                    # Toggle visualization mode
                    self.visualization_mode = 'mask' if self.visualization_mode == 'anchor' else 'anchor'
                    mode_name = "Segmentation Mask" if self.visualization_mode == 'mask' else "Pulsing Anchor"
                    last_key_press = f"M - {mode_name}"
                    key_press_time = time.time()
                    print(f"[INFO] Visualization mode: {mode_name}")
        
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
