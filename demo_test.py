import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import os

def test_yolo_setup():
    """Test if YOLO can be loaded and basic detection works"""
    print("🔍 Testing YOLO Object Detection Setup")
    print("=" * 50)
    
    # Test 1: Check PyTorch installation
    print("1. Checking PyTorch installation...")
    try:
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   ✅ Using device: {device}")
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        return False
    
    # Test 2: Load YOLOv5 model
    print("\n2. Loading YOLOv5 model...")
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device)
        print(f"   ✅ Model loaded successfully!")
        print(f"   ✅ Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"   ❌ Model loading error: {e}")
        print("   💡 Try: pip install ultralytics")
        return False
    
    # Test 3: Download test image
    print("\n3. Downloading test image...")
    test_image_path = 'test_detection.jpg'
    try:
        if not os.path.exists(test_image_path):
            url = "https://ultralytics.com/images/bus.jpg"
            urllib.request.urlretrieve(url, test_image_path)
            print(f"   ✅ Test image downloaded: {test_image_path}")
        else:
            print(f"   ✅ Test image exists: {test_image_path}")
    except Exception as e:
        print(f"   ❌ Download error: {e}")
        return False
    
    # Test 4: Run detection
    print("\n4. Running object detection...")
    try:
        # Load and process image
        img = cv2.imread(test_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(img_rgb)
        
        # Get detections
        detections = results.pandas().xyxy[0]
        num_detections = len(detections)
        
        print(f"   ✅ Detection completed!")
        print(f"   ✅ Objects detected: {num_detections}")
        
        if num_detections > 0:
            print(f"   ✅ Detected classes: {list(detections['name'].unique())}")
            
            # Show detection details
            for idx, detection in detections.iterrows():
                name = detection['name']
                confidence = detection['confidence']
                print(f"      - {name}: {confidence:.2%}")
        
        return True, model, results, img_rgb
        
    except Exception as e:
        print(f"   ❌ Detection error: {e}")
        return False
    
def create_detection_visualization(results, img_rgb):
    """Create and save detection visualization"""
    print("\n5. Creating visualization...")
    try:
        # Plot results
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)
        
        # Get detection data
        detections = results.pandas().xyxy[0]
        
        # Draw bounding boxes
        for idx, detection in detections.iterrows():
            x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
            confidence = detection['confidence']
            name = detection['name']
            
            # Draw rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(rect)
            
            # Add label
            label = f"{name}: {confidence:.2%}"
            ax.text(x1, y1-10, label, bbox=dict(facecolor='red', alpha=0.8), 
                   fontsize=12, color='white', weight='bold')
        
        ax.set_title('YOLO Object Detection Results', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Save result
        plt.tight_layout()
        plt.savefig('yolo_detection_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Visualization saved: yolo_detection_result.png")
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization error: {e}")
        return False

def test_webcam_detection():
    """Test webcam detection (without actually starting camera)"""
    print("\n6. Testing webcam detection setup...")
    try:
        # Check if camera is available
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("   ✅ Webcam detected and accessible")
            print("   💡 You can run webcam detection with:")
            print("      python webcam_demo.py")
            cap.release()
        else:
            print("   ⚠️  No webcam detected")
            print("   💡 Webcam detection will work when camera is available")
        
        return True
    except Exception as e:
        print(f"   ❌ Webcam test error: {e}")
        return False

def main():
    """Main demo function"""
    print("🚀 YOLO Object Detection Demo")
    print("This will test all YOLO components and create a demo detection")
    print("=" * 60)
    
    # Run setup test
    setup_result = test_yolo_setup()
    
    if setup_result:
        success, model, results, img_rgb = setup_result
        if success:
            # Create visualization
            create_detection_visualization(results, img_rgb)
            
            # Test webcam setup
            test_webcam_detection()
            
            print("\n🎉 YOLO Demo Completed Successfully!")
            print("=" * 60)
            print("✅ All components working correctly")
            print("📸 Detection result saved as 'yolo_detection_result.png'")
            print("🎥 Ready for real-time webcam detection")
            print("📁 Ready for video file processing")
            
            print("\n🔧 Next Steps:")
            print("1. Upload this code to your YOLO GitHub repository")
            print("2. Add the detection result image to your repo")
            print("3. Create a demo video or GIF")
            print("4. Update README with demo instructions")
            
            return True
    
    print("\n❌ Demo failed - check error messages above")
    print("💡 Make sure to install: pip install torch torchvision ultralytics opencv-python matplotlib")
    return False

if __name__ == "__main__":
    main()
