# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opencv-python",
#     "ultralytics",
#     "numpy",
# ]
# ///

import cv2
import time
import os
from ultralytics import YOLO
import torch


def main():
    print("=== YOLOv12 Nano Real-time Object Detection ===")
    print("Fixed for CPU-only usage")
    print("Press 'q' to quit\n")
    
    # Force CPU usage to avoid CUDA issues
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Load YOLOv12 nano model (most efficient for CPU)
    print("Loading YOLOv12 nano model...")
    try:
        model = YOLO("yolo12n.pt")  # Nano model for best CPU performance
        model.to('cpu')  # Force CPU usage
        print("✓ YOLOv12 nano model loaded successfully on CPU")
        print(f"✓ Model can detect {len(model.names)} classes:")
        
        # Print classes in a compact format
        class_list = list(model.names.values())
        for i in range(0, len(class_list), 10):
            row = class_list[i:i+10]
            print(f"   {', '.join(row)}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return
    
    # Set camera properties for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera initialized (640x480 @ 30fps)")
    
    # Create window
    cv2.namedWindow("YOLOv12 Real-time Detection", cv2.WINDOW_AUTOSIZE)
    
    # FPS calculation variables
    fps_counter = 0
    start_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Run YOLO detection with explicit CPU device
            results = model(frame, conf=0.5, verbose=False, device='cpu')
            
            # Draw detections on frame
            annotated_frame = results[0].plot()
            
            # Add info overlay
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            cv2.putText(annotated_frame, f"YOLOv12n CPU | Detections: {num_detections}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate FPS
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow("YOLOv12 Real-time Detection", annotated_frame)
            
            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹ Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup completed")


if __name__ == "__main__":
    main()