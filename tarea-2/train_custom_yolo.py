# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opencv-python",
#     "ultralytics",
#     "numpy",
#     "matplotlib",
# ]
# ///

import os
from pathlib import Path
from ultralytics import YOLO
import yaml


def setup_dataset_paths():
    """Fix dataset paths to be absolute"""
    
    # Get absolute path to data directory
    data_dir = Path(__file__).parent / "data"
    dataset_config = data_dir / "data.yaml"
    
    if not dataset_config.exists():
        print(f"❌ Dataset config not found: {dataset_config}")
        return None
    
    # Read current config
    with open(dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths to be absolute
    base_path = data_dir.parent  # Go up one level from data/
    
    # Create corrected config
    corrected_config = {
        'path': str(base_path),  # Root path
        'train': 'data/train/images',  # Relative to path
        'val': 'data/valid/images' if (base_path / 'data/valid').exists() else 'data/train/images',
        'test': 'data/test/images' if (base_path / 'data/test').exists() else 'data/train/images',
        'nc': config['nc'],
        'names': config['names']
    }
    
    # Save corrected config
    corrected_file = data_dir / "data_corrected.yaml"
    with open(corrected_file, 'w') as f:
        yaml.dump(corrected_config, f, default_flow_style=False)
    
    print(f"✓ Created corrected dataset config: {corrected_file}")
    print(f"✓ Classes: {corrected_config['names']}")
    print(f"✓ Number of classes: {corrected_config['nc']}")
    
    return str(corrected_file)


def train_custom_yolo():
    """Fine-tune YOLOv12 on custom cap/hat dataset"""
    
    print("=== YOLOv12 Custom Training: Cap/Hat Detection ===")
    print("Fine-tuning YOLOv12 nano on custom dataset\n")
    
    # Setup dataset configuration
    dataset_config = setup_dataset_paths()
    if not dataset_config:
        return
    
    # Load pre-trained YOLOv12 nano model
    print("Loading pre-trained YOLOv12 nano model...")
    model = YOLO("yolo12n.pt")  # Load pre-trained weights for transfer learning
    
    # Configure training parameters for CPU
    training_config = {
        'data': dataset_config,       # Dataset configuration
        'epochs': 50,#,                 # Reduced epochs for quick training
        'imgsz': 640,                # Image size
        'device': 'cpu',             # Use CPU
        'batch': 4,                  # Small batch size for CPU
        'patience': 10,              # Early stopping patience
        'save_period': 5,            # Save checkpoint every 5 epochs
        'workers': 2,                # Number of data loading workers
        'project': 'custom_training', # Output directory
        'name': 'cap_hat_yolo12',    # Experiment name
        'exist_ok': True,            # Overwrite existing results
        'pretrained': True,          # Use pre-trained weights
        'verbose': True              # Show training progress
    }
    
    print(f"✓ Training configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    print()
    
    # Start training
    print("🚀 Starting fine-tuning...")
    print("This will train the model to detect caps and hats in addition to COCO objects")
    print("Training may take several minutes on CPU...\n")
    
    try:
        # Train the model
        results = model.train(**training_config)
        
        print("\n✅ Training completed!")
        print(f"✓ Best model saved to: custom_training/cap_hat_yolo12/weights/best.pt")
        print(f"✓ Last model saved to: custom_training/cap_hat_yolo12/weights/last.pt")
        
        # Load the best trained model for testing
        best_model_path = "custom_training/cap_hat_yolo12/weights/best.pt"
        if os.path.exists(best_model_path):
            print(f"\n📊 Model ready for inference:")
            print(f"   Load with: model = YOLO('{best_model_path}')")
            
            # Quick test
            trained_model = YOLO(best_model_path)
            print(f"   Classes: {trained_model.names}")
        
        return best_model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


def test_custom_model(model_path):
    """Test the trained model with webcam"""
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        return
    
    print(f"\n=== Testing Custom Model ===")
    print("Loading custom trained model...")
    
    import cv2
    import time
    
    # Load custom model
    model = YOLO(model_path)
    
    print(f"✓ Custom model loaded")
    print(f"✓ Model classes: {list(model.names.values())}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✓ Camera initialized")
    print("✓ Press 'q' to quit")
    
    cv2.namedWindow("Custom YOLOv12 - Cap/Hat Detection", cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame, conf=0.3, verbose=False)  # Lower confidence for custom objects
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Add info
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            cv2.putText(annotated_frame, f"Custom YOLOv12 | Detections: {num_detections}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Custom YOLOv12 - Cap/Hat Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Test completed")


def main():
    """Main function with options"""
    
    print("Choose an option:")
    print("1. Train custom model (fine-tune YOLOv12 for cap/hat detection)")
    print("2. Test existing custom model")
    print("3. Both (train then test)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        train_custom_yolo()
    elif choice == "2":
        model_path = input("Enter model path (or press Enter for default): ").strip()
        if not model_path:
            model_path = "custom_training/cap_hat_yolo12/weights/best.pt"
        test_custom_model(model_path)
    elif choice == "3":
        model_path = train_custom_yolo()
        if model_path:
            test_custom_model(model_path)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
