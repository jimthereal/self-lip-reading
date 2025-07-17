"""
Debug script to test video processing
"""

import cv2
import numpy as np
from pathlib import Path
import json
import mediapipe as mp

def test_video_processing():
    """Test processing a single video"""
    
    # Setup paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    dataset_path = base_dir / "data" / "augmented_lipreading_dataset"
    
    print("="*60)
    print("VIDEO PROCESSING DEBUG TEST")
    print("="*60)
    
    # Check dataset exists
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return
    
    # Load metadata
    metadata_file = dataset_path / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Get first video
    first_word = list(metadata.keys())[0]
    first_video_info = metadata[first_word][0]
    video_path = dataset_path / first_video_info['filepath']
    
    print(f"\nTesting with video: {video_path}")
    print(f"Video exists: {video_path.exists()}")
    
    if not video_path.exists():
        print("ERROR: Video file not found!")
        return
    
    # Try to open video
    print("\n1. Testing OpenCV video capture...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: Cannot open video with OpenCV")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video opened successfully!")
    print(f"  FPS: {fps}")
    print(f"  Frames: {frame_count}")
    print(f"  Size: {width}x{height}")
    
    # Read first frame
    print("\n2. Testing frame reading...")
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        cap.release()
        return
    
    print(f"First frame read successfully!")
    print(f"  Frame shape: {frame.shape}")
    
    cap.release()
    
    # Test MediaPipe
    print("\n3. Testing MediaPipe...")
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        print("Face detection initialized successfully!")
        
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            print(f"Face detected! Found {len(results.detections)} face(s)")
        else:
            print("No face detected in first frame")
        
        face_detection.close()
        
    except Exception as e:
        print(f"ERROR with MediaPipe: {e}")
        import traceback
        traceback.print_exc()
    
    # Test face mesh
    print("\n4. Testing MediaPipe Face Mesh...")
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("Face mesh initialized successfully!")
        
        # Process frame
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            print(f"Face landmarks detected! Found {len(results.multi_face_landmarks[0].landmark)} landmarks")
        else:
            print("No face landmarks detected in first frame")
        
        face_mesh.close()
        
    except Exception as e:
        print(f"ERROR with Face Mesh: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Debug test complete!")
    
    # Try simple processing loop
    print("\n5. Testing video processing loop...")
    cap = cv2.VideoCapture(str(video_path))
    
    frames_processed = 0
    max_frames = 10  # Just test first 10 frames
    
    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1
    
    cap.release()
    print(f"Successfully processed {frames_processed} frames")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_video_processing()