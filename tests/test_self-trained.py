# real_time_lip_reading.py
import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import argparse
from collections import deque
import os
import sys

print("Starting script...")

# Import the model
from video_cnn import VideoCNN
from model import VideoModel

class Args:
    def __init__(self):
        self.n_class = 500  # Start with original size
        self.border = False
        self.se = False
        self.dataset = 'lrw'

class RealTimeLipReader:
    def __init__(self, model_path, vocab_list):
        self.vocab_list = vocab_list
        self.args = Args()
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            sys.exit(1)
        
        print(f"Model file found at: {model_path}")
        
        # Load model
        print("Loading model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            # First create model with original 500 classes
            self.model = VideoModel(self.args).to(self.device)
            print("VideoModel created successfully")
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print(f"Checkpoint keys: {checkpoint.keys()}")
            
            # Load state dict but ignore the FC layer
            state_dict = checkpoint['video_model']
            
            # Remove v_cls (and potentially FC) layers that don't match
            keys_to_remove = []
            for key in state_dict.keys():
                if 'v_cls' in key or 'FC' in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del state_dict[key]
            
            # Load the partial state dict
            self.model.load_state_dict(state_dict, strict=False)
            print("Loaded partial state dict (without classifier layers)")
            
            # Now replace the classifier with the correct size
            self.model.v_cls = torch.nn.Linear(2048, 30).to(self.device)
            
            # If there's an FC layer in the checkpoint specifically for 30 classes
            if 'FC.weight' in checkpoint['video_model'] and checkpoint['video_model']['FC.weight'].shape[0] == 30:
                # Create and load the FC layer
                self.model.FC = torch.nn.Sequential(
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(2048, 30)
                ).to(self.device)
                
                # Load FC weights if they exist
                fc_state = {}
                for key in checkpoint['video_model'].keys():
                    if 'FC' in key:
                        fc_state[key] = checkpoint['video_model'][key]
                
                if fc_state:
                    self.model.FC.load_state_dict(fc_state, strict=False)
                    print("Loaded fine-tuned FC layer")
            
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Initialize MediaPipe
        print("Initializing MediaPipe...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Recording state
        self.recording = False
        self.recording_frames = []
        self.processed_frames = deque(maxlen=29)
        self.fps_time = time.time()
        self.fps_counter = 0
        
    def extract_mouth_region(self, frame):
        """Extract mouth region from frame using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            face_landmarks = results.multi_face_landmarks[0]
            
            # Mouth landmarks
            mouth_indices = [61, 84, 17, 314, 405, 409, 415, 308, 324, 318,
                           78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
            
            mouth_points = []
            for idx in mouth_indices:
                if idx < len(face_landmarks.landmark):
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    mouth_points.append((x, y))
            
            if mouth_points:
                mouth_points = np.array(mouth_points)
                x_min, y_min = mouth_points.min(axis=0)
                x_max, y_max = mouth_points.max(axis=0)
                
                # Add padding
                width = x_max - x_min
                height = y_max - y_min
                size = int(max(width, height) * 1.5)
                
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                
                x1 = max(0, x_center - size // 2)
                y1 = max(0, y_center - size // 2)
                x2 = min(w, x1 + size)
                y2 = min(h, y1 + size)
                
                mouth_roi = frame[y1:y2, x1:x2]
                
                if mouth_roi.size > 0:
                    gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (88, 88))
                    normalized = resized.astype(np.float32) / 255.0
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    return normalized, frame, (x1, y1, x2, y2)
        
        return None, frame, None
    
    def process_recording(self):
        """Process recorded frames: evenly sample 29 frames from the recording"""
        if len(self.recording_frames) < 29:
            print(f"Warning: Only recorded {len(self.recording_frames)} frames, need at least 29")
            return False
        
        # Use np.linspace to evenly sample 29 frames from the entire recording
        # This maintains temporal consistency better than simple downsampling
        num_recorded = len(self.recording_frames)
        indices = np.linspace(0, num_recorded - 1, 29, dtype=int)
        
        # Clear and fill processed frames with evenly sampled frames
        self.processed_frames.clear()
        for idx in indices:
            self.processed_frames.append(self.recording_frames[idx])
        
        print(f"Processed: {num_recorded} frames -> 29 frames (evenly sampled)")
        return True
    
    def predict(self):
        """Make prediction on current frame buffer"""
        if len(self.processed_frames) < 29:
            return None, 0.0
        
        try:
            frames = np.array(list(self.processed_frames))
            frames = frames.reshape(29, 1, 88, 88)
            frames_tensor = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(frames_tensor)
                
                # If the model has an FC layer, the output might already be 30-dim
                if output.shape[-1] != 30:
                    # Use only the first 30 classes or map somehow
                    output = output[:, :30]
                
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_idx = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, predicted_idx].item()
            
            return self.vocab_list[predicted_idx], confidence
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def run(self):
        """Main loop for real-time lip reading"""
        print("Opening camera...")
        
        # Try different camera indices
        cap = None
        for camera_idx in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                print(f"Camera opened at index {camera_idx}")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("Error: Could not open any camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nCamera opened successfully!")
        print("Starting real-time lip reading...")
        print("Press 'SPACE' to start recording 70 frames")
        print("Press 'q' to quit")
        
        current_word = ""
        confidence = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Extract mouth region
                mouth_region, annotated_frame, bbox = self.extract_mouth_region(frame)
                
                # Handle recording
                if self.recording and mouth_region is not None:
                    self.recording_frames.append(mouth_region)
                    
                    if len(self.recording_frames) >= 70:
                        self.recording = False
                        print("Recording complete! Processing frames...")
                        
                        if self.process_recording():
                            # Make prediction
                            pred_word, conf = self.predict()
                            
                            if pred_word:
                                current_word = pred_word
                                confidence = conf
                                print(f"Prediction: {current_word} ({confidence:.2%})")
                
                # Display
                display_frame = annotated_frame.copy()
                
                # FPS
                self.fps_counter += 1
                fps = 30 / max(time.time() - self.fps_time, 0.001)
                if self.fps_counter % 30 == 0:
                    self.fps_time = time.time()
                
                # Text overlay
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Recording status
                if self.recording:
                    cv2.putText(display_frame, f"RECORDING: {len(self.recording_frames)}/70", (200, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # Red recording dot
                    cv2.circle(display_frame, (170, 65), 10, (0, 0, 255), -1)
                else:
                    cv2.putText(display_frame, "Press SPACE to record", (200, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                if current_word:
                    cv2.putText(display_frame, f"Word: {current_word}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Confidence: {confidence:.2%}", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Real-time Lip Reading', display_frame)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if not self.recording:
                        self.recording = True
                        self.recording_frames = []
                        current_word = ""
                        confidence = 0.0
                        print("Started recording...")
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up...")
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()

def main():
    parser = argparse.ArgumentParser(description='Real-time lip reading')
    parser.add_argument('--model_path', type=str,
                       default='checkpoints/balanced_finetuned_acc_86.67.pt',
                       help='Path to the fine-tuned model checkpoint')
    args = parser.parse_args()
    
    # Your 30-word vocabulary
    vocab_list = [
        "BAD", "BYE", "COME", "FIVE", "FOUR", "GO", "GOOD", "GOODBYE",
        "HELLO", "HELP", "HI", "HOW", "ME", "NEED", "NO", "NOW",
        "OKAY", "ONE", "PLEASE", "SORRY", "THANKS", "THREE", "TIME",
        "TODAY", "TWO", "WANT", "WHAT", "WHERE", "YES", "YOU"
    ]
    
    print(f"Loaded vocabulary with {len(vocab_list)} words")
    
    try:
        lip_reader = RealTimeLipReader(args.model_path, vocab_list)
        lip_reader.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()