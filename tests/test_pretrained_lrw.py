# test_pretrained_lrw_modified.py
import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import argparse
from collections import deque
import os
import sys

print("Starting LRW pre-trained model test...")

# Import the model
from video_cnn import VideoCNN
from model import VideoModel

class Args:
    def __init__(self, model_path=""):
        self.n_class = 500  # LRW has 500 classes
        # Check if model uses border (word boundary) feature
        self.border = 'border' in model_path.lower()
        # Check if model uses SE (Squeeze-and-Excitation) blocks
        self.se = 'se' in model_path.lower()
        self.dataset = 'lrw'

class LRWLipReader:
    def __init__(self, model_path):
        # Pass model_path to Args to auto-detect features
        self.args = Args(model_path)
        
        # Load vocabulary from file
        self.vocab_list = self.load_vocabulary()
        
        print(f"Loaded vocabulary with {len(self.vocab_list)} words")
        print(f"Model configuration - Border: {self.args.border}, SE: {self.args.se}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            sys.exit(1)
        
        print(f"Model file found at: {model_path}")
        
        # Load model
        print("Loading pre-trained LRW model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            # Create model with 500 classes
            self.model = VideoModel(self.args).to(self.device)
            print("VideoModel created successfully")
            
            # Load the pre-trained weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # For pre-trained model, we expect 'video_model' key
            if 'video_model' in checkpoint:
                self.model.load_state_dict(checkpoint['video_model'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.eval()
            print("Pre-trained model loaded successfully!")
            
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
    
    def load_vocabulary(self):
        """Load LRW vocabulary from file"""
        if not os.path.exists("lrw_vocab.txt"):
            print("Error: lrw_vocab.txt not found!")
            sys.exit(1)
        
        with open("lrw_vocab.txt", "r") as f:
            vocab = [line.strip().upper() for line in f.readlines() if line.strip()]
        
        if len(vocab) != 500:
            print(f"Warning: Expected 500 words in vocabulary, but found {len(vocab)}")
        
        return vocab
        
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
            return None, 0.0, []
        
        try:
            frames = np.array(list(self.processed_frames))
            frames = frames.reshape(29, 1, 88, 88)
            frames_tensor = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
            
            # If model uses border (word boundary), create border tensor
            if self.args.border:
                # Create word boundaries tensor (1 for frames with lip movement, 0 otherwise)
                # For real-time, we assume all frames contain the word
                borders = torch.ones(1, 29, 1).to(self.device)
                
                with torch.no_grad():
                    output = self.model(frames_tensor, borders)
            else:
                with torch.no_grad():
                    output = self.model(frames_tensor)
            
            # Check if output is valid
            if output is None:
                print("Model returned None")
                return None, 0.0, []
            
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probabilities[0], 5)
            
            predicted_idx = top5_indices[0].item()
            confidence = top5_probs[0].item()
            
            # Get top 5 words with bounds checking
            top5_words = []
            for idx, prob in zip(top5_indices, top5_probs):
                idx_val = idx.item()
                if 0 <= idx_val < len(self.vocab_list):
                    word = self.vocab_list[idx_val]
                else:
                    word = f"CLASS_{idx_val}"
                top5_words.append((word, prob.item()))
            
            # Get predicted word with bounds checking
            if 0 <= predicted_idx < len(self.vocab_list):
                predicted_word = self.vocab_list[predicted_idx]
            else:
                predicted_word = f"CLASS_{predicted_idx}"
            
            return predicted_word, confidence, top5_words
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, []
    
    def run(self):
        """Main loop for real-time lip reading"""
        print("Opening camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nCamera opened successfully!")
        print("Testing LRW pre-trained model (500 words)")
        print("Press 'SPACE' to start recording 70 frames")
        print("Press 'q' to quit")
        print("Press 't' to show/hide top 5 predictions")
        
        current_word = ""
        confidence = 0.0
        top5_words = []
        show_top5 = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
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
                            pred_word, conf, top5 = self.predict()
                            
                            if pred_word:
                                current_word = pred_word
                                confidence = conf
                                top5_words = top5
                                print(f"Prediction: {current_word} ({confidence:.2%})")
                
                # Display
                display_frame = annotated_frame.copy()
                
                # FPS
                self.fps_counter += 1
                fps = 30 / max(time.time() - self.fps_time, 0.001)
                if self.fps_counter % 30 == 0:
                    self.fps_time = time.time()
                
                # Text overlay
                cv2.putText(display_frame, f"LRW Model (500 words)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (500, 30), 
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
                    
                    # Show top 5 predictions
                    if show_top5 and top5_words:
                        cv2.putText(display_frame, "Top 5:", (10, 190), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        for i, (word, prob) in enumerate(top5_words):
                            cv2.putText(display_frame, f"{i+1}. {word}: {prob:.2%}", (10, 220 + i*25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow('LRW Pre-trained Model Test', display_frame)
                
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
                        top5_words = []
                        print("Started recording...")
                elif key == ord('t'):
                    show_top5 = not show_top5
                    print(f"Top 5 display: {'ON' if show_top5 else 'OFF'}")
                    
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
    parser = argparse.ArgumentParser(description='Test LRW pre-trained model')
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/lrw-cosine-lr-acc-0.85080.pt',
                       help='Path to the pre-trained LRW model')
    args = parser.parse_args()
    
    try:
        lip_reader = LRWLipReader(args.model_path)
        lip_reader.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()