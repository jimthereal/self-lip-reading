"""
Custom Dataset Preprocessor for Self-Recorded Lip Reading Videos
Processes your 30-word dataset with proper train/validation/test splits
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import warnings
import random
from sklearn.model_selection import train_test_split
import mediapipe as mp
warnings.filterwarnings('ignore')

class CustomLipReadingProcessor:
    """Processor for self-recorded lip reading dataset"""
    
    def __init__(self, dataset_path="my_lipreading_dataset", output_path="processed_custom_dataset"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.metadata_file = self.dataset_path / "metadata.json"
        
        # Video processing parameters (matching GRID setup)
        self.target_size = (128, 64)  # Width x Height for lip region
        self.sequence_length = 75  # Fixed sequence length
        
        # Split ratios
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Load metadata
        self.load_metadata()
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        print("Custom Dataset Processor Initialized")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output path: {self.output_path}")
        print(f"Train/Val/Test split: {self.train_ratio:.0%}/{self.val_ratio:.0%}/{self.test_ratio:.0%}")
    
    def load_metadata(self):
        """Load metadata from JSON file"""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_file}")
        
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded metadata for {len(self.metadata)} words")
        total_videos = sum(len(videos) for videos in self.metadata.values())
        print(f"Total videos: {total_videos}")
    
    def extract_lip_region_advanced(self, frame):
        """Extract lip region using MediaPipe Face Mesh for better accuracy"""
        if frame is None:
            return None
        
        try:
            # Convert to RGB
            if len(frame.shape) == 2:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with face mesh
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                # Lip landmarks indices in MediaPipe
                # Upper lip: 61, 84, 17, 314, 405, 308, 324, 318
                # Lower lip: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324
                lip_indices = [61, 84, 17, 314, 405, 308, 324, 318,
                              78, 95, 88, 178, 87, 14, 317, 402]
                
                # Get lip coordinates
                lip_points = []
                for idx in lip_indices:
                    if idx < len(landmarks.landmark):
                        point = landmarks.landmark[idx]
                        x = int(point.x * w)
                        y = int(point.y * h)
                        lip_points.append((x, y))
                
                if lip_points:
                    # Find bounding box
                    xs = [p[0] for p in lip_points]
                    ys = [p[1] for p in lip_points]
                    
                    x_min = max(0, min(xs) - 20)
                    x_max = min(w, max(xs) + 20)
                    y_min = max(0, min(ys) - 10)
                    y_max = min(h, max(ys) + 10)
                    
                    # Extract and process lip region
                    if len(frame.shape) == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = frame
                    
                    lip_region = gray[y_min:y_max, x_min:x_max]
                    
                    if lip_region.size > 0:
                        # Resize to target size
                        lip_resized = cv2.resize(lip_region, self.target_size, 
                                               interpolation=cv2.INTER_CUBIC)
                        # Apply histogram equalization for better contrast
                        return cv2.equalizeHist(lip_resized)
            
            # Fall back to face detection method
            return self.extract_lip_region_simple(frame)
            
        except Exception as e:
            # Fall back to simple method
            return self.extract_lip_region_simple(frame)
    
    def extract_lip_region_simple(self, frame):
        """Simple lip extraction using face detection"""
        if frame is None:
            return None
        
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                gray = frame
                rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # Face detection
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = gray.shape[:2]
                
                # Calculate face region
                face_x = int(bbox.xmin * w)
                face_y = int(bbox.ymin * h)
                face_w = int(bbox.width * w)
                face_h = int(bbox.height * h)
                
                # Estimate mouth region (lower 40% of face)
                mouth_y = face_y + int(face_h * 0.6)
                mouth_h = int(face_h * 0.4)
                mouth_x = face_x + int(face_w * 0.2)
                mouth_w = int(face_w * 0.6)
                
                # Ensure valid bounds
                mouth_x = max(0, mouth_x)
                mouth_y = max(0, mouth_y)
                mouth_x2 = min(w, mouth_x + mouth_w)
                mouth_y2 = min(h, mouth_y + mouth_h)
                
                if mouth_x2 > mouth_x and mouth_y2 > mouth_y:
                    mouth_region = gray[mouth_y:mouth_y2, mouth_x:mouth_x2]
                    
                    if mouth_region.size > 0:
                        mouth_resized = cv2.resize(mouth_region, self.target_size, 
                                                 interpolation=cv2.INTER_CUBIC)
                        return cv2.equalizeHist(mouth_resized)
            
            # Fallback to geometric center
            return self.extract_geometric_center(gray)
            
        except Exception:
            return self.extract_geometric_center(gray)
    
    def extract_geometric_center(self, gray):
        """Fallback geometric extraction"""
        if gray is None:
            return None
        
        h, w = gray.shape
        
        # Take center region
        center_y = h // 2
        center_x = w // 2
        
        # Define region around center
        region_h = min(h // 3, 100)
        region_w = min(w // 2, 150)
        
        y1 = max(0, center_y - region_h // 2)
        y2 = min(h, center_y + region_h // 2)
        x1 = max(0, center_x - region_w // 2)
        x2 = min(w, center_x + region_w // 2)
        
        mouth_region = gray[y1:y2, x1:x2]
        
        if mouth_region.size > 0:
            mouth_resized = cv2.resize(mouth_region, self.target_size, 
                                     interpolation=cv2.INTER_CUBIC)
            return cv2.equalizeHist(mouth_resized)
        else:
            return None
    
    def process_video(self, video_path, word, recording_idx):
        """Process a single video file"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error: Cannot open video {video_path}")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                print(f"Error: Video has 0 frames: {video_path}")
                cap.release()
                return None
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.sequence_length).astype(int)
            
            frames = []
            successful_extractions = 0
            
            for idx, target_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Try advanced extraction first, then simple
                    lip_region = self.extract_lip_region_advanced(frame)
                    
                    if lip_region is not None:
                        frames.append(lip_region)
                        successful_extractions += 1
                    else:
                        # Use last valid frame or zeros
                        if frames:
                            frames.append(frames[-1].copy())
                        else:
                            frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
                else:
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
            
            cap.release()
            
            # Quality check
            if successful_extractions < self.sequence_length * 0.5:
                print(f"Warning: Low extraction rate for {video_path.name}")
                return None
            
            # Ensure correct length
            frames = frames[:self.sequence_length]
            while len(frames) < self.sequence_length:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
            
            # Convert to tensor
            frames_array = np.stack(frames).astype(np.float32)
            frames_normalized = frames_array / 255.0
            video_tensor = torch.FloatTensor(frames_normalized).unsqueeze(0)  # [1, T, H, W]
            
            return {
                'video': video_tensor,
                'sentence': word,  # Single word is the "sentence"
                'word': word,
                'video_path': str(video_path),
                'recording_idx': recording_idx,
                'fps': fps,
                'extraction_rate': successful_extractions / self.sequence_length
            }
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return None
    
    def split_dataset(self):
        """Split dataset into train/val/test sets - ensuring each word is represented in all splits"""
        all_samples = []
        
        # Collect all video info
        for word, recordings in self.metadata.items():
            for idx, recording in enumerate(recordings):
                video_path = Path(recording['filepath'])
                if video_path.exists():
                    all_samples.append({
                        'word': word,
                        'video_path': video_path,
                        'recording_idx': idx,
                        'metadata': recording
                    })
                else:
                    print(f"Warning: Video not found: {recording['filepath']}")
        
        print(f"\nTotal valid videos: {len(all_samples)}")
        
        # Group by word for stratified splitting
        samples_by_word = {}
        for sample in all_samples:
            word = sample['word']
            if word not in samples_by_word:
                samples_by_word[word] = []
            samples_by_word[word].append(sample)
        
        # Show distribution
        print("\nVideos per word:")
        for word in sorted(samples_by_word.keys()):
            print(f"  {word}: {len(samples_by_word[word])} videos")
        
        # Split each word's samples individually
        train_samples = []
        val_samples = []
        test_samples = []
        
        print("\nSplitting each word individually...")
        for word, word_samples in samples_by_word.items():
            # Shuffle samples for this word
            random.shuffle(word_samples)
            
            n_samples = len(word_samples)
            
            # Calculate splits for this word
            # Ensure at least 1 sample in each split if possible
            if n_samples >= 3:
                n_train = max(1, int(n_samples * self.train_ratio))
                n_val = max(1, int(n_samples * self.val_ratio))
                n_test = n_samples - n_train - n_val
                
                # Adjust if needed
                if n_test < 1 and n_train > 1:
                    n_train -= 1
                    n_test = 1
            else:
                # Too few samples, put most in train
                n_train = n_samples
                n_val = 0
                n_test = 0
                print(f"  Warning: {word} has only {n_samples} samples, all going to train")
            
            # Split
            train_samples.extend(word_samples[:n_train])
            if n_val > 0:
                val_samples.extend(word_samples[n_train:n_train + n_val])
            if n_test > 0:
                test_samples.extend(word_samples[n_train + n_val:])
            
            print(f"  {word}: train={n_train}, val={n_val}, test={n_test}")
        
        # Shuffle final sets
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        print(f"\nFinal dataset split:")
        print(f"  Train: {len(train_samples)} videos")
        print(f"  Val: {len(val_samples)} videos")
        print(f"  Test: {len(test_samples)} videos")
        
        # Verify each word appears in train
        train_words = set(s['word'] for s in train_samples)
        val_words = set(s['word'] for s in val_samples) if val_samples else set()
        test_words = set(s['word'] for s in test_samples) if test_samples else set()
        
        print(f"\nUnique words in train: {len(train_words)}")
        print(f"Unique words in val: {len(val_words)}")
        print(f"Unique words in test: {len(test_words)}")
        
        return train_samples, val_samples, test_samples
    
    def process_dataset(self):
        """Process entire dataset"""
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_path / split).mkdir(parents=True, exist_ok=True)
        
        # Split dataset with improved strategy
        train_samples, val_samples, test_samples = self.split_dataset()
        
        # Process each split
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        all_results = {}
        
        for split_name, samples in splits.items():
            if not samples:
                print(f"\nSkipping {split_name} split (no samples)")
                all_results[split_name] = {'successful': 0, 'failed': 0}
                continue
                
            print(f"\nProcessing {split_name} split ({len(samples)} videos)...")
            split_dir = self.output_path / split_name
            
            successful = 0
            failed = 0
            failed_videos = []
            
            # Process with progress bar
            for i, sample_info in enumerate(samples):
                # Show progress every 10 videos
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(samples)} ({successful} successful, {failed} failed)")
                
                try:
                    # Process video
                    result = self.process_video(
                        sample_info['video_path'],
                        sample_info['word'],
                        sample_info['recording_idx']
                    )
                    
                    if result is not None:
                        # Save processed sample
                        sample_file = split_dir / f"{split_name}_{successful:05d}.pkl"
                        with open(sample_file, 'wb') as f:
                            pickle.dump(result, f)
                        successful += 1
                    else:
                        failed += 1
                        failed_videos.append(str(sample_info['video_path']))
                        
                except Exception as e:
                    print(f"    Error processing {sample_info['video_path']}: {str(e)}")
                    failed += 1
                    failed_videos.append(str(sample_info['video_path']))
            
            print(f"  Final: {successful} successful, {failed} failed")
            
            if failed > 0 and len(failed_videos) <= 10:
                print(f"  Failed videos: {failed_videos}")
            
            all_results[split_name] = {'successful': successful, 'failed': failed}
        
        # Save processing info
        info = {
            'vocabulary': sorted(list(self.metadata.keys())),
            'vocab_size': len(self.metadata.keys()),
            'splits': all_results,
            'target_size': self.target_size,
            'sequence_length': self.sequence_length,
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            }
        }
        
        with open(self.output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print(f"Vocabulary size: {len(self.metadata.keys())} words")
        print(f"Train samples: {all_results.get('train', {}).get('successful', 0)}")
        print(f"Val samples: {all_results.get('val', {}).get('successful', 0)}")
        print(f"Test samples: {all_results.get('test', {}).get('successful', 0)}")
        
        total_processed = sum(r['successful'] for r in all_results.values())
        total_failed = sum(r['failed'] for r in all_results.values())
        print(f"\nTotal processed: {total_processed}/{total_processed + total_failed}")
        
        if total_processed == 0:
            print("\nERROR: No videos were processed successfully!")
            print("Check that video paths are correct and videos are readable.")
        
        return self.output_path

def main():
    """Main execution"""
    print("CUSTOM LIP READING DATASET PROCESSING")
    print("="*60)
    
    # First, clean metadata if needed
    print("\nDo you want to clean metadata first? (y/n): ", end='')
    if input().strip().lower() == 'y':
        print("Please run the metadata_cleaner.py first, then come back.")
        return
    
    # Ask for test mode
    print("\nRun in test mode? (process only 1 video per word) (y/n): ", end='')
    test_mode = input().strip().lower() == 'y'
    
    # Process dataset
    processor = CustomLipReadingProcessor()
    
    if test_mode:
        print("\nTEST MODE: Processing only 1 video per word...")
        # Temporarily modify metadata to include only first video per word
        original_metadata = processor.metadata.copy()
        for word in processor.metadata:
            processor.metadata[word] = processor.metadata[word][:1]
        
        # Process with modified metadata
        output_path = processor.process_dataset()
        
        # Restore original metadata
        processor.metadata = original_metadata
    else:
        # Normal processing
        output_path = processor.process_dataset()
    
    print("\nNext steps:")
    print("1. Check the processed files in:", output_path)
    print("2. If successful, train the model with: python train_model.py")

if __name__ == "__main__":
    main()