"""
Custom Dataset Preprocessor for Augmented Lip Reading Videos - Fixed Version
Processes the augmented dataset with proper train/validation/test splits
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
import mediapipe as mp
import sys
import traceback
warnings.filterwarnings('ignore')

class AugmentedLipReadingProcessor:
    """Processor for augmented lip reading dataset"""
    
    def __init__(self, dataset_path="data/augmented_lipreading_dataset", 
                 output_path="data/processed_augmented_dataset"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.metadata_file = self.dataset_path / "metadata.json"
        
        # Video processing parameters (matching GRID setup)
        self.target_size = (128, 64)  # Width x Height for lip region
        self.sequence_length = 75  # Fixed sequence length
        
        # Split ratios - adjusted for augmented dataset
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        
        # Load metadata
        self.load_metadata()
        
        # Initialize MediaPipe
        print("Initializing MediaPipe...")
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
        
        print("Augmented Dataset Processor Initialized")
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
        
        # Count original vs augmented
        original_count = sum(len([v for v in videos if not v.get('augmented', False)]) 
                           for videos in self.metadata.values())
        augmented_count = total_videos - original_count
        print(f"  Original: {original_count}")
        print(f"  Augmented: {augmented_count}")
    
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
            return self.extract_geometric_center(frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    
    def extract_geometric_center(self, gray):
        """Fallback geometric extraction"""
        if gray is None:
            return None
        
        # Ensure grayscale
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
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
    
    def process_video(self, video_path, word, video_info):
        """Process a single video file"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"\nError: Cannot open video {video_path}")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                print(f"\nError: Video has 0 frames: {video_path}")
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
                print(f"\nWarning: Low extraction rate for {video_path.name}")
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
                'sentence': word,
                'word': word,
                'video_path': str(video_path),
                'video_info': video_info,
                'fps': fps,
                'extraction_rate': successful_extractions / self.sequence_length
            }
            
        except Exception as e:
            print(f"\nError processing video {video_path}: {str(e)}")
            traceback.print_exc()
            return None
    
    def split_dataset_smart(self):
        """Smart split that ensures original videos are distributed across splits"""
        all_samples = []
        
        # Collect all video info
        for word, recordings in self.metadata.items():
            for video_info in recordings:
                video_path = self.dataset_path / video_info['filepath']
                if video_path.exists():
                    all_samples.append({
                        'word': word,
                        'video_path': video_path,
                        'video_info': video_info,
                        'is_original': not video_info.get('augmented', False),
                        'original_index': video_info.get('original_index', -1)
                    })
                else:
                    print(f"Warning: Video not found: {video_info['filepath']}")
        
        print(f"\nTotal valid videos: {len(all_samples)}")
        
        # Group by word and original index
        samples_by_word_and_original = {}
        for sample in all_samples:
            word = sample['word']
            orig_idx = sample['original_index']
            key = (word, orig_idx)
            
            if key not in samples_by_word_and_original:
                samples_by_word_and_original[key] = []
            samples_by_word_and_original[key].append(sample)
        
        # Now split by original video groups
        train_samples = []
        val_samples = []
        test_samples = []
        
        print("\nSplitting dataset by original video groups...")
        
        # Get all unique original indices for each word
        for word in sorted(set(s['word'] for s in all_samples)):
            word_groups = [(k, v) for k, v in samples_by_word_and_original.items() if k[0] == word]
            
            # Shuffle groups for this word
            random.shuffle(word_groups)
            
            n_groups = len(word_groups)
            n_train = int(n_groups * self.train_ratio)
            n_val = int(n_groups * self.val_ratio)
            n_test = n_groups - n_train - n_val
            
            # Ensure at least one group in each split if possible
            if n_groups >= 3:
                n_train = max(1, n_train)
                n_val = max(1, n_val)
                n_test = max(1, n_test)
                
                # Adjust if needed
                total_assigned = n_train + n_val + n_test
                if total_assigned > n_groups:
                    n_train = n_groups - n_val - n_test
            
            # Assign groups to splits
            train_groups = word_groups[:n_train]
            val_groups = word_groups[n_train:n_train + n_val]
            test_groups = word_groups[n_train + n_val:]
            
            # Add all samples from each group to respective splits
            for _, samples in train_groups:
                train_samples.extend(samples)
            for _, samples in val_groups:
                val_samples.extend(samples)
            for _, samples in test_groups:
                test_samples.extend(samples)
            
            print(f"  {word}: {n_groups} original videos -> train={n_train}, val={n_val}, test={n_test}")
        
        # Shuffle final sets
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        print(f"\nFinal dataset split:")
        print(f"  Train: {len(train_samples)} videos")
        print(f"  Val: {len(val_samples)} videos")
        print(f"  Test: {len(test_samples)} videos")
        
        # Verify distribution
        for split_name, split_samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
            if split_samples:
                original_count = sum(1 for s in split_samples if s['is_original'])
                augmented_count = len(split_samples) - original_count
                print(f"\n{split_name} breakdown:")
                print(f"  Original: {original_count}")
                print(f"  Augmented: {augmented_count}")
        
        return train_samples, val_samples, test_samples
    
    def process_dataset(self):
        """Process entire augmented dataset"""
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_path / split).mkdir(parents=True, exist_ok=True)
        
        # Split dataset with smart strategy
        train_samples, val_samples, test_samples = self.split_dataset_smart()
        
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
            
            # Process videos one by one with explicit progress
            for i, sample_info in enumerate(samples):
                # Print progress every 10 videos
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(samples)} ({i/len(samples)*100:.1f}%)")
                    sys.stdout.flush()
                
                try:
                    # Process video
                    result = self.process_video(
                        sample_info['video_path'],
                        sample_info['word'],
                        sample_info['video_info']
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
                    print(f"\n  Error processing {sample_info['video_path']}: {str(e)}")
                    traceback.print_exc()
                    failed += 1
                    failed_videos.append(str(sample_info['video_path']))
            
            print(f"\n  {split_name} complete: {successful} successful, {failed} failed")
            
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
            },
            'augmented': True
        }
        
        with open(self.output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("AUGMENTED DATASET PROCESSING COMPLETE!")
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
        
        # Clean up MediaPipe
        self.face_detection.close()
        self.face_mesh.close()
        
        return self.output_path

def main():
    """Main execution"""
    print("AUGMENTED LIP READING DATASET PROCESSING")
    print("="*60)
    
    # Get the correct base path
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # Go up one level from scripts/
    
    augmented_path = base_dir / "data" / "augmented_lipreading_dataset"
    output_path = base_dir / "data" / "processed_augmented_dataset"
    
    print(f"\nScript directory: {script_dir}")
    print(f"Base directory: {base_dir}")
    print(f"Looking for augmented dataset at: {augmented_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Check if augmented dataset exists
    if not augmented_path.exists():
        print("\nERROR: Augmented dataset not found!")
        print("Please run the augmentation script first.")
        return
    
    try:
        # Process dataset
        processor = AugmentedLipReadingProcessor(
            dataset_path=augmented_path,
            output_path=output_path
        )
        
        output_path = processor.process_dataset()
        
        print("\nNext steps:")
        print("1. Check the processed files in:", output_path)
        print("2. Train the model with the augmented dataset")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()