"""
Data Augmentation for Custom Lip Reading Dataset
Applies various augmentation techniques to increase dataset size before train/val/test split
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import random
import shutil
from datetime import datetime

class LipReadingAugmenter:
    """Augments lip reading video dataset"""
    
    def __init__(self, source_path="data/my_lipreading_dataset", 
                 augmented_path="data/augmented_lipreading_dataset",
                 augmentation_factor=3):
        self.source_path = Path(source_path)
        self.augmented_path = Path(augmented_path)
        self.augmentation_factor = augmentation_factor  # How many augmented versions per original
        
        # Load original metadata
        self.metadata_file = self.source_path / "metadata.json"
        with open(self.metadata_file, 'r') as f:
            self.original_metadata = json.load(f)
        
        # Augmentation parameters
        self.augmentations = {
            'brightness': {'min': 0.6, 'max': 1.4},
            'contrast': {'min': 0.7, 'max': 1.3},
            'blur': {'kernel_sizes': [(3,3), (5,5)]},
            'noise': {'intensity': [0.01, 0.02, 0.03]},
            'flip': {'probability': 0.3},  # Horizontal flip
            'rotation': {'angles': [-5, -3, 3, 5]},  # Small rotations
            'zoom': {'scales': [0.9, 0.95, 1.05, 1.1]},  # Slight zoom in/out
            'temporal_speed': {'factors': [0.9, 1.1]}  # Slight speed variations
        }
        
        print(f"Augmenter initialized")
        print(f"Source: {self.source_path}")
        print(f"Target: {self.augmented_path}")
        print(f"Augmentation factor: {self.augmentation_factor}x")
        print(f"Source path exists: {self.source_path.exists()}")
        print(f"Metadata file exists: {self.metadata_file.exists()}")
        print(f"Number of words in metadata: {len(self.original_metadata)}")
    
    def adjust_brightness(self, frame, factor):
        """Adjust brightness of frame"""
        return cv2.convertScaleAbs(frame, alpha=factor, beta=0)
    
    def adjust_contrast(self, frame, factor):
        """Adjust contrast of frame"""
        mean = np.mean(frame)
        return cv2.convertScaleAbs(frame, alpha=factor, beta=(1-factor)*mean)
    
    def add_gaussian_noise(self, frame, intensity):
        """Add Gaussian noise to frame"""
        h, w = frame.shape[:2]
        noise = np.random.randn(h, w) * 255 * intensity
        
        if len(frame.shape) == 3:
            noise = np.stack([noise]*3, axis=-1)
        
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_blur(self, frame, kernel_size):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(frame, kernel_size, 0)
    
    def rotate_frame(self, frame, angle):
        """Rotate frame by angle degrees"""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h))
    
    def zoom_frame(self, frame, scale):
        """Zoom in/out of frame"""
        h, w = frame.shape[:2]
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        
        if scale > 1:  # Zoom in - crop center
            resized = cv2.resize(frame, (new_w, new_h))
            y_start = (new_h - h) // 2
            x_start = (new_w - w) // 2
            return resized[y_start:y_start+h, x_start:x_start+w]
        else:  # Zoom out - pad
            resized = cv2.resize(frame, (new_w, new_h))
            canvas = np.zeros((h, w, 3) if len(frame.shape) == 3 else (h, w), dtype=frame.dtype)
            y_start = (h - new_h) // 2
            x_start = (w - new_w) // 2
            canvas[y_start:y_start+new_h, x_start:x_start+new_w] = resized
            return canvas
    
    def augment_video(self, video_path, augmentation_params):
        """Apply augmentations to a video"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"Warning: {video_path} has 0 frames")
            cap.release()
            return None
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        # Apply temporal speed change if specified
        if 'temporal_speed' in augmentation_params:
            speed_factor = augmentation_params['temporal_speed']
            if speed_factor != 1.0:
                # Resample frames
                original_indices = np.arange(len(frames))
                new_length = int(len(frames) / speed_factor)
                new_indices = np.linspace(0, len(frames)-1, new_length)
                
                resampled_frames = []
                for idx in new_indices:
                    frame_idx = int(idx)
                    if frame_idx < len(frames):
                        resampled_frames.append(frames[frame_idx])
                frames = resampled_frames
        
        # Apply frame-wise augmentations
        augmented_frames = []
        for frame in frames:
            aug_frame = frame.copy()
            
            # Brightness
            if 'brightness' in augmentation_params:
                aug_frame = self.adjust_brightness(aug_frame, augmentation_params['brightness'])
            
            # Contrast
            if 'contrast' in augmentation_params:
                aug_frame = self.adjust_contrast(aug_frame, augmentation_params['contrast'])
            
            # Blur
            if 'blur' in augmentation_params:
                aug_frame = self.apply_blur(aug_frame, augmentation_params['blur'])
            
            # Noise
            if 'noise' in augmentation_params:
                aug_frame = self.add_gaussian_noise(aug_frame, augmentation_params['noise'])
            
            # Rotation
            if 'rotation' in augmentation_params:
                aug_frame = self.rotate_frame(aug_frame, augmentation_params['rotation'])
            
            # Zoom
            if 'zoom' in augmentation_params:
                aug_frame = self.zoom_frame(aug_frame, augmentation_params['zoom'])
            
            # Horizontal flip
            if 'flip' in augmentation_params and augmentation_params['flip']:
                aug_frame = cv2.flip(aug_frame, 1)
            
            augmented_frames.append(aug_frame)
        
        return augmented_frames, fps, (width, height)
    
    def generate_augmentation_params(self):
        """Generate random augmentation parameters"""
        params = {}
        
        # Each augmentation has a probability of being applied
        if random.random() > 0.3:
            params['brightness'] = random.uniform(
                self.augmentations['brightness']['min'],
                self.augmentations['brightness']['max']
            )
        
        if random.random() > 0.3:
            params['contrast'] = random.uniform(
                self.augmentations['contrast']['min'],
                self.augmentations['contrast']['max']
            )
        
        if random.random() > 0.5:
            params['blur'] = random.choice(self.augmentations['blur']['kernel_sizes'])
        
        if random.random() > 0.4:
            params['noise'] = random.choice(self.augmentations['noise']['intensity'])
        
        if random.random() < self.augmentations['flip']['probability']:
            params['flip'] = True
        
        if random.random() > 0.5:
            params['rotation'] = random.choice(self.augmentations['rotation']['angles'])
        
        if random.random() > 0.5:
            params['zoom'] = random.choice(self.augmentations['zoom']['scales'])
        
        if random.random() > 0.6:
            params['temporal_speed'] = random.choice(self.augmentations['temporal_speed']['factors'])
        
        return params
    
    def save_video(self, frames, output_path, fps, size):
        """Save frames as video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def augment_dataset(self):
        """Augment entire dataset"""
        # Create augmented dataset directory
        if self.augmented_path.exists():
            print(f"Removing existing augmented dataset...")
            shutil.rmtree(self.augmented_path)
        
        self.augmented_path.mkdir(parents=True, exist_ok=True)
        
        # New metadata
        augmented_metadata = {}
        
        print(f"\nStarting augmentation process...")
        print(f"Original videos: {sum(len(recordings) for recordings in self.original_metadata.values())}")
        print(f"Target augmented videos: {sum(len(recordings) for recordings in self.original_metadata.values()) * (1 + self.augmentation_factor)}")
        
        # Process each word
        for word, recordings in tqdm(self.original_metadata.items(), desc="Processing words"):
            word_dir = self.augmented_path / word
            word_dir.mkdir(exist_ok=True)
            
            augmented_metadata[word] = []
            
            # Process each recording
            for rec_idx, recording in enumerate(tqdm(recordings, desc=f"  {word}", leave=False)):
                # Fix the filepath - construct it properly
                # The filepath in metadata includes the dataset folder, so we need just the relative part
                filepath_parts = recording['filepath'].replace('\\', '/').split('/')
                if filepath_parts[0] == 'my_lipreading_dataset':
                    filepath_parts = filepath_parts[1:]  # Remove the dataset folder name
                original_path = self.source_path / '/'.join(filepath_parts)
                
                # Debug output
                if rec_idx == 0:  # Only print for first file of each word
                    print(f"\n  Looking for: {original_path}")
                    print(f"  Exists: {original_path.exists()}")
                
                if not original_path.exists():
                    print(f"Warning: {original_path} not found")
                    continue
                
                # Copy original video
                original_name = f"{word}_{rec_idx:03d}_original.mp4"
                original_dest = word_dir / original_name
                shutil.copy2(original_path, original_dest)
                
                # Add to metadata - preserve original format
                augmented_metadata[word].append({
                    'filename': original_name,
                    'filepath': f"{word}/{original_name}",
                    'timestamp': recording.get('timestamp', datetime.now().isoformat()),
                    'duration': recording.get('duration', 1.5),
                    'frames': recording.get('frames', 45),
                    'fps': recording.get('fps', 30),
                    'augmented': False,
                    'original_index': rec_idx,
                    'augmentation_type': 'original'
                })
                
                # Generate augmented versions
                for aug_idx in range(self.augmentation_factor):
                    # Generate random augmentation parameters
                    aug_params = self.generate_augmentation_params()
                    
                    # Apply augmentation
                    result = self.augment_video(original_path, aug_params)
                    if result is None:
                        continue
                    
                    aug_frames, fps, size = result
                    
                    # Save augmented video
                    aug_name = f"{word}_{rec_idx:03d}_aug_{aug_idx:02d}.mp4"
                    aug_path = word_dir / aug_name
                    self.save_video(aug_frames, aug_path, fps, size)
                    
                    # Add to metadata
                    augmented_metadata[word].append({
                        'filename': aug_name,
                        'filepath': f"{word}/{aug_name}",
                        'timestamp': datetime.now().isoformat(),
                        'duration': recording.get('duration', 1.5),
                        'frames': len(aug_frames),
                        'fps': fps,
                        'augmented': True,
                        'original_index': rec_idx,
                        'augmentation_type': 'augmented',
                        'augmentation_params': aug_params
                    })
        
        # Save augmented metadata
        metadata_path = self.augmented_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(augmented_metadata, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("AUGMENTATION COMPLETE!")
        print("="*60)
        
        total_original = sum(len([r for r in recordings if not r.get('augmented', False)]) 
                           for recordings in augmented_metadata.values())
        total_augmented = sum(len([r for r in recordings if r.get('augmented', False)]) 
                            for recordings in augmented_metadata.values())
        total_videos = sum(len(recordings) for recordings in augmented_metadata.values())
        
        print(f"Original videos: {total_original}")
        print(f"Augmented videos: {total_augmented}")
        print(f"Total videos: {total_videos}")
        
        if total_original > 0:
            print(f"Augmentation ratio: {total_videos / total_original:.1f}x")
        else:
            print("No videos were processed!")
        
        print("\nVideos per word:")
        for word in sorted(augmented_metadata.keys()):
            count = len(augmented_metadata[word])
            print(f"  {word}: {count} videos")
        
        return self.augmented_path

def main():
    """Main execution"""
    print("LIP READING DATASET AUGMENTATION")
    print("="*60)
    
    # Get the correct base path
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # Go up one level from scripts/
    
    source_path = base_dir / "data" / "my_lipreading_dataset"
    augmented_path = base_dir / "data" / "augmented_lipreading_dataset"
    
    print(f"\nScript directory: {script_dir}")
    print(f"Base directory: {base_dir}")
    print(f"Looking for dataset at: {source_path}")
    print(f"Dataset exists: {source_path.exists()}")
    
    if not source_path.exists():
        print(f"\nERROR: Dataset not found at {source_path}")
        print("Please ensure your directory structure is:")
        print("  self_dataset/")
        print("    scripts/")
        print("      augment_dataset.py")
        print("    data/")
        print("      my_lipreading_dataset/")
        print("        metadata.json")
        print("        BAD/")
        print("        BYE/")
        print("        ...")
        return
    
    # Get augmentation factor from user
    print("\nHow many augmented versions per original video?")
    print("(Recommended: 2-4 for your dataset size)")
    aug_factor = input("Augmentation factor [3]: ").strip()
    aug_factor = int(aug_factor) if aug_factor else 3
    
    # Create augmenter
    augmenter = LipReadingAugmenter(
        source_path=source_path,
        augmented_path=augmented_path,
        augmentation_factor=aug_factor
    )
    
    # Run augmentation
    augmented_path = augmenter.augment_dataset()
    
    print(f"\nAugmented dataset saved to: {augmented_path}")
    print("\nNext steps:")
    print("1. Update your preprocessor to use the augmented dataset")
    print("2. Run the preprocessing script on the augmented dataset")
    print("3. Train your model with the larger dataset")

if __name__ == "__main__":
    main()