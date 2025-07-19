import os
import json
from datetime import datetime

class MetadataCleaner:
    def __init__(self, dataset_dir='my_lipreading_dataset'):
        self.dataset_dir = dataset_dir
        self.metadata_file = os.path.join(dataset_dir, 'metadata.json')
        
        # Backup original metadata
        backup_file = os.path.join(dataset_dir, f'metadata_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            # Create backup
            with open(backup_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"Backup created: {backup_file}")
        else:
            print("No metadata file found!")
            self.metadata = {}
    
    def clean_metadata(self):
        """Remove entries for files that don't exist"""
        if not self.metadata:
            print("No metadata to clean")
            return
        
        print("\nScanning for missing files...")
        
        cleaned_metadata = {}
        total_removed = 0
        
        for word, recordings in self.metadata.items():
            cleaned_recordings = []
            
            for recording in recordings:
                filepath = recording.get('filepath', '')
                
                # Check if file exists
                if os.path.exists(filepath):
                    cleaned_recordings.append(recording)
                else:
                    print(f"  Missing file: {filepath}")
                    total_removed += 1
            
            if cleaned_recordings:
                cleaned_metadata[word] = cleaned_recordings
        
        self.metadata = cleaned_metadata
        
        print(f"\nRemoved {total_removed} missing file entries")
        
        # Save cleaned metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print("Metadata cleaned and saved!")
    
    def validate_and_renumber(self):
        """Validate files and optionally renumber them sequentially"""
        print("\nValidating and organizing files...")
        
        updated_metadata = {}
        
        for word in sorted(self.metadata.keys()):
            word_dir = os.path.join(self.dataset_dir, word)
            
            if not os.path.exists(word_dir):
                print(f"Warning: Directory missing for word '{word}'")
                continue
            
            # Get all video files in directory
            video_files = [f for f in os.listdir(word_dir) if f.endswith('.mp4')]
            
            # Check metadata consistency
            metadata_files = [rec['filename'] for rec in self.metadata[word]]
            
            # Find orphaned files (in folder but not in metadata)
            orphaned = set(video_files) - set(metadata_files)
            if orphaned:
                print(f"\n{word}: Found {len(orphaned)} orphaned files not in metadata:")
                for f in orphaned:
                    print(f"  - {f}")
            
            # Sort recordings by timestamp
            recordings = sorted(self.metadata[word], 
                              key=lambda x: x.get('timestamp', ''))
            
            updated_metadata[word] = recordings
            
            print(f"{word}: {len(recordings)} valid recordings")
        
        self.metadata = updated_metadata
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def print_summary(self):
        """Print current dataset summary"""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        total_recordings = sum(len(recordings) for recordings in self.metadata.values())
        print(f"Total words: {len(self.metadata)}")
        print(f"Total recordings: {total_recordings}")
        
        print("\nRecordings per word:")
        print("-"*40)
        
        # Assuming we want 20 recordings per word now
        target = 20
        
        for word in sorted(self.metadata.keys()):
            count = len(self.metadata[word])
            status = "✓" if count >= target else f"Need {target - count} more"
            bar = "█" * min(count, target) + "░" * max(0, target - count)
            print(f"{word:12} : {count:2}/{target} [{bar}] {status}")
        
        complete = sum(1 for word in self.metadata if len(self.metadata[word]) >= target)
        print("-"*40)
        print(f"Words complete ({target}+ recordings): {complete}/{len(self.metadata)}")
        print("="*60)
    
    def add_orphaned_files_to_metadata(self):
        """Add video files that exist but aren't in metadata"""
        print("\nScanning for orphaned video files...")
        
        added_count = 0
        
        for word_dir in os.listdir(self.dataset_dir):
            word_path = os.path.join(self.dataset_dir, word_dir)
            
            if os.path.isdir(word_path) and word_dir.upper() == word_dir:
                word = word_dir
                
                # Get all video files
                video_files = [f for f in os.listdir(word_path) if f.endswith('.mp4')]
                
                if word not in self.metadata:
                    self.metadata[word] = []
                
                # Get existing filenames in metadata
                existing_files = [rec['filename'] for rec in self.metadata[word]]
                
                # Find orphaned files
                for video_file in video_files:
                    if video_file not in existing_files:
                        filepath = os.path.join(word_path, video_file)
                        
                        # Get file info
                        file_stat = os.stat(filepath)
                        
                        # Create metadata entry
                        new_entry = {
                            'filename': video_file,
                            'filepath': filepath,
                            'timestamp': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                            'duration': 1.5,  # Default duration
                            'fps': 30,  # Default fps
                            'frames': 45  # Default frames (1.5s * 30fps)
                        }
                        
                        self.metadata[word].append(new_entry)
                        added_count += 1
                        print(f"  Added: {word}/{video_file}")
        
        if added_count > 0:
            # Sort recordings by timestamp
            for word in self.metadata:
                self.metadata[word] = sorted(self.metadata[word], 
                                           key=lambda x: x.get('timestamp', ''))
            
            # Save updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"\nAdded {added_count} orphaned files to metadata")
        else:
            print("No orphaned files found")

def main():
    print("METADATA CLEANER AND VALIDATOR")
    print("="*60)
    
    cleaner = MetadataCleaner('my_lipreading_dataset')
    
    while True:
        print("\nOptions:")
        print("1. Clean metadata (remove missing files)")
        print("2. Add orphaned files to metadata")
        print("3. Validate and show summary")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            cleaner.clean_metadata()
            cleaner.print_summary()
            
        elif choice == '2':
            cleaner.add_orphaned_files_to_metadata()
            cleaner.print_summary()
            
        elif choice == '3':
            cleaner.validate_and_renumber()
            cleaner.print_summary()
            
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()