import cv2
import os
import time
from datetime import datetime
import json
import numpy as np
import random
import sys
import urllib.request

# Optimized 30-word list
FINAL_30_WORDS = [
    # Already recorded (15 words)
    "HELLO", "HI", "GOODBYE", "BYE", "THANKS", "PLEASE", "SORRY", "YES", "NO", "OKAY",
    "ONE", "TWO", "THREE", "FOUR", "FIVE",
    
    # Additional essential words (15 words)
    "WHAT", "WHERE", "HOW",  # Questions
    "GO", "COME", "HELP", "WANT", "NEED",  # Verbs
    "NOW", "TODAY", "TIME",  # Time
    "YOU", "ME",  # People
    "GOOD", "BAD"  # Adjectives
]

def get_face_cascade():
    """Get face cascade classifier with multiple fallback options"""
    
    # Method 1: Try cv2.data.haarcascades
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            return cv2.CascadeClassifier(cascade_path)
    except AttributeError:
        pass
    
    # Method 2: Common installation paths
    possible_paths = [
        # Windows conda paths
        os.path.join(sys.prefix, 'Lib', 'site-packages', 'cv2', 'data', 'haarcascade_frontalface_default.xml'),
        os.path.join(sys.prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml'),
        # Direct cv2 package path
        os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return cv2.CascadeClassifier(path)
    
    # Method 3: Download the cascade file if not found
    cascade_file = 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_file):
        return cv2.CascadeClassifier(cascade_file)
    
    print("Face cascade not found. Downloading...")
    cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    
    try:
        urllib.request.urlretrieve(cascade_url, cascade_file)
        print("Face cascade downloaded successfully!")
        return cv2.CascadeClassifier(cascade_file)
    except Exception as e:
        print(f"Warning: Could not load face cascade: {e}")
        print("Face detection will be disabled.")
        return None

class SimpleLipReadingRecorder:
    def __init__(self, output_dir='my_lipreading_dataset'):
        self.output_dir = output_dir
        self.metadata = {}
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing metadata if it exists
        self.metadata_file = os.path.join(output_dir, 'metadata.json')
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Face detection for better framing
        self.face_cascade = get_face_cascade()
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_word_count(self, word):
        """Get how many times a word has been recorded"""
        return len(self.metadata.get(word, []))
    
    def detect_face(self, frame):
        """Detect face in frame and return bounding box"""
        if self.face_cascade is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            return faces[0]  # Return first face
        return None
    
    def record_word(self, word, duration=1.5, countdown=3):
        """Record a single word"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Video settings
        fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create word directory
        word_dir = os.path.join(self.output_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        
        # Generate filename
        word_count = self.get_word_count(word)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{word}_{word_count:03d}_{timestamp}.mp4"
        filepath = os.path.join(word_dir, filename)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        # Preview phase
        print(f"\nPreparing to record '{word}' (#{word_count + 1})")
        print("Please position your face in the center of the frame")
        
        # Show preview for positioning with face detection
        face_detected = False
        start_preview = time.time()
        while time.time() - start_preview < 3:  # 3 seconds to position face
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
                
                # Try face detection if available
                if self.face_cascade is not None:
                    face = self.detect_face(frame)
                    
                    if face is not None:
                        x, y, w, h = face
                        # Draw face rectangle
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Check if face is centered
                        face_center_x = x + w//2
                        frame_center_x = width//2
                        if abs(face_center_x - frame_center_x) < 50:
                            cv2.putText(display_frame, "Face detected - Good position!", (50, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            face_detected = True
                        else:
                            cv2.putText(display_frame, "Please center your face", (50, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        cv2.putText(display_frame, "No face detected", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Fallback if face detection not available
                    cv2.line(display_frame, (width//2, 0), (width//2, height), (0, 255, 0), 1)
                    cv2.line(display_frame, (0, height//2), (width, height//2), (0, 255, 0), 1)
                    cv2.putText(display_frame, "Center your face", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Position Check', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return None
        
        if self.face_cascade is not None and not face_detected:
            print("Tip: Try to center your face for better results")
        
        # Countdown
        for i in range(countdown, 0, -1):
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
                cv2.putText(display_frame, str(i), (width//2-50, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
                cv2.putText(display_frame, f"Say: {word}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.imshow('Get Ready', display_frame)
                cv2.waitKey(1)
            time.sleep(1)
        
        # Recording
        print("RECORDING! Say the word clearly!")
        start_time = time.time()
        frames_recorded = 0
        
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                
                # Show recording indicator with face detection
                display_frame = frame.copy()
                cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display_frame, f"RECORDING: {word}", (60, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                remaining = duration - (time.time() - start_time)
                cv2.putText(display_frame, f"Time: {remaining:.1f}s", (60, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show face detection during recording if available
                if self.face_cascade is not None:
                    face = self.detect_face(frame)
                    if face is not None:
                        x, y, w, h = face
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.imshow('Recording', display_frame)
                frames_recorded += 1
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Update metadata
        if word not in self.metadata:
            self.metadata[word] = []
        
        self.metadata[word].append({
            'filename': filename,
            'filepath': filepath,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'frames': frames_recorded,
            'fps': fps
        })
        
        self.save_metadata()
        print(f"✓ Saved: {filename}")
        return filepath
    
    def smart_recording_session(self, target_recordings=20):
        """Smart recording that prioritizes words with fewer recordings"""
        print("\n" + "="*60)
        print("SMART RECORDING SESSION")
        print("="*60)
        print(f"Target: {target_recordings} recordings per word")
        print(f"Total words: {len(FINAL_30_WORDS)}")
        
        # Calculate what needs to be recorded
        needed_recordings = []
        for word in FINAL_30_WORDS:
            current_count = self.get_word_count(word)
            needed = max(0, target_recordings - current_count)
            for _ in range(needed):
                needed_recordings.append(word)
        
        if not needed_recordings:
            print("\nAll words have reached the target number of recordings!")
            return
        
        # Shuffle for variety
        random.shuffle(needed_recordings)
        
        print(f"Recordings needed: {len(needed_recordings)}")
        print(f"Estimated time: {len(needed_recordings) * 5 / 60:.1f} minutes")
        print("\nPress Enter to start, or 'q' to quit")
        
        if input().strip().lower() == 'q':
            return
        
        # Recording loop
        completed = 0
        total = len(needed_recordings)
        
        for i, word in enumerate(needed_recordings):
            # Show progress
            print(f"\n[{i+1}/{total}] Progress: {100*completed/total:.1f}%")
            
            # Show current status
            print("\nCurrent word counts:")
            words_status = []
            for w in sorted(set(FINAL_30_WORDS)):
                count = self.get_word_count(w)
                status = "✓" if count >= target_recordings else f"{count}/{target_recordings}"
                words_status.append(f"{w}:{status}")
            
            # Print in columns
            for j in range(0, len(words_status), 5):
                print("  " + "  ".join(words_status[j:j+5]))
            
            print(f"\nNext: '{word}'")
            print("ENTER=record, 's'=skip, 'p'=pause, 'q'=quit: ", end='')
            
            user_input = input().strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == 's':
                continue
            elif user_input == 'p':
                input("\nPAUSED. Press Enter to continue...")
                continue
            
            # Record
            if self.record_word(word) is None:
                break
            
            completed += 1
            time.sleep(0.5)
        
        print(f"\nSession complete! Recorded {completed} videos")
        self.print_statistics()
    
    def print_statistics(self):
        """Print recording statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        total_recordings = sum(len(recordings) for recordings in self.metadata.values())
        print(f"Total words: {len(self.metadata)}/{len(FINAL_30_WORDS)}")
        print(f"Total recordings: {total_recordings}")
        
        if self.metadata:
            print(f"Average recordings per word: {total_recordings/len(self.metadata):.1f}")
        
        print("\nDetailed breakdown:")
        print("-" * 40)
        
        # Show status for all 30 words
        for word in sorted(FINAL_30_WORDS):
            count = self.get_word_count(word)
            status = "✓" if count >= 20 else f"Need {20-count} more"
            print(f"{word:12} : {count:3} recordings  [{status}]")
        
        # Summary
        words_complete = sum(1 for word in FINAL_30_WORDS if self.get_word_count(word) >= 20)
        print("-" * 40)
        print(f"Words complete (20+ recordings): {words_complete}/{len(FINAL_30_WORDS)}")
        print("="*60)

def main():
    recorder = SimpleLipReadingRecorder('my_lipreading_dataset')
    
    while True:
        print("\n" + "="*60)
        print("30-WORD LIP READING DATASET RECORDER")
        print("="*60)
        print("1. Smart recording (fills missing recordings)")
        print("2. Record specific word")
        print("3. View statistics")
        print("4. Export progress report")
        print("5. Exit")
        print("="*60)
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            target = input("Target recordings per word (default=20): ").strip()
            target = int(target) if target else 20
            recorder.smart_recording_session(target)
            
        elif choice == '2':
            print("\nAvailable words:")
            for i, word in enumerate(FINAL_30_WORDS, 1):
                count = recorder.get_word_count(word)
                print(f"{i:2}. {word} ({count} recordings)")
            
            word_input = input("\nEnter word or number: ").strip().upper()
            if word_input.isdigit():
                idx = int(word_input) - 1
                if 0 <= idx < len(FINAL_30_WORDS):
                    word = FINAL_30_WORDS[idx]
                    recorder.record_word(word)
            elif word_input in FINAL_30_WORDS:
                recorder.record_word(word_input)
            else:
                print("Invalid word!")
            
        elif choice == '3':
            recorder.print_statistics()
            
        elif choice == '4':
            # Export progress report
            report_file = 'recording_progress.txt'
            with open(report_file, 'w') as f:
                f.write("LIP READING DATASET PROGRESS REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                total = sum(len(recordings) for recordings in recorder.metadata.values())
                f.write(f"Total recordings: {total}\n")
                f.write(f"Words recorded: {len(recorder.metadata)}/{len(FINAL_30_WORDS)}\n\n")
                
                f.write("Detailed breakdown:\n")
                f.write("-"*40 + "\n")
                for word in sorted(FINAL_30_WORDS):
                    count = recorder.get_word_count(word)
                    f.write(f"{word:12} : {count:3} recordings\n")
            
            print(f"Progress report saved to {report_file}")
            
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()