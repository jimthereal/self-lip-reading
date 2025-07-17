"""
Improved Model Training with Regularization and Better Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pickle
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Your 30 words
VOCABULARY = [
    "HELLO", "HI", "GOODBYE", "BYE", "THANKS", "PLEASE", "SORRY", "YES", "NO", "OKAY",
    "ONE", "TWO", "THREE", "FOUR", "FIVE",
    "WHAT", "WHERE", "HOW", "GO", "COME", "HELP", "WANT", "NEED",
    "NOW", "TODAY", "TIME", "YOU", "ME", "GOOD", "BAD"
]

class ImprovedLipNet(nn.Module):
    """Improved architecture with better regularization"""
    def __init__(self, num_classes=30):
        super(ImprovedLipNet, self).__init__()
        
        # 3D CNN with dropout
        self.conv1 = nn.Conv3d(1, 32, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.drop1 = nn.Dropout3d(0.2)
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        self.drop2 = nn.Dropout3d(0.2)
        
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(96)
        self.drop3 = nn.Dropout3d(0.3)
        
        # Global pooling instead of LSTM for better generalization
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier with heavy dropout
        self.fc1 = nn.Linear(96, 128)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 3D CNN
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = F.max_pool3d(x, (1, 2, 2))
        
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = F.max_pool3d(x, (1, 2, 2))
        
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        
        return x

class CustomDataset(Dataset):
    """Dataset loader"""
    def __init__(self, data_path, word_to_idx, augment_online=False):
        self.data_path = Path(data_path)
        self.sample_files = sorted(list(self.data_path.glob("*.pkl")))
        self.word_to_idx = word_to_idx
        self.augment_online = augment_online
        print(f"Found {len(self.sample_files)} samples in {data_path}")
    
    def __len__(self):
        return len(self.sample_files)
    
    def online_augment(self, video):
        """Random augmentation during training"""
        if random.random() > 0.5:
            # Random brightness
            factor = random.uniform(0.8, 1.2)
            video = video * factor
        
        if random.random() > 0.5:
            # Random noise
            noise = torch.randn_like(video) * 0.02
            video = video + noise
        
        return torch.clamp(video, 0, 1)
    
    def __getitem__(self, idx):
        with open(self.sample_files[idx], 'rb') as f:
            sample = pickle.load(f)
        
        video = sample['video'].float()
        
        # Online augmentation for training
        if self.augment_online and self.data_path.name == 'train':
            video = self.online_augment(video)
        
        word = sample['word']
        label = self.word_to_idx.get(word, 0)
        
        return {
            'video': video,
            'label': torch.tensor(label, dtype=torch.long),
            'word': word
        }

def create_word_mappings():
    word_to_idx = {word: idx for idx, word in enumerate(VOCABULARY)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

def train_improved_model(use_augmented=True):
    """Train with improved settings"""
    print("IMPROVED TRAINING FOR LIP READING")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Choose dataset
    if use_augmented and Path("augmented_custom_dataset").exists():
        base_path = Path("augmented_custom_dataset")
        print("Using AUGMENTED dataset")
    else:
        base_path = Path("processed_custom_dataset")
        print("Using original dataset")
    
    train_path = base_path / "train"
    val_path = base_path / "val"
    test_path = base_path / "test"
    
    # Word mappings
    word_to_idx, idx_to_word = create_word_mappings()
    
    # Create datasets with online augmentation
    train_dataset = CustomDataset(train_path, word_to_idx, augment_online=True)
    val_dataset = CustomDataset(val_path, word_to_idx, augment_online=False)
    test_dataset = CustomDataset(test_path, word_to_idx, augment_online=False)
    
    # Smaller batch size for better generalization
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Create model
    model = ImprovedLipNet(num_classes=30).to(device)
    
    # Loss with label smoothing for better generalization
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, classes, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            self.classes = classes
        
        def forward(self, pred, target):
            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.classes - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
            return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))
    
    criterion = LabelSmoothingLoss(classes=30, smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training
    best_val_acc = 0
    patience_counter = 0
    
    print("\nStarting improved training...")
    
    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            videos = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
            
            current_acc = 100 * train_correct / train_total
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{current_acc:.2f}%"})
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                videos = batch['video'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'epoch': epoch,
                'vocabulary': VOCABULARY,
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word
            }, 'best_improved_model.pth')
            print(f"  Saved new best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        scheduler.step()
        
        # Early stopping
        if patience_counter >= 10:
            print("Early stopping")
            break
    
    # Test evaluation
    print("\nEvaluating on test set...")
    checkpoint = torch.load('best_improved_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            videos = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    import sys
    
    # Then train
    train_improved_model(use_augmented=True)