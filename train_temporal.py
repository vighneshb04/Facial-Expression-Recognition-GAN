import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from temporal_model import TemporalConsistencyModel
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

# Settings
BATCH_SIZE = 16
EPOCHS = 10  # Increased from 5 to 10 for better convergence
LR = 0.0001
SEQUENCE_LENGTH = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust']
print(f"Using device: {DEVICE}")

# Temporal dataset
class TemporalDataset(Dataset):
    def __init__(self, landmark_dir, image_dir, seq_length=8):
        self.landmark_dir = landmark_dir
        self.image_dir = image_dir
        self.seq_length = seq_length
        
        # Group files by video
        self.video_frames = {}
        landmark_files = sorted([f for f in os.listdir(landmark_dir) if f.endswith('.npy')])
        
        # Match landmarks to images
        for lm_file in landmark_files:
            parts = os.path.splitext(lm_file)[0].split('_frame_')
            if len(parts) != 2:
                continue
                
            video_name = parts[0]
            frame_num = int(parts[1])
            
            if video_name not in self.video_frames:
                self.video_frames[video_name] = []
                
            img_file = f"{parts[0]}_frame_{parts[1]}.jpg"
            if os.path.exists(os.path.join(image_dir, img_file)):
                self.video_frames[video_name].append((lm_file, img_file, frame_num))
        
        # Create sequences
        self.sequences = []
        for video_name, frames in self.video_frames.items():
            sorted_frames = sorted(frames, key=lambda x: x[2])
            
            # Use sliding window to create sequences
            for i in range(len(sorted_frames) - seq_length + 1):
                self.sequences.append(sorted_frames[i:i+seq_length])
                
        print(f"Created {len(self.sequences)} sequences from {len(self.video_frames)} videos")
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Load landmarks and images
        landmarks_seq = []
        images_seq = []
        
        for lm_file, img_file, _ in sequence:
            # Load landmark
            landmark = np.load(os.path.join(self.landmark_dir, lm_file))
            
            # Normalize landmarks per frame (IMPORTANT FOR ACCURACY)
            landmark = (landmark - np.mean(landmark, axis=0)) / np.std(landmark)
            
            landmarks_seq.append(torch.tensor(landmark, dtype=torch.float))
            
            # Load image
            image = Image.open(os.path.join(self.image_dir, img_file)).convert('RGB')
            image = self.transform(image)
            images_seq.append(image)
            
        # Stack into tensors
        landmarks_tensor = torch.stack(landmarks_seq)
        images_tensor = torch.stack(images_seq)
        
        # Get label from filename
        video_name = sequence[0][0].split('_frame_')[0]
        label = self._get_label_from_filename(video_name)
        labels = torch.tensor([label] * self.seq_length, dtype=torch.long)
        
        return {
            'landmark_seq': landmarks_tensor,
            'image_seq': images_tensor,
            'label_seq': labels,
            'video_name': video_name
        }
        
    def _get_label_from_filename(self, filename):
        # Extract emotion label based on your naming convention
        emotion_map = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'SUR': 3, 'FEA': 4, 'DIS': 5}
        
        for emotion, idx in emotion_map.items():
            if emotion in filename:
                return idx
                
        # Default to neutral
        return 0

def main():
    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = TemporalDataset(
        landmark_dir="output_landmarks",
        image_dir="output_images",
        seq_length=SEQUENCE_LENGTH
    )

    # Use smaller subset for faster training
    subset_size = min(4000, len(dataset))  # Increased from 2000 to 4000
    indices = torch.randperm(len(dataset))[:subset_size]
    dataset = torch.utils.data.Subset(dataset, indices)
    print(f"Using subset of {len(dataset)} sequences for speed")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # Initialize model
    model = TemporalConsistencyModel(
        fusion_model_path="results/best_fusion_model.pth", 
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)

    # Optimizer and scaler for mixed precision
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    scaler = GradScaler()

    # Class weights to address class imbalance
    class_weights = torch.tensor([1.0, 1.2, 1.0, 1.2, 1.5, 1.2], device=DEVICE)  # Higher weight for underrepresented classes
    weighted_loss = torch.nn.NLLLoss(weight=class_weights)

    # Training loop
    best_acc = 0
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            image_seq = batch['image_seq'].to(DEVICE)
            landmark_seq = batch['landmark_seq'].to(DEVICE)
            label_seq = batch['label_seq'].to(DEVICE)
            
            # Mixed precision forward pass
            optimizer.zero_grad()
            with autocast():
                outputs = model(image_seq, landmark_seq)
                # FIXED: Use only the first label from each sequence
                loss = weighted_loss(outputs, label_seq[:, 0])
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                image_seq = batch['image_seq'].to(DEVICE)
                landmark_seq = batch['landmark_seq'].to(DEVICE)
                label_seq = batch['label_seq'].to(DEVICE)
                
                outputs = model(image_seq, landmark_seq)
                # FIXED: Use only first label from each sequence
                loss = F.nll_loss(outputs, label_seq[:, 0])
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_seq[:, 0].cpu().numpy())
        
        # Calculate metrics
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "results/best_temporal_model.pth")
            print(f"New best model saved! Accuracy: {val_acc:.4f}")
            
            # Create confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Temporal Model Confusion Matrix (Acc: {val_acc:.4f})')
            plt.tight_layout()
            plt.savefig("results/temporal_confusion_matrix.png")

    print("Training complete!")

if __name__ == "__main__":
    main()
