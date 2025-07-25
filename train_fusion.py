import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from fusion_model import FusionModel
from multimodal_dataset import MultimodalDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os
from torch.amp import autocast, GradScaler

# Settings
BATCH_SIZE = 512
EPOCHS = 3
LR = 0.001
LANDMARK_DIR = "output_landmarks"
IMAGE_DIR = "output_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust']

def main():
    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Load dataset
    dataset = MultimodalDataset(LANDMARK_DIR, IMAGE_DIR)
    print(f"Full dataset size: {len(dataset)}")

    # Use small subset for fast training
    subset_size = min(10000, len(dataset))
    indices = torch.randperm(len(dataset))[:subset_size]
    dataset = Subset(dataset, indices)
    print(f"Training on subset of {len(dataset)} samples for speed")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Optimize dataloaders for speed - use 0 workers to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=0,
        pin_memory=True
    )

    # Initialize model, optimizer and mixed precision
    model = FusionModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    scaler = GradScaler()

    # Training loop
    best_acc = 0
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            images = batch['image'].to(DEVICE)
            landmarks = batch['landmarks'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Mixed precision forward pass - FIX: Added device_type parameter
            optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images, landmarks)
                loss = F.nll_loss(outputs, labels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(DEVICE)
                landmarks = batch['landmarks'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                outputs = model(images, landmarks)
                val_loss += F.nll_loss(outputs, labels).item()
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
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
            torch.save(model.state_dict(), "results/best_fusion_model.pth")
            print(f"New best model saved! Accuracy: {val_acc:.4f}")
            
            # Create confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Fusion Model Confusion Matrix (Acc: {val_acc:.4f})')
            plt.tight_layout()
            plt.savefig("results/fusion_confusion_matrix.png")
    
    print("Training complete!")

# For proper multiprocessing
if __name__ == "__main__":
    main()
