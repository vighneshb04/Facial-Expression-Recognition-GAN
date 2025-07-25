import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from GATDataset import FacialGraphDataset
from gat_model import ImprovedGAT
from tqdm import tqdm
import time
import os
from sklearn.metrics import confusion_matrix

# Configuration
BATCH_SIZE = 512  # Reduced to prevent overfitting
EPOCHS = 100
LEARNING_RATE = 0.0003  # Lower learning rate
WEIGHT_DECAY = 0.01  # L2 regularization
PATIENCE = 15
CLASS_NAMES = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

# GPU setup
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    # Load dataset
    dataset = FacialGraphDataset("graph_data.pt")
    
    # Stratified sampling for train/val split
    class_indices = {}
    for i, data in enumerate(dataset):
        label = data.y.item()
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    # Create balanced train/val split
    train_indices = []
    val_indices = []
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE*2,
        num_workers=0,
        pin_memory=True
    )

    # Initialize model
    model = ImprovedGAT(
        input_dim=5,
        hidden_dim=128,
        output_dim=len(CLASS_NAMES),
        heads=6,
        dropout=0.5
    ).to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    scaler = torch.amp.GradScaler()
    best_val_acc = 0
    patience_counter = 0

    # Training loop
    for epoch in range(1, EPOCHS+1):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Training phase
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                out = model(data.x, data.edge_index, data.batch)
                loss = F.nll_loss(out, data.y)
            
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                val_preds.extend(out.argmax(dim=1).cpu().numpy())
                val_labels.extend(data.y.cpu().numpy())
        
        # Calculate metrics
        train_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_acc = (np.array(val_preds) == np.array(val_labels)).mean()
        epoch_time = time.time() - epoch_start
        
        # Log results
        print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✓ New best model saved! Val Acc: {val_acc:.4f}")
            
            # Print per-class accuracy
            cm = confusion_matrix(val_labels, val_preds)
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class_acc)):
                print(f"  - {name}: {acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == "__main__":
    main()
