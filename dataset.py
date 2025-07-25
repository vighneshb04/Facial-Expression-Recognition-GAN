import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class CombinedDataset(Dataset):
    """Dataset that loads both facial images and their corresponding landmarks."""
    
    def __init__(self, image_dir, landmark_dir, transform=None):
        """
        Args:
            image_dir: Directory with face images
            landmark_dir: Directory with corresponding landmark files (.npy)
            transform: Optional transform for images
        """
        self.image_dir = image_dir
        self.landmark_dir = landmark_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Get list of files (ensure they match between images and landmarks)
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.landmark_files = sorted([f for f in os.listdir(landmark_dir) if f.endswith('.npy')])
        
        # Match filenames (assuming same base names)
        self.matched_files = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            landmark_file = f"{base_name}.npy"
            if landmark_file in self.landmark_files:
                self.matched_files.append((img_file, landmark_file))
        
    def __len__(self):
        return len(self.matched_files)
    
    def __getitem__(self, idx):
        # Load image
        img_file, landmark_file = self.matched_files[idx]
        image = Image.open(os.path.join(self.image_dir, img_file)).convert('RGB')
        
        # Load landmarks
        landmarks = np.load(os.path.join(self.landmark_dir, landmark_file))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert landmarks to tensor
        landmarks = torch.tensor(landmarks, dtype=torch.float)
        
        # Create the edge_index for graph structure (same as in your GAT model)
        edge_index = self._create_edge_index()
        
        return {
            'image': image, 
            'landmarks': landmarks, 
            'edge_index': edge_index,
            'filename': img_file
        }
    
    def _create_edge_index(self):
        """Create facial landmark graph structure"""
        edges = [
            # Jaw line
            *[(i, i+1) for i in range(0, 16)],
            # Eyebrows
            *[(i, i+1) for i in range(17, 21)], *[(i, i+1) for i in range(22, 26)],
            # Nose bridge and bottom
            *[(i, i+1) for i in range(27, 30)], *[(i, i+1) for i in range(31, 35)], (30, 35),
            # Eyes
            *[(i, i+1) for i in range(36, 41)], (41, 36),
            *[(i, i+1) for i in range(42, 47)], (47, 42),
            # Lips
            *[(i, i+1) for i in range(48, 59)], (59, 48),
            *[(i, i+1) for i in range(60, 67)], (67, 60)
        ]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
