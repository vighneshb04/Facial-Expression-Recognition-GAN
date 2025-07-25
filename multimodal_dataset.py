import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms

class MultimodalDataset(Dataset):
    def __init__(self, landmark_dir, image_dir, transform=None):
        self.landmark_dir = landmark_dir
        self.image_dir = image_dir
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Get files
        self.landmark_files = [f for f in os.listdir(landmark_dir) if f.endswith('.npy')]
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Found {len(self.landmark_files)} landmark files in {landmark_dir}")
        print(f"Found {len(self.image_files)} image files in {image_dir}")
        
        # Match files
        self.pairs = []
        for lm_file in self.landmark_files:
            base_name = os.path.splitext(lm_file)[0]
            for ext in ['.jpg', '.png', '.jpeg']:
                img_file = base_name + ext
                if img_file in self.image_files:
                    self.pairs.append((lm_file, img_file))
                    break
        
        print(f"Found {len(self.pairs)} valid image-landmark pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        lm_file, img_file = self.pairs[idx]
        
        # Load landmark
        landmarks = np.load(os.path.join(self.landmark_dir, lm_file))
        landmarks = torch.tensor(landmarks, dtype=torch.float)
        
        # Load image
        image = Image.open(os.path.join(self.image_dir, img_file)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Extract label from filename
        label_map = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'SUR': 3, 'FEA': 4, 'DIS': 5}
        label = 0  # Default to neutral
        
        for key, val in label_map.items():
            if key in lm_file:
                label = val
                break
        
        # Create edge index for graph
        edge_index = self._create_edge_index()
        
        return {
            'landmarks': landmarks,
            'image': image,
            'edge_index': edge_index,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _create_edge_index(self):
        edges = [
            *[(i, i+1) for i in range(0, 16)],
            *[(i, i+1) for i in range(17, 21)], *[(i, i+1) for i in range(22, 26)],
            *[(i, i+1) for i in range(27, 30)], *[(i, i+1) for i in range(31, 35)], (30, 35),
            *[(i, i+1) for i in range(36, 41)], (41, 36),
            *[(i, i+1) for i in range(42, 47)], (47, 42),
            *[(i, i+1) for i in range(48, 59)], (59, 48),
            *[(i, i+1) for i in range(60, 67)], (67, 60)
        ]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
