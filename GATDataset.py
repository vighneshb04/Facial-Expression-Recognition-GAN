import torch
from torch_geometric.data import Data, Dataset
import numpy as np

class FacialGraphDataset(Dataset):
    def __init__(self, graph_pt_path):
        super().__init__()
        print(f"Loading dataset from {graph_pt_path}")
        # Use weights_only=False for PyTorch 2.6+ compatibility
        self.graphs = torch.load(graph_pt_path, weights_only=False)
        assert isinstance(self.graphs, list), "Expected list of Data objects"
        
        # Filter out invalid samples
        self.graphs = [g for g in self.graphs if (g is not None and hasattr(g, 'y') and g.y is not None)]
        
        # Add data normalization here
        self._normalize_features()
        
        print(f"Successfully loaded {len(self.graphs)} graph samples")
        
        # Calculate class distribution
        self.labels = [g.y.item() for g in self.graphs]
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples")
    
    def _normalize_features(self):
        """Normalize node features across the dataset"""
        # Collect all features
        all_features = torch.cat([g.x for g in self.graphs], dim=0)
        
        # Calculate mean and std
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        std[std < 1e-5] = 1.0  # Prevent division by zero
        
        # Apply normalization
        for i in range(len(self.graphs)):
            self.graphs[i].x = (self.graphs[i].x - mean) / std

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)
