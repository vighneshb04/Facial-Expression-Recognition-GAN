import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import time
import os

def create_graphs(npz_path, save_path):
    print(f"Started processing at {time.strftime('%H:%M:%S')}")
    data = np.load(npz_path)
    landmarks = data['X'].reshape(-1, 68, 2).astype(np.float32)
    labels = data['y']
    print(f"Found {len(landmarks)} samples and {len(np.unique(labels))} classes")
    
    # Create facial graph structure with more connections
    edge_index = torch.tensor([
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
        *[(i, i+1) for i in range(60, 67)], (67, 60),
        # Extra connections for better information flow
        (27, 17), (27, 22),  # Nose to eyebrows
        (30, 51), (30, 33),  # Nose to lips
        (36, 17), (45, 26),  # Eyes to eyebrows
        (48, 3), (54, 13)    # Lips to jaw
    ], dtype=torch.long).t().contiguous()

    # Process each face
    graphs = []
    for i in tqdm(range(len(landmarks)), desc="Processing landmarks"):
        # Skip samples without labels
        if i >= len(labels) or labels[i] is None:
            continue
            
        # Create 3 augmented versions for each sample
        for aug_idx in range(3):  # Original + 2 augmentations
            lm = landmarks[i].copy()
            label = labels[i]
            
            # Apply augmentation (except for the original)
            if aug_idx > 0:
                # Random rotation (-20 to 20 degrees)
                angle = np.random.uniform(-20, 20) * np.pi / 180
                rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                      [np.sin(angle), np.cos(angle)]])
                lm = lm @ rot_matrix
                
                # Random scaling (0.8 to 1.2)
                scale = np.random.uniform(0.8, 1.2)
                lm = lm * scale
                
                # Small random noise
                lm += np.random.normal(0, 0.01, lm.shape)
            
            # Normalize landmarks
            center = lm[30].copy()  # Nose tip as center
            lm = lm - center
            max_dist = np.max(np.linalg.norm(lm, axis=1)) + 1e-6
            lm = lm / max_dist
            
            # Enhanced feature extraction
            # Calculate facial features - distances between key points
            left_eye_center = lm[36:42].mean(axis=0)
            right_eye_center = lm[42:48].mean(axis=0)
            mouth_center = lm[48:68].mean(axis=0)
            
            # Angular features - captures facial expressions better
            features = np.zeros((68, 5), dtype=np.float32)
            for j in range(68):
                # Original coordinates
                features[j, 0:2] = lm[j]
                
                # Distances to facial centers
                features[j, 2] = np.linalg.norm(lm[j] - left_eye_center)
                features[j, 3] = np.linalg.norm(lm[j] - right_eye_center)
                features[j, 4] = np.linalg.norm(lm[j] - mouth_center)
                
            # Create graph data
            x = torch.tensor(features, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    # Save processed data
    torch.save(graphs, save_path)
    print(f"✅ Finished at {time.strftime('%H:%M:%S')}")
    print(f"✅ Created {len(graphs)} samples in graph_data.pt ({os.path.getsize(save_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    create_graphs("emotion_dataset.npz", "graph_data.pt")
