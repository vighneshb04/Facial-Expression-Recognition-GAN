import numpy as np
import torch
from torch_geometric.data import Data

def create_graphs(npz_path, save_path):
    data = np.load(npz_path)
    landmarks = data['X'].reshape(-1, 68, 2).astype(np.float32)
    labels = data['y']
    edge_index = torch.tensor([
        *[(i, i+1) for i in range(0, 16)],
        *[(i, i+1) for i in range(17, 21)], *[(i, i+1) for i in range(22, 26)],
        *[(i, i+1) for i in range(27, 30)], *[(i, i+1) for i in range(31, 35)], (30, 35),
        *[(i, i+1) for i in range(36, 41)], (41, 36),
        *[(i, i+1) for i in range(42, 47)], (47, 42),
        *[(i, i+1) for i in range(48, 59)], (59, 48),
        *[(i, i+1) for i in range(60, 67)], (67, 60)
    ], dtype=torch.long).t().contiguous()

    graphs = []
    for i in range(len(landmarks)):
        lm = landmarks[i]
        label = labels[i]
        center = lm[30]
        lm = lm - center
        max_dist = np.max(np.linalg.norm(lm, axis=1)) + 1e-6
        lm = lm / max_dist
        left_eye_center = lm[36:42].mean(axis=0)
        right_eye_center = lm[42:48].mean(axis=0)
        mouth_center = lm[48:68].mean(axis=0)
        distances = np.zeros((68, 3), dtype=np.float32)
        for j in range(68):
            distances[j, 0] = np.linalg.norm(lm[j] - left_eye_center)
            distances[j, 1] = np.linalg.norm(lm[j] - right_eye_center)
            distances[j, 2] = np.linalg.norm(lm[j] - mouth_center)
        features = np.hstack([lm, distances])
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    torch.save(graphs, save_path)
    print(f"Saved {len(graphs)} graphs to {save_path}")

if __name__ == "__main__":
    create_graphs("emotion_dataset.npz", "graph_data.pt")
