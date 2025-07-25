import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ----- Configure Your Folder -----
DATA_DIR = "output_landmarks"  # Replace with the folder path with .npy files
SAVE_PATH = "graph_data.pt"      # This is where we'll save all the graphs

# ----- Landmark Edges -----
edges = [
    *[(i, i + 1) for i in range(0, 16)],
    *[(i, i + 1) for i in range(17, 21)],
    *[(i, i + 1) for i in range(22, 26)],
    *[(i, i + 1) for i in range(27, 30)],
    *[(i, i + 1) for i in range(31, 35)],
    (30, 35),
    *[(i, i + 1) for i in range(36, 41)], (41, 36),
    *[(i, i + 1) for i in range(42, 47)], (47, 42),
    *[(i, i + 1) for i in range(48, 59)], (59, 48),
    *[(i, i + 1) for i in range(60, 67)], (67, 60),
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# ----- Build Graphs -----
graph_data_list = []

all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npy')])
print(f"Total files: {len(all_files)}")

for file in tqdm(all_files):
    try:
        file_path = os.path.join(DATA_DIR, file)
        landmarks = np.load(file_path)

        # Safety check
        if landmarks.shape != (68, 2):
            print(f"Skipping {file}, shape mismatch: {landmarks.shape}")
            continue

        x = torch.tensor(landmarks, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        data.name = file  # Add filename for tracking

        graph_data_list.append(data)

    except Exception as e:
        print(f"Error in {file}: {e}")

# ----- Save All Graphs -----
torch.save(graph_data_list, SAVE_PATH)
print(f"✅ Saved {len(graph_data_list)} graphs to {SAVE_PATH}")
