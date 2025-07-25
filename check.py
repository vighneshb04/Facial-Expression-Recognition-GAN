import torch
import numpy as np
from GATDataset import FacialGraphDataset
from gat_model import ImprovedGAT  # Make sure this matches your model class name
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Settings ---
BATCH_SIZE = 512
MODEL_PATH = "best_model.pth"
GRAPH_DATA_PATH = "graph_data.pt"
# Fix: Use exactly 6 class names to match your data
CLASS_NAMES = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Load dataset ---
dataset = FacialGraphDataset(GRAPH_DATA_PATH)
val_size = int(0.2 * len(dataset))
val_dataset = torch.utils.data.Subset(dataset, range(len(dataset)-val_size, len(dataset)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Load model (with correct output_dim) ---
model = ImprovedGAT(
    input_dim=5, 
    hidden_dim=128, 
    output_dim=7,  # Keep this as 7 to match saved weights
    heads=6,       # Must match your trained model
    dropout=0.5
).to(device)

# Load weights with weights_only=False for PyTorch 2.6 compatibility
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Evaluation ---
all_preds = []
all_labels = []
with torch.no_grad():
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        # Only consider predictions for the 6 valid classes
        out = out[:, :6]  
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

# --- Metrics ---
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

# Create confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")
