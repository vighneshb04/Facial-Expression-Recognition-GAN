import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_model import FusionModel
import os

class TemporalConsistencyModel(nn.Module):
    def __init__(self, fusion_model_path=None, num_classes=6):
        super().__init__()
        # For training (using fusion model)
        if fusion_model_path:
            self.fusion_model = FusionModel()
            if os.path.exists(fusion_model_path):
                print(f"Loading fusion model from {fusion_model_path}")
                self.fusion_model.load_state_dict(torch.load(fusion_model_path))
            for param in self.fusion_model.parameters():
                param.requires_grad = False
        
        # CRITICAL: These dimensions MUST match what's in your checkpoint
        self.gru = nn.GRU(
            input_size=num_classes,
            hidden_size=128,      # Keep at 128 to match checkpoint
            num_layers=2,
            bidirectional=True,   # Must be True to match checkpoint
            batch_first=True
        )
        self.fc = nn.Linear(256, num_classes)  # 256 = 128*2 for bidirectional
        
    def forward(self, x, landmarks=None):
        # Training mode: process raw inputs through fusion
        if landmarks is not None and hasattr(self, 'fusion_model'):
            batch_size, seq_len = x.size(0), x.size(1)
            frame_preds = []
            
            for t in range(seq_len):
                with torch.no_grad():
                    pred_t = self.fusion_model(x[:, t], landmarks[:, t])
                    frame_preds.append(pred_t.unsqueeze(1))
            
            x = torch.cat(frame_preds, dim=1)
        
        # Process sequence through GRU
        x, _ = self.gru(x)
        
        # Take final timestep for classification
        x = self.fc(x[:, -1, :])  # Use last timestep output
        
        return F.log_softmax(x, dim=1)
