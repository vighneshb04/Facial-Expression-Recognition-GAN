import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FusionModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # CNN branch
        self.cnn = models.resnet18(weights='DEFAULT')
        self.cnn.fc = nn.Linear(512, 128)
        
        # MLP for landmarks
        self.landmark_mlp = nn.Sequential(
            nn.Linear(68 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, images, landmarks, edge_index=None):
        # Process images
        cnn_features = self.cnn(images)
        
        # Process landmarks
        batch_size = landmarks.size(0)
        landmarks_flat = landmarks.view(batch_size, -1)
        landmark_features = self.landmark_mlp(landmarks_flat)
        
        # Combine features
        combined = torch.cat([landmark_features, cnn_features], dim=1)
        
        # Classification
        logits = self.fusion(combined)
        return F.log_softmax(logits, dim=1)
