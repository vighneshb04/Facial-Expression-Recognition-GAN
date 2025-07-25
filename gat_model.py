import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.aggr import AttentionalAggregation

class ImprovedGAT(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=7, heads=6, dropout=0.5):
        super().__init__()
        # Input embedding with batch norm
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        # First GAT layer with residual connection
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim*heads)
        
        # Second GAT layer
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim, heads=4, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim*4)
        
        # Alternative pathway with GCN (creates ensemble effect)
        self.gcn = GCNConv(hidden_dim, hidden_dim*4)
        self.bn_gcn = nn.BatchNorm1d(hidden_dim*4)
        
        # Gating mechanism to combine GAT and GCN paths
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim*4*2, hidden_dim*4),
            nn.Sigmoid()
        )
        
        # Attention pooling
        self.pool = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(hidden_dim*4, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        # Classifier with L2 regularization on last layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Initial embedding
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = F.leaky_relu(x)
        
        # GAT pathway
        x_gat1 = self.gat1(x, edge_index)
        x_gat1 = self.bn1(x_gat1)
        x_gat1 = F.leaky_relu(x_gat1)
        x_gat1 = F.dropout(x_gat1, p=0.2, training=self.training)
        
        x_gat2 = self.gat2(x_gat1, edge_index)
        x_gat2 = self.bn2(x_gat2)
        x_gat2 = F.leaky_relu(x_gat2)
        
        # GCN pathway for ensemble effect
        x_gcn = self.gcn(x, edge_index)
        x_gcn = self.bn_gcn(x_gcn)
        x_gcn = F.leaky_relu(x_gcn)
        
        # Combine pathways with gate
        combined = torch.cat([x_gat2, x_gcn], dim=1)
        gate_weights = self.gate(combined)
        x_combined = gate_weights * x_gat2 + (1 - gate_weights) * x_gcn
        
        # Graph-level pooling
        x_pooled = self.pool(x_combined, batch)
        
        # Classification
        logits = self.classifier(x_pooled)
        return F.log_softmax(logits, dim=1)
