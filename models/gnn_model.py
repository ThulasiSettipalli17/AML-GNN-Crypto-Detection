import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class GNN_Model(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN_Model, self).__init__()
        # Using a simple Graph Convolutional Network (GCN)
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc = nn.Linear(8, 2) # Binary classification: Good (0) or Bad (1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling or just node level?
        # User wants transaction identification, so node-level prediction
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

def build_gnn(num_features):
    return GNN_Model(num_features)
