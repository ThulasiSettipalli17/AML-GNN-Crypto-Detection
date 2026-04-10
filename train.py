import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

from models.rf_model import train_rf
from models.gnn_model import build_gnn

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('static/data', exist_ok=True)

def prepare_data():
    df = pd.read_csv('data/transactions.csv')
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X_scaled, y

def train_gnn_model(X_scaled, y):
    # For GNN, we need a graph. 
    # In a real scenario, wallets are nodes and transactions are edges.
    # For this synthetic dataset, we'll create a "similarity" graph or "random-link" graph 
    # to demonstrate GNN capabilities on tabular data features as node attributes.
    
    num_nodes = len(X_scaled)
    x = torch.tensor(X_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    
    # Create random edges for demonstration (in real app, use sender/receiver IDs)
    # We'll link each node to 2 random other nodes to create a graph structure
    edge_index_1 = torch.randint(0, num_nodes, (1, num_nodes * 2))
    edge_index_2 = torch.randint(0, num_nodes, (1, num_nodes * 2))
    edge_index = torch.cat([edge_index_1, edge_index_2], dim=0)
    
    data = Data(x=x, edge_index=edge_index, y=y_tensor)
    
    # Split for GNN
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_indices = np.random.choice(num_nodes, int(num_nodes * 0.8), replace=False)
    train_mask[train_indices] = True
    test_mask[~train_mask] = True
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y[train_mask.numpy()])
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    
    model = build_gnn(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(300): # Increased epochs
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.nll_loss(out[train_mask], data.y[train_mask], weight=weights)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'GNN Epoch {epoch}, Loss: {loss.item():.4f}')
            
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    
    correct = (pred[test_mask] == data.y[test_mask]).sum().item()
    acc = correct / test_mask.sum().item()
    
    from sklearn.metrics import f1_score, recall_score
    f1 = f1_score(data.y[test_mask].numpy(), pred[test_mask].numpy())
    rec = recall_score(data.y[test_mask].numpy(), pred[test_mask].numpy())
    
    print(f"GNN - Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {rec:.4f}")
    
    torch.save(model.state_dict(), 'models/gnn_model.pth')
    return acc, f1, rec

def save_metrics(rf_metrics, gnn_metrics):
    metrics = {
        'rf': {
            'accuracy': rf_metrics[0],
            'f1': rf_metrics[1],
            'recall': rf_metrics[2]
        },
        'gnn': {
            'accuracy': gnn_metrics[0],
            'f1': gnn_metrics[1],
            'recall': gnn_metrics[2]
        }
    }
    with open('static/data/metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_scaled, y_all = prepare_data()
    
    print("Training Random Forest...")
    rf_m = train_rf(X_train, y_train, X_test, y_test)
    
    print("Training GNN...")
    gnn_m = train_gnn_model(X_scaled, y_all)
    
    save_metrics(rf_m, gnn_m)
    print("All models trained and metrics saved.")
