from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import joblib
import os
import json
from models.gnn_model import build_gnn

app = Flask(__name__)

# Load models and scaler
scaler = joblib.load('models/scaler.joblib')
rf_model = joblib.load('models/rf_model.joblib')

gnn_state_dict = torch.load('models/gnn_model.pth')
gnn_model = build_gnn(8) # 8 features
gnn_model.load_state_dict(gnn_state_dict)
gnn_model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['Amount']),
            float(data['Sender_Balance']),
            float(data['Receiver_Balance']),
            float(data['Tx_Freq']),
            float(data['Hop_Count']),
            float(data['Time_Delta']),
            float(data['Is_Mixing']),
            float(data['Network_Centrality'])
        ]
        
        # Scale
        features_scaled = scaler.transform([features])
        features_tensor = torch.tensor(features_scaled, dtype=torch.float)
        
        # RF Prediction
        rf_pred = rf_model.predict(features_scaled)[0]
        
        # GNN Prediction
        # For a single prediction, we treat it as a single node with a self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        gnn_data = Data(x=features_tensor, edge_index=edge_index)
        with torch.no_grad():
            gnn_out = gnn_model(gnn_data)
            gnn_pred = gnn_out.argmax(dim=1).item()
            
        return jsonify({
            'rf_result': 'Laundering (Bad)' if rf_pred == 1 else 'Normal (Good)',
            'gnn_result': 'Laundering (Bad)' if gnn_pred == 1 else 'Normal (Good)',
            'rf_raw': int(rf_pred),
            'gnn_raw': int(gnn_pred)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    with open('static/data/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    df = pd.read_csv('data/transactions.csv')
    dataset_details = {
        'total': len(df),
        'bad': int(df['Label'].sum()),
        'good': int(len(df) - df['Label'].sum()),
        'features': 8
    }
    
    return render_template('dashboard.html', metrics=metrics, details=dataset_details)

if __name__ == '__main__':
    app.run(debug=True)
