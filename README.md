# 🔍 AML Detection using Graph Neural Networks (GNN)

## 📌 Project Overview
This project focuses on detecting illicit cryptocurrency transactions using Graph Neural Networks (GNNs). Due to the decentralized and pseudonymous nature of blockchain systems, traditional AML techniques struggle to identify suspicious activities. This system leverages graph-based learning to model transaction relationships and detect fraudulent patterns effectively.

## 🚀 Key Features
- 📊 Transaction graph construction (nodes = wallets, edges = transactions)
- 🧠 Multiple GNN models: GCN, GAT, GraphSAGE
- ⚠️ Risk scoring for AML compliance
- 🔍 Classification: Licit vs Illicit transactions
- 📈 Visualization of transaction insights

## 🛠️ Tech Stack
- **Programming:** Python
- **Frameworks:** PyTorch / TensorFlow
- **Libraries:** NetworkX, Pandas, NumPy
- **Frontend:** HTML, CSS

## 📊 Dataset
- Cryptocurrency transaction dataset (Kaggle)
- Includes transaction features, timestamps, and labels

## 🧠 Models Implemented
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE

## 🏗️ System Architecture
_Add your architecture diagram here_

![Architecture](images/architecture.png)

## 📸 Output Screenshots
_Add your project outputs here_

![Output](images/output1.png)

## 📐 UML Diagrams
_Add your UML diagrams here_

![Class Diagram](images/class-diagram.png)
![Use Case](images/usecase.png)

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app.py
