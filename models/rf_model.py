from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
import joblib
import pandas as pd
import os

def create_rf_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def train_rf(X_train, y_train, X_test, y_test):
    os.makedirs('models', exist_ok=True)
    model = create_rf_model()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print(f"Random Forest - Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {rec:.4f}")
    
    joblib.dump(model, 'models/rf_model.joblib')
    return acc, f1, rec

if __name__ == "__main__":
    # Test script would go here if needed individually
    pass
