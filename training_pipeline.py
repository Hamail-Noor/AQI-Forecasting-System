import pickle
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# --- SETTINGS ---
PROJECT_ID = "pearls-aqi-predictor-497407"  
DATASET_ID = "aqi_storage"
TABLE_ID = "rawalpindi_feature_store"

def execute_model_training_pipeline():
    print("🔄 Pulling historical feature matrix from BigQuery...")
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    query = f"SELECT * FROM `{table_ref}` ORDER BY timestamp ASC"
    df = client.query(query).to_dataframe()
    
    feature_cols = [
        "temperature", "humidity", "pressure", "wind_speed", 
        "pm25", "pm10", "hour_sin", "hour_cos", "aqi_change_rate"
    ]
    target_col = "current_aqi"
    
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🚀 Training Ridge Regression Champion Model Architecture...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)

    # Serialize champion asset directly to repository storage workspace
    with open("best_aqi_model.pkl", "wb") as f:
        pickle.dump(ridge_model, f)
        
    print("📁 Champion Model asset successfully compiled and saved as: 'best_aqi_model.pkl'")

if __name__ == "__main__":
    execute_model_training_pipeline()
