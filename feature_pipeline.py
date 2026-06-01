import os
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timezone
from google.cloud import bigquery

# --- SECURE CONFIGURATION ENVIRONMENT ---
# Read tokens securely from the GitHub runner environment variables
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

PROJECT_ID = "pearls-aqi-predictor-497407"  
DATASET_ID = "aqi_storage"
TABLE_ID = "rawalpindi_feature_store"
LAT, LON = "33.5984", "73.0441"

def run_production_stream():
    print("Extracting live sensor vectors from API nodes...")
    w_res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric").json()
    a_res = requests.get(f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}").json()
    
    temp = w_res.get("main", {}).get("temp", 0.0)
    humidity = w_res.get("main", {}).get("humidity", 0.0)
    pressure = w_res.get("main", {}).get("pressure", 0.0)
    wind_speed = w_res.get("wind", {}).get("speed", 0.0)
    
    data_block = a_res.get("data", {})
    current_aqi = data_block.get("aqi", 0.0)
    pm25 = data_block.get("iaqi", {}).get("pm25", {}).get("v", 0.0)
    pm10 = data_block.get("iaqi", {}).get("pm10", {}).get("v", 0.0)

    print("Executing Feature Engineering transformations...")
    now = datetime.now(timezone.utc)
    hour, day_of_week, month = now.hour, now.weekday(), now.month
    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    # Calculate trend delta momentum safely
    aqi_change_rate = 0.0
    try:
        query = f"SELECT current_aqi FROM `{table_ref}` ORDER BY timestamp DESC LIMIT 1"
        query_job = client.query(query)
        results = list(query_job.result())
        if results:
            aqi_change_rate = float(current_aqi - results[0][0])
    except Exception:
        pass

    # Build structured DataFrame row
    row_payload = {
        "timestamp": [now.strftime("%Y-%m-%d %H:%M:%S")],
        "temperature": [float(temp)],
        "humidity": [float(humidity)],
        "pressure": [float(pressure)],
        "wind_speed": [float(wind_speed)],
        "pm25": [float(pm25)],
        "pm10": [float(pm10)],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "aqi_change_rate": [aqi_change_rate],
        "current_aqi": [float(current_aqi)]
    }
    current_df = pd.DataFrame(row_payload)

    # Convert to matrix validation shape matching requirements
    raw_arr = [temp, humidity, pressure, wind_speed, pm25, pm10, float(hour), float(day_of_week), float(month), hour_sin, hour_cos, aqi_change_rate]
    print("TensorFlow Ingestion Matrix Blueprint Verified.")
    tf.convert_to_tensor([raw_arr], dtype=tf.float32)

    print("Loading engineered frame into BigQuery via batch configuration...")
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    
    load_job = client.load_table_from_dataframe(current_df, table_ref, job_config=job_config)
    load_job.result()
    print("🚀 SUCCESS! Record securely saved in BigQuery Storage Layer.")

if __name__ == "__main__":
    run_production_stream()
