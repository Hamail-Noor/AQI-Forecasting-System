import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

# --- SECURE CONFIGURATION ENVIRONMENT ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

PROJECT_ID = "pearls-aqi-predictor-497407"  
DATASET_ID = "aqi_storage"
TABLE_ID = "rawalpindi_feature_store"
LAT, LON = "33.5984", "73.0441"

def run_production_stream():
    print("🔄 Initializing fail-safe PKT ingestion stream...")
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    # 1. Fetch current status from APIs
    try:
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric"
        w_res = requests.get(w_url, timeout=15).json()
    except Exception as e:
        print(f"⚠️ Weather API Warning: {e}")
        w_res = {}

    temp = w_res.get("main", {}).get("temp", 25.0)
    humidity = w_res.get("main", {}).get("humidity", 50.0)
    pressure = w_res.get("main", {}).get("pressure", 1010.0)
    wind_speed = w_res.get("wind", {}).get("speed", 1.5)
    
    try:
        a_url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}"
        a_res = requests.get(a_url, timeout=15).json()
    except Exception as e:
        print(f"⚠️ Pollution API Warning: {e}")
        a_res = {}

    data_block = a_res.get("data", {}) if isinstance(a_res.get("data"), dict) else {}
    raw_aqi = data_block.get("aqi", 100.0)
    try:
        current_aqi = float(raw_aqi)
    except (ValueError, TypeError):
        current_aqi = 100.0

    iaqi_block = data_block.get("iaqi", {})
    pm25 = iaqi_block.get("pm25", {}).get("v", current_aqi * 0.9)
    pm10 = iaqi_block.get("pm10", {}).get("v", current_aqi * 0.7)

    # 2. TIMEZONE CONFIGURATION: Force alignment to Pakistan Standard Time (PKT = UTC+5)
    tz_pakistan = timezone(timedelta(hours=5))
    now = datetime.now(tz_pakistan)
    current_hour_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"⏰ Current Pakistan Ingestion Time: {current_hour_str} PKT")

    # 3. Duplicate Detection: Check if this local PKT hour is already logged
    try:
        dup_query = f"""
            SELECT COUNT(1) FROM `{table_ref}` 
            WHERE timestamp BETWEEN '{now.strftime("%Y-%m-%d")} 00:00:00' AND '{now.strftime("%Y-%m-%d")} 23:59:59'
            AND EXTRACT(HOUR FROM timestamp) = {now.hour}
        """
        dup_job = client.query(dup_query)
        dup_count = list(dup_job.result())[0][0]
        if dup_count > 0:
            print(f"🛑 Hour {now.hour} PKT already populated in BigQuery. Skipping to prevent duplicate records.")
            return
    except Exception as e:
        print(f"⚠️ Duplicate check bypassed: {e}")

    # 4. Feature Engineering using Local Time Weights
    hour, day_of_week, month = now.hour, now.weekday(), now.month
    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    
    aqi_change_rate = 0.0
    try:
        query = f"SELECT current_aqi FROM `{table_ref}` ORDER BY timestamp DESC LIMIT 1"
        results = list(client.query(query).result())
        if results:
            aqi_change_rate = float(current_aqi - results[0][0])
    except Exception:
        pass

    # Assemble the payload with the explicit PKT string
    row_payload = {
        "timestamp": [current_hour_str],
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

    print("Loading engineered PKT frame into BigQuery data stream...")
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    load_job = client.load_table_from_dataframe(current_df, table_ref, job_config=job_config)
    load_job.result()
    print("🚀 SUCCESS! Record securely saved using Pakistan Standard Time boundaries.")

if __name__ == "__main__":
    run_production_stream()
