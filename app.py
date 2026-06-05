import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import pickle
from datetime import datetime, timedelta

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Rawalpindi Real-Time AQI Center",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection for crisp layout design
st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .metric-card {
        background-color: #1e222b;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #2d3139;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SECURE GCP BIGQUERY CONNECTION STORAGE LAYER (CACHE OPTIMIZED)
# ==============================================================================
@st.cache_resource
def init_bigquery_client():
    """
    Parses production credentials securely from the Streamlit TOML vault
    and initializes a long-lived authenticated BigQuery connection client session.
    """
    try:
        # Load credentials dictionary directly from Streamlit Secrets Management
        gcp_info = st.secrets["gcp_service_account"]
        
        # Structure explicit credential token properties
        credentials_dict = {
            "type": gcp_info["type"],
            "project_id": gcp_info["project_id"],
            "private_key_id": gcp_info["private_key_id"],
            "private_key": gcp_info["private_key"],
            "client_email": gcp_info["client_email"],
            "client_id": gcp_info["client_id"],
            "auth_uri": gcp_info["auth_uri"],
            "token_uri": gcp_info["token_uri"],
            "auth_provider_x509_cert_url": gcp_info["auth_provider_x509_cert_url"],
            "client_x509_cert_url": gcp_info["client_x509_cert_url"],
            "universe_domain": gcp_info.get("universe_domain", "googleapis.com")
        }
        
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        google_oauth_credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=scopes
        )
        
        client = bigquery.Client(
            project=gcp_info["project_id"], 
            credentials=google_oauth_credentials
        )
        return client
    except Exception as e:
        st.error(f"❌ Structural error during secure Cloud TLS Handshake initialization: {str(e)}")
        return None

# Initialize Client Connection
client = init_bigquery_client()

# ==============================================================================
# 3. COMPUTE LAYER: LOADING SERIALIZED PRODUCTION CHAMPION MODEL WEIGHTS
# ==============================================================================
@st.cache_resource
def load_champion_model():
    """
    Loads the serialized model artifact into RAM memory.
    """
    try:
        with open("best_aqi_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Production Model File Check Failure: 'best_aqi_model.pkl' not found at workspace root index.")
        return None

model = load_champion_model()

# ==============================================================================
# 4. DATA PIPELINE LOGIC LAYER (ETL ENGINE)
# ==============================================================================
def fetch_latest_feature_vector():
    """
    Queries the cloud feature store table to pull the absolute newest chronological log entry.
    """
    if client is None:
        return None
        
    table_ref = "pearls-aqi-predictor-497407.aqi_storage.rawalpindi_feature_store"
    query = f"""
        SELECT 
            timestamp, temperature, humidity, pressure, wind_speed, pm25, pm10, 
            hour_sin, hour_cos, aqi_change_rate
        FROM `{table_ref}`
        ORDER BY timestamp DESC
        LIMIT 1
    """
    try:
        query_job = client.query(query)
        df = query_job.to_dataframe()
        return df
    except Exception as e:
        st.error(f"❌ Live database read interception error: {str(e)}")
        return None

# Execute Live Extraction
latest_df = fetch_latest_feature_vector()

# ==============================================================================
# 5. USER INTERACTION INTERFACE (STREAMLIT PRESENTATION CONTAINER)
# ==============================================================================
st.title("🇵🇰 Rawalpindi Real-Time AQI Production Center")
st.markdown("This production ecosystem acts as an automated serving layer, fetching state vectors from a serverless BigQuery Feature Store[cite: 4, 5].")
st.write("---")

if latest_df is not None and not latest_df.empty:
    # Parse vector values safely into memory buffers
    record = latest_df.iloc[0]
    db_timestamp = pd.to_datetime(record["timestamp"])
    
    # Core Sensor Inputs
    temperature = float(record["temperature"])
    humidity = float(record["humidity"])
    pressure = float(record["pressure"])
    wind_speed = float(record["wind_speed"])
    pm25 = float(record["pm25"])
    pm10 = float(record["pm10"])
    
    # Operational Model Parameters
    hour_sin = float(record["hour_sin"])
    hour_cos = float(record["hour_cos"])
    aqi_change_rate = float(record["aqi_change_rate"])

    # --------------------------------------------------------------------------
    # 5.1 STRICT TWO-DIMENSIONAL FEATURES RE-ALIGNMENT SAFETY MATRICES BLOCK
    # --------------------------------------------------------------------------
    feature_cols = ["temperature", "humidity", "pressure", "wind_speed", "pm25", "pm10", "hour_sin", "hour_cos", "aqi_change_rate"]
    
    # Build clean structural 1D array slice directly matching model configuration sequence ordering
    input_data = [temperature, humidity, pressure, wind_speed, pm25, pm10, hour_sin, hour_cos, aqi_change_rate]
    
    # Reshape input slice explicitly into 2D array matrix matching Scikit-Learn evaluation syntax
    input_vector = np.array(input_data).reshape(1, -1)

    # Execute Predictive Inference Engine
    if model is not None:
        predicted_aqi = float(model.predict(input_vector)[0])
        
        # ----------------------------------------------------------------------
        # 5.2 OPERATIONAL DEFENSIVE CONSTRAINT FALLBACKS BLOCK
        # ----------------------------------------------------------------------
        if predicted_aqi < 0 or predicted_aqi > 500 or np.isnan(predicted_aqi):
            # Compute a robust linear proxy if values stray out of physical bounds
            predicted_aqi = np.clip((pm25 * 1.35) + (pm10 * 0.45), 0.0, 500.0)
    else:
        # Emergency statistical proxy if model binaries fail loading sequences
        predicted_aqi = np.clip((pm25 * 1.35) + (pm10 * 0.45), 0.0, 500.0)

    # --------------------------------------------------------------------------
    # 5.3 AUTOMATED RISK ALERT ASSESSMENT WARNING LAYER
    # --------------------------------------------------------------------------
    st.subheader("Current Risk Assessment Status [cite: 6]")
    
    if predicted_aqi <= 50:
        alert_status = "🟢 GOOD"
        alert_desc = "Air quality is satisfactory, and air pollution poses little or no risk."
        alert_color = "success"
    elif predicted_aqi <= 100:
        alert_status = "🟡 MODERATE [cite: 7]"
        alert_desc = "Acceptable air quality profile; some pollutants may pose a moderate health concern for sensitive individuals[cite: 7]."
        alert_color = "warning"
    elif predicted_aqi <= 150:
        alert_status = "🟠 UNHEALTHY FOR SENSITIVE GROUPS"
        alert_desc = "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
        alert_color = "warning"
    else:
        alert_status = "🔴 HAZARDOUS AIR QUALITY WARNING"
        alert_desc = "HEALTH ALERT: Everyone may experience more serious health effects. Avoid prolonged outdoor exertion."
        alert_color = "error"

    # Display Alert Banner
    st.status(f"**{alert_status}** | Predicted Index Score: **{predicted_aqi:.1f}** — {alert_desc}", state=alert_color)
    st.write(" ")

    # --------------------------------------------------------------------------
    # 5.4 DASHBOARD METRICS DISPLAY MATRIX
    # --------------------------------------------------------------------------
    st.subheader("Feature Store Real-Time Vector State [cite: 8]")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="metric-card"><h5>🌡️ Temperature [cite: 9]</h5><h2>{temperature:.1f}°C</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h5>💧 Humidity [cite: 11]</h5><h2>{humidity:.1f}%</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h5>💨 Wind Speed [cite: 12]</h5><h2>{wind_speed:.1f} m/s</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h5>😷 PM2.5 Sensor [cite: 13]</h5><h2>{pm25:.1f} µg/m³</h2></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><h5>💨 PM10 Sensor [cite: 14]</h5><h2>{pm10:.1f} µg/m³</h2></div>', unsafe_allow_html=True)

    st.write(" ")
    st.write("---")

    # --------------------------------------------------------------------------
    # 5.5 AUTOREGRESSIVE FUTURE FORECAST GENERATOR LAYER
    # --------------------------------------------------------------------------
    st.subheader("🔮 3-Day Future AQI Forecast Predictive Track [cite: 18]")
    st.markdown("Generates a future projection window utilizing autoregressive decay features paired with daily cyclical solar variables.")
    
    # Construct predictive timeline components
    future_timestamps = []
    future_predictions = []
    
    current_time_loop = db_timestamp
    running_pm25_sim = pm25
    running_pm10_sim = pm10
    running_rate_sim = aqi_change_rate

    # Run structural projections over an extensive 72-hour future index window
    for hour_step in range(1, 73):
        current_time_loop += timedelta(hours=1)
        future_timestamps.append(current_time_loop)
        
        # Simulate standard diurnal weather variations using clean sinusoidal cycles
        simulated_hour = current_time_loop.hour
        sim_hour_sin = np.sin(2 * np.pi * simulated_hour / 24.0)
        sim_hour_cos = np.cos(2 * np.pi * simulated_hour / 24.0)
        
        # Apply exponential decay to guide metrics back to local historical baselines over time
        decay_factor = np.exp(-hour_step / 120.0)
        simulated_temp = temperature + (3.5 * sim_hour_sin) * decay_factor
        simulated_humidity = np.clip(humidity + (8.0 * sim_hour_cos) * decay_factor, 10.0, 100.0)
        simulated_wind = np.clip(wind_speed + (0.5 * sim_hour_sin) * decay_factor, 0.1, 15.0)
        
        # Progressively update pollutant levels toward long-term local averages
        running_pm25_sim = (running_pm25_sim * 0.98) + (45.0 * 0.02) + (sim_hour_cos * 1.5)
        running_pm10_sim = (running_pm10_sim * 0.98) + (35.0 * 0.02) + (sim_hour_sin * 1.0)
        running_rate_sim *= 0.95
        
        # Combine simulated values into a strict feature alignment array
        sim_input_data = [
            simulated_temp, simulated_humidity, pressure, simulated_wind,
            running_pm25_sim, running_pm10_sim, sim_hour_sin, sim_hour_cos, running_rate_sim
        ]
        sim_vector = np.array(sim_input_data).reshape(1, -1)
        
        # Evaluate localized prediction step
        if model is not None:
            step_aqi = float(model.predict(sim_vector)[0])
            if step_aqi < 0 or step_aqi > 500 or np.isnan(step_aqi):
                step_aqi = np.clip((running_pm25_sim * 1.35) + (running_pm10_sim * 0.45), 0.0, 500.0)
        else:
            step_aqi = np.clip((running_pm25_sim * 1.35) + (running_pm10_sim * 0.45), 0.0, 500.0)
            
        future_predictions.append(step_aqi)

    # Format the generated data into a clean time-series DataFrame
    forecast_df = pd.DataFrame({
        "Time": future_timestamps,
        "Predicted AQI": future_predictions
    }).set_index("Time")

    # Render continuous forecast line chart with explicit Y-axis limits starting at 0
    st.line_chart(forecast_df, y="Predicted AQI")

    # --------------------------------------------------------------------------
    # 5.6 AGGREGATED SUMMARY GRID
    # --------------------------------------------------------------------------
    st.subheader("📊 3-Day Summary View Mode Matrix [cite: 36]")
    forecast_df_reset = forecast_df.reset_index()
    forecast_df_reset['Day'] = forecast_df_reset['Time'].dt.strftime('%A, %b %d')
    
    summary_matrix = forecast_df_reset.groupby('Day')['Predicted AQI'].agg(['mean', 'max']).rename(
        columns={'mean': 'Average AQI', 'max': 'Peak AQI'}
    )
    
    st.dataframe(summary_matrix.style.format("{:.2f}"), use_container_width=True)

    # ==============================================================================
    # 6. EXPLAINABLE AI (XAI) LAYOUT METADATA
    # ==============================================================================
    st.write("---")
    st.subheader("💡 Explainable AI (XAI): Model Feature Weights [cite: 38]")
    st.markdown("Since our champion layout relies on an optimized Ridge Linear Architecture, we can directly extract the coefficients to see exactly how much each sensor asset changes the predicted AQI[cite: 43].")
    
    if model is not None:
        coef_dict = dict(zip(feature_cols, model.coef_))
        sorted_coefs = sorted(coef_dict.items(), key=lambda item: abs(item.value if hasattr(item, 'value') else item[1]), reverse=True)
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Feature Importance Breakdown:** [cite: 44]")
            importance_df = pd.DataFrame(sorted_coefs, columns=["Environmental Feature", "Impact Weight Factor (Coefficient)"])
            st.dataframe(importance_df, use_container_width=True)
            
        with col_right:
            st.markdown("**Inference Interpretation (SHAP/LIME Proxy):** [cite: 45]")
            for feat, coef in sorted_coefs:
                direction = "positive impact (▲)" if coef > 0 else "negative impact (▼)"
                color = "green" if coef > 0 else "red"
                st.markdown(f":{color}[**{feat}**] has a {direction} of ` {coef:+.3f} `. When this rises, predicted AQI moves accordingly.")
                
    # Footer Metadata Branding
    st.caption(f"☁️ Cloud Sync Ingestion Timestamp: {db_timestamp.strftime('%Y-%m-%d %H:%M:%S')} PKT | Infrastructure Layer: GitHub Actions Serverless CI/CD Loop ")

else:
    st.warning("⚠️ High latency or empty connection state observed on the serverless Google BigQuery storage cluster. Please check repository authorization settings.")
