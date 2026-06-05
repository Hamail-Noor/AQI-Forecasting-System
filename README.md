# 🇵🇰 Rawalpindi Real-Time AQI Production Dashboard & Forecasting System

[![Live App Demo](https://img.shields.io/badge/🚀%20LIVE%20PRODUCTION%20DEMO-STREAMLIT%20CLOUD-FF4B4B?style=for-the-badge&logo=streamlit)](https://aqi-forecasting-system-dpxybkgx2l5d686kpmyzle.streamlit.app/)
[![BigQuery Dataset](https://img.shields.io/badge/☁️%20DATASET%20REPOSITORY-GOOGLE%20BIGQUERY-4285F4?style=for-the-badge&logo=google-cloud)](https://console.cloud.google.com/bigquery?project=pearls-aqi-predictor-497407)

> ### 🎯 Production Gateway Links for Recruiters & Reviewers
> - **Live Web Application:** [https://aqi-forecasting-system-dpxybkgx2l5d686kpmyzle.streamlit.app/](https://aqi-forecasting-system-dpxybkgx2l5d686kpmyzle.streamlit.app/)
> - **Google Cloud BigQuery Console Portal:** [https://console.cloud.google.com/bigquery?project=pearls-aqi-predictor-497407](https://console.cloud.google.com/bigquery?project=pearls-aqi-predictor-497407)

An end-to-end, production-grade machine learning ecosystem designed to monitor, analyze, and forecast real-time Air Quality Index (AQI) levels for Rawalpindi, Pakistan. This repository implements a fully decoupled data engineering architecture that interacts live with a serverless cloud feature store to run continuous predictive inference.

---

## 🛠️ Production Stack Architecture

The pipeline completely bypasses fragile static file-loading setups by fully separating data persistence, cloud runtime compute, and the visualization layer:

- **Data Storage Layer:** Serverless Enterprise Google BigQuery Feature Store (`rawalpindi_feature_store`).
- **Inference Pipeline Engine:** Optimized Scikit-Learn Ridge Linear Regression Model with $L_2$ Regularization.
- **Explainable AI (XAI) Framework:** Shapley Additive Explanations (SHAP) Game-Theoretic Models.
- **Deployment Infrastructure:** Streamlit Cloud Virtualized Linux Container Engine synced to automated GitHub CI/CD loops.
- **Ground-Truth Validation Networks:** Verifiable cross-reference integrations with IQAir and OpenWeatherMap API engines.

---

## 📊 Core Engineering Features & Deep Technical Details

### 1. Advanced Live Cloud Feature Ingestion & ETL Pipeline
The web serving layer completely bypasses file-system overhead by executing secure, serverless queries directly against Google Cloud Platform (GCP) BigQuery infrastructure via the `google-cloud-bigquery` library.

* **The Streaming Architecture:** A cron-driven upstream serverless collector pushes environmental parameters into BigQuery at regular intervals. The application establishes an authenticated client session using service account key metadata stored within encrypted environment variables.
* **Chronological Optimization:** To minimize billing queries and optimize cache hit rates, the inference engine isolates only the true, state-of-the-moment sensor snapshot using a fast, indexed chronological sort filter:
```sql
SELECT 
    timestamp, temperature, humidity, pressure, wind_speed, pm25, pm10, 
    hour_sin, hour_cos, aqi_change_rate 
FROM `pearls-aqi-predictor-497407.aqi_storage.rawalpindi_feature_store` 
ORDER BY timestamp DESC 
LIMIT 1
