# Toronto Traffic Prediction with Apache Spark and Hadoop

## Overview
This project builds a big data pipeline to process raw traffic and weather data for the city of Toronto. Using Hadoop and Spark, we transform and clean the data, then use it to train a machine learning model to predict traffic congestion levels.

![image](https://github.com/user-attachments/assets/cd026e3b-2640-4355-a741-48b5c1e3f90f)

---

## Objectives

- Set up a big data environment using Hadoop and Spark
- Clean and process large traffic and weather datasets
- Merge datasets to form a unified source for ML
- Build and evaluate a prediction model
- Visualize and access results through Jupyter

---

## Data Collection & Preprocessing

### Traffic Data

- Daily traffic counts from 2022 to 2024 were collected for over 1100 traffic signal locations across Toronto.
- The original dataset had 335 rows and 1100 columns, where each column represented traffic count for a day at a given location.
- To make it usable for time-series analysis and machine learning, the dataset was converted to a long format (~292,611 rows).
- Example intersections covered include:
  - YORK ST / BREMNER BLVD / RAPTORS WAY
  - SPADINA AVE / FRONT ST W
  - EGLINTON AVE E / DON MILLS RD
  - SHEPPARD AVE E / MCCOWAN RD
  - YONGE ST / DUNDAS ST
  - and 100+ more across all Toronto boroughs.

📎 [Source](https://www150.statcan.gc.ca/n1/pub/71-607-x/71-607-x2022018-eng.htm)

### Weather Data

- Daily weather records were collected from Environment Canada for 2022–2024.
- Each day’s record included temperature, precipitation, wind gusts, and quality flags.
- Sample fields include:
  - `Max Temp (°C)`, `Min Temp (°C)`, `Total Rain (mm)`, `Snow on Grnd (cm)`, `Dir of Max Gust (10s deg)`, `Spd of Max Gust (km/h)`.

📎 [Source](https://climate.weather.gc.ca/climate_data/daily_data_e.html?StationID=51459&timeframe=2&StartYear=1840&EndYear=2025&Day=27&Year=2024&Month=4)

### Combined Dataset

- After transformation and merging, the final dataset (`final_traffic_weather.csv`) had:
  - 37 columns
  - 292,611 rows
- Fields included:
  - `date`, `traffic_camera`, `traffic_count`, `Longitude (x)`, `Latitude (y)`
  - All weather features listed above
  - Suitable for training supervised ML models like Random Forest

---


## Setup

### 1. Start Hadoop & Spark
```bash
# Hadoop
./start-dfs.sh
./start-yarn.sh

# Spark
start-master.sh
start-worker.sh spark://<your-machine-name>:7077

# Check
jps  # confirm processes like NameNode, DataNode, ResourceManager, etc.
```

### 2. Virtual Environment
```bash
python3 -m venv spark-venv
source spark-venv/bin/activate
```

---

## Data Ingestion

### 1. HDFS Directory Setup
```bash
hdfs dfs -mkdir /user/hdoop/toronto_traffic
hdfs dfs -mkdir /user/hdoop/toronto_traffic/input
```

### 2. Upload Data
```bash
hdfs dfs -put *.csv /user/hdoop/toronto_traffic/input
```

---

## Spark Job Pipeline

### 1. `run_transformation.py`
- Converts wide-format traffic data into long format
- Fixes malformed headers and parses date strings
- Saves to: `transformed_traffic_data`

### 2. `run_ingestion.py`
- Reads transformed traffic and multiple weather files
- Combines into Parquet: `raw_traffic.parquet`, `raw_weather.parquet`

### 3. `run_preprocessing.py`
- Filters Toronto-specific rows
- Fills missing values (0 or median)
- Outputs: `cleaned_traffic.parquet`, `cleaned_weather.parquet`

### 4. `run_merge.py`
- Merges traffic and weather on `date`
- Output: `final_traffic_weather.parquet`

### 5. `run_saving.py`
- Converts merged Parquet to CSV
- Output: `final_traffic_weather.csv`

---

## Feature Engineering & Cleaning

### Engineer Features
```bash
spark-submit engineer_balance_export.py
```

### Handle Nulls
```bash
spark-submit clean_nulls_from_csv.py
```

### Combine Cleaned CSV
```bash
spark-submit combine.py
```

### Upload to HDFS
```bash
hdfs dfs -put combined.csv /user/hdoop/toronto_traffic/input
```

---

## Model Training and Prediction

```bash
spark-submit predict_final_pipeline.py
```

Output:
- `final_predictions_csv/`
- `final_rf_model/`

---

## Evaluation Metrics
- Accuracy: 0.6420
- F1 Score: 0.6373
- Precision: 0.6494
- Recall: 0.6420
- Confusion Matrix:
```
[[11488. 10241.]
 [ 5344. 16459.]]
```

---

## Visualization

### Jupyter Setup
```bash
pip install notebook pyspark ipykernel
python -m ipykernel install --user --name=spark-venv --display-name "Spark (PySpark)"
jupyter notebook
```

Open the URL displayed and explore results using the notebook: `predict_final_pipeline_analysis.ipynb`

---

## Final Output Files (HDFS)
| File | Format | Purpose |
|------|--------|---------|
| transformed_traffic_data/ | CSV | Long format traffic |
| raw_traffic.parquet | Parquet | Clean input |
| raw_weather.parquet | Parquet | Weather (3 years) |
| cleaned_traffic.parquet | Parquet | Filled & filtered |
| cleaned_weather.parquet | Parquet | Weather cleaned |
| final_traffic_weather.parquet | Parquet | Merged |
| final_traffic_weather.csv | CSV | Easy access |
| combined.csv | CSV | Cleaned, engineered |
| final_predictions_csv/ | CSV | Model predictions |
| final_rf_model/ | Binary | Trained model |

---

## Note
- Keep environment variables consistent (`JAVA_HOME`, `PATH`, etc.)
- Avoid creating multiple Spark sessions
- Always deactivate and reactivate your virtual environment if unexpected errors occur

---
