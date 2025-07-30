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

## Methodology

### Step 1 - Start Hadoop and Spark

```bash
# Hadoop
cd ~/hadoop-3.4.1/sbin
./start-dfs.sh
./start-yarn.sh

# Spark
cd /opt/spark/sbin
start-master.sh
start-worker.sh spark://<your-host>:7077

# Check processes
jps
```

### Step 2 - Create and Activate Virtual Environment (if not already created)

```bash
python3 -m venv spark-venv
source spark-venv/bin/activate
```

### Step 3 - Fix HDFS Directory Paths

```bash
hdfs dfs -rm -r /user/hadoop/toronto_traffic/input
hdfs dfs -rm -r /user/hadoop/toronto_traffic/
hdfs dfs -mkdir /user/hdoop/toronto_traffic
hdfs dfs -mkdir /user/hdoop/toronto_traffic/input
```

### Step 4 - Upload Data to HDFS

```bash
hdfs dfs -put path/to/*.csv /user/hdoop/toronto_traffic/input
```

### Step 5 - Run the Pipeline

```bash
export PYSPARK_PYTHON=/home/hdoop/spark-venv/bin/python
spark-submit run_pipeline.py
```

### Step 6 - Pipeline Modules Overview

- `run_transformation(spark)`: Reads and reshapes traffic data into long format
- `run_ingestion(spark)`: Reads and combines weather and traffic into Parquet
- `run_preprocessing(spark)`: Filters Toronto records, fills nulls
- `run_merge(spark)`: Joins weather and traffic on `date`
- `run_saving(spark)`: Converts Parquet to CSV

### Step 7 - Handle Multiple Spark Sessions

Avoid creating multiple Spark sessions across files. Use imports and function calls instead of `os.system`.

### Step 8 - Export Java Path (if Spark Worker doesn't show up)

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

Make this permanent by appending to `~/.bashrc`.

### Step 9 - Merge Output CSV Parts

```bash
hdfs dfs -getmerge /user/hdoop/toronto_traffic/input/final_traffic_weather.csv final_traffic_weather.csv
```

### Step 10 - View in Jupyter Notebook

```bash
pip install notebook
jupyter notebook
```

Copy URL shown in terminal and open in browser.

### Step 11 - Configure PySpark in Jupyter

```bash
pip install pyspark ipykernel
python -m ipykernel install --user --name=spark-venv --display-name "Spark (PySpark)"
```

### Step 12 - Feature Engineering & Cleaning

```bash
spark-submit engineer_balance_export.py
spark-submit clean_nulls_from_csv.py
spark-submit combine.py
```

### Step 13 - Upload Combined File to HDFS

```bash
hdfs dfs -put combined.csv /user/hdoop/toronto_traffic/input
```

### Step 14 - Run Final Prediction

```bash
spark-submit predict_final_pipeline.py
```

### Step 15 - Download Final Outputs

```bash
hdfs dfs -get /user/hdoop/toronto_traffic/output/final_predictions_csv
hdfs dfs -get /user/hdoop/toronto_traffic/output/final_rf_model
```

### Step 16 - Evaluate in Notebook

Open `predict_final_pipeline_analysis.ipynb` in Folder "Iteration3"

**Classification Report:**
- Accuracy: 0.6420
- F1 Score: 0.6373
- Precision: 0.6494
- Recall: 0.6420

**Confusion Matrix:**
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

Open and explore results using the notebook: `predict_final_pipeline_analysis.ipynb`

![image](https://github.com/user-attachments/assets/4fe0d196-4ac9-485b-894f-514ad30666ce)


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
