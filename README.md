Toronto Traffic Prediction with Apache Spark and Hadoop

Overview

This project builds a big data pipeline to process raw traffic and weather data for the city of Toronto. Using Hadoop and Spark, we transform and clean the data, then use it to train a machine learning model to predict traffic congestion levels.

Objectives

Set up a big data environment using Hadoop and Spark

Clean and process large traffic and weather datasets

Merge datasets to form a unified source for ML

Build and evaluate a prediction model

Visualize and access results through Jupyter

Setup

1. Start Hadoop & Spark

# Hadoop
./start-dfs.sh
./start-yarn.sh

# Spark
start-master.sh
start-worker.sh spark://<your-machine-name>:7077

# Check
jps  # confirm processes like NameNode, DataNode, ResourceManager, etc.

2. Virtual Environment

python3 -m venv spark-venv
source spark-venv/bin/activate

Data Ingestion

1. HDFS Directory Setup

hdfs dfs -mkdir /user/hdoop/toronto_traffic
hdfs dfs -mkdir /user/hdoop/toronto_traffic/input

2. Upload Data

hdfs dfs -put *.csv /user/hdoop/toronto_traffic/input

Spark Job Pipeline

1. run_transformation.py

Converts wide-format traffic data into long format

Fixes malformed headers and parses date strings

Saves to: transformed_traffic_data

2. run_ingestion.py

Reads transformed traffic and multiple weather files

Combines into Parquet: raw_traffic.parquet, raw_weather.parquet

3. run_preprocessing.py

Filters Toronto-specific rows

Fills missing values (0 or median)

Outputs: cleaned_traffic.parquet, cleaned_weather.parquet

4. run_merge.py

Merges traffic and weather on date

Output: final_traffic_weather.parquet

5. run_saving.py

Converts merged Parquet to CSV

Output: final_traffic_weather.csv

Feature Engineering & Cleaning

Engineer Features

spark-submit engineer_balance_export.py

Handle Nulls

spark-submit clean_nulls_from_csv.py

Combine Cleaned CSV

spark-submit combine.py

Upload to HDFS

hdfs dfs -put combined.csv /user/hdoop/toronto_traffic/input

Model Training and Prediction

spark-submit predict_final_pipeline.py

Output:

final_predictions_csv/

final_rf_model/

Evaluation Metrics

Accuracy: 0.6420

F1 Score: 0.6373

Precision: 0.6494

Recall: 0.6420

Confusion Matrix:

[[11488. 10241.]
 [ 5344. 16459.]]

Visualization

Jupyter Setup

pip install notebook pyspark ipykernel
python -m ipykernel install --user --name=spark-venv --display-name "Spark (PySpark)"
jupyter notebook

Open the URL displayed and explore results using the notebook: predict_final_pipeline_analysis.ipynb

Final Output Files (HDFS)

File

Format

Purpose

transformed_traffic_data/

CSV

Long format traffic

raw_traffic.parquet

Parquet

Clean input

raw_weather.parquet

Parquet

Weather (3 years)

cleaned_traffic.parquet

Parquet

Filled & filtered

cleaned_weather.parquet

Parquet

Weather cleaned

final_traffic_weather.parquet

Parquet

Merged

final_traffic_weather.csv

CSV

Easy access

combined.csv

CSV

Cleaned, engineered

final_predictions_csv/

CSV

Model predictions

final_rf_model/

Binary

Trained model

Note

Keep environment variables consistent (JAVA_HOME, PATH, etc.)

Avoid creating multiple Spark sessions

Always deactivate and reactivate your virtual environment if unexpected errors occur
