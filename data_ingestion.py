from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TrafficPrediction").getOrCreate()

HDFS_BASE_PATH = "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/"

# Load Traffic Data
traffic_df = spark.read.csv(f"{HDFS_BASE_PATH}transformed_traffic_data", header=True, inferSchema=True)

# Load Weather Data (All Years)
weather_2022 = spark.read.csv(f"{HDFS_BASE_PATH}en_climate_daily_ON_6158731_2022_P1D.csv", header=True, inferSchema=True)
weather_2023 = spark.read.csv(f"{HDFS_BASE_PATH}en_climate_daily_ON_6158731_2023_P1D.csv", header=True, inferSchema=True)
weather_2024 = spark.read.csv(f"{HDFS_BASE_PATH}en_climate_daily_ON_6158731_2024_P1D.csv", header=True, inferSchema=True)

# Combine Weather Data
weather_df = weather_2022.union(weather_2023).union(weather_2024)

# Save for next steps
traffic_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}raw_traffic.parquet")
weather_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}raw_weather.parquet")

print("Data Ingestion Complete.")
