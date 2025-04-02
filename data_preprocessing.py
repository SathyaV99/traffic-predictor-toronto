from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("TrafficPreprocessing").getOrCreate()

# Load raw data
traffic_df = spark.read.parquet("hdfs://path/to/raw_traffic.parquet")
weather_df = spark.read.parquet("hdfs://path/to/raw_weather.parquet")

# Filter only Toronto data
traffic_df = traffic_df.filter(traffic_df["traffic_source"] == "Toronto")

# Handle missing values
traffic_df = traffic_df.fillna({"traffic_count": 0})
weather_df = weather_df.fillna({"Max Temp (°C)": weather_df.selectExpr("percentile(Max Temp (°C), 0.5)").collect()[0][0]})
weather_df = weather_df.fillna({"Min Temp (°C)": weather_df.selectExpr("percentile(Min Temp (°C), 0.5)").collect()[0][0]})
weather_df = weather_df.fillna({"Total Precip (mm)": 0})  # Assume no rain if missing

# Save cleaned data
traffic_df.write.mode("overwrite").parquet("hdfs://path/to/cleaned_traffic.parquet")
weather_df.write.mode("overwrite").parquet("hdfs://path/to/cleaned_weather.parquet")

print("Data Preprocessing Complete.")
