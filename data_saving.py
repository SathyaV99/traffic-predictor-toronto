from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SaveProcessedData").getOrCreate()

HDFS_BASE_PATH = "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/"

# Load final processed data
final_df = spark.read.parquet(f"{HDFS_BASE_PATH}final_traffic_weather.parquet")

# Save to CSV or other formats for analysis
final_df.write.mode("overwrite").csv(f"{HDFS_BASE_PATH}final_traffic_weather.csv", header=True)

print("Final Data Saved for Analysis.")

