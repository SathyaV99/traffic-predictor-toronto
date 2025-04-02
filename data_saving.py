from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SaveProcessedData").getOrCreate()

# Load final processed data
final_df = spark.read.parquet("hdfs://path/to/final_traffic_weather.parquet")

# Save to CSV or other formats for analysis
final_df.write.mode("overwrite").csv("hdfs://path/to/final_traffic_weather.csv", header=True)

print("Final Data Saved for Analysis.")

