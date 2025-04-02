from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TrafficMerge").getOrCreate()

# Load transformed traffic & weather data
traffic_long_df = spark.read.parquet("hdfs://path/to/transformed_traffic.parquet")
weather_df = spark.read.parquet("hdfs://path/to/cleaned_weather.parquet")

# Merge datasets on 'date'
final_df = traffic_long_df.join(weather_df, on="date", how="left")

# Save merged dataset
final_df.write.mode("overwrite").parquet("hdfs://path/to/final_traffic_weather.parquet")

print("Data Merging Complete.")
