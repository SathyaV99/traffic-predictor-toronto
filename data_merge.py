from pyspark.sql import SparkSession

def run_merge(spark):
    HDFS_BASE_PATH = "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/"

    # Load transformed traffic & weather data
    traffic_long_df = spark.read.parquet(f"{HDFS_BASE_PATH}cleaned_traffic.parquet")
    weather_df = spark.read.parquet(f"{HDFS_BASE_PATH}cleaned_weather.parquet")

    # Rename weather date column to match traffic data
    weather_df = weather_df.withColumnRenamed("Date/Time", "date")

    # Merge datasets on 'date'
    final_df = traffic_long_df.join(weather_df, on="date", how="left")

    # Save merged dataset
    final_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}final_traffic_weather.parquet")

    print("✅ Data Merging Complete.")
