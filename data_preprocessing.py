"""from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Initialize Spark Session
spark = SparkSession.builder.appName("TrafficPreprocessing").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # Optional: reduce verbosity

HDFS_BASE_PATH = "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/"

# Load raw data
traffic_df = spark.read.parquet(f"{HDFS_BASE_PATH}raw_traffic.parquet")
weather_df = spark.read.parquet(f"{HDFS_BASE_PATH}raw_weather.parquet")

print("✅ Raw data loaded successfully.")

# Filter only Toronto data
traffic_df = traffic_df.filter(traffic_df["traffic_source"] == "Toronto")

print("✅ Filtered Toronto traffic data.")

# ----------- 🔥 Fix: Ensure consistent data types for traffic date columns -----------

# Identify all date columns (they start with 'x' based on your dataset)
date_columns = [col_name for col_name in traffic_df.columns if col_name.startswith('x')]
print(f"Date_Columns Length: {len(date_columns)}")
# Cast all date columns to double and handle invalid entries
for column in date_columns:
    print(column)
    traffic_df = traffic_df.withColumn(
        column,
        when(col(column).cast("double").isNotNull(), col(column).cast("double")).otherwise(0.0)  # replace invalids with 0.0
    )

print(f"✅ Cleaned and casted {len(date_columns)} traffic date columns to consistent types.")

# Handle other missing values in traffic data
traffic_df = traffic_df.fillna({"traffic_count": 0})

# Handle missing values in weather data
weather_df = weather_df.fillna({
    "Max Temp (°C)": weather_df.selectExpr("percentile(Max Temp (°C), 0.5)").collect()[0][0],
    "Min Temp (°C)": weather_df.selectExpr("percentile(Min Temp (°C), 0.5)").collect()[0][0],
    "Total Precip (mm)": 0  # Assume no rain if missing
})

print("✅ Filled missing values in traffic and weather data.")

# Save cleaned data
traffic_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}cleaned_traffic.parquet")
weather_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}cleaned_weather.parquet")

print("🎉 Data Preprocessing Complete.")


"""


from pyspark.sql.functions import col

def run_preprocessing(spark):
    HDFS_BASE_PATH = "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/"

    # Load raw data
    traffic_df = spark.read.parquet(f"{HDFS_BASE_PATH}raw_traffic.parquet")
    weather_df = spark.read.parquet(f"{HDFS_BASE_PATH}raw_weather.parquet")

    # Filter only Toronto traffic data
    traffic_df = traffic_df.filter(traffic_df["traffic_source"] == "Toronto")

    # Handle missing values
    traffic_df = traffic_df.fillna({"traffic_count": 0})

    # Fill weather missing values
    median_max_temp = weather_df.selectExpr("percentile(`Max Temp (°C)`, 0.5)").collect()[0][0]
    median_min_temp = weather_df.selectExpr("percentile(`Min Temp (°C)`, 0.5)").collect()[0][0]

    weather_df = weather_df.fillna({
        "Max Temp (°C)": median_max_temp,
        "Min Temp (°C)": median_min_temp,
        "Total Precip (mm)": 0  # Assume no rain if missing
    })

    # Save cleaned data
    traffic_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}cleaned_traffic.parquet")
    weather_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}cleaned_weather.parquet")

    print("✅ Data Preprocessing Complete.")
