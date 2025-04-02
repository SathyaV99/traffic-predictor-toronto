from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, to_date

spark = SparkSession.builder.appName("TrafficTransformation").getOrCreate()

# Load cleaned traffic data
traffic_df = spark.read.parquet("hdfs://path/to/cleaned_traffic.parquet")

# Convert to long format
traffic_long_df = traffic_df.selectExpr(
    "WKT", "CSDUID", "traffic_camera", "traffic_source", "camera_road",
    "stack(15, 'x2022_02_02', x2022_02_02, 'x2022_02_03', x2022_02_03, 'x2022_02_04', x2022_02_04, \
    'x2022_02_05', x2022_02_05, 'x2022_02_06', x2022_02_06, 'x2022_02_07', x2022_02_07, \
    'x2022_02_08', x2022_02_08, 'x2022_02_09', x2022_02_09, 'x2022_02_10', x2022_02_10, \
    'x2022_02_11', x2022_02_11, 'x2022_02_12', x2022_02_12, 'x2022_02_13', x2022_02_13, \
    'x2022_02_14', x2022_02_14, 'x2022_02_15', x2022_02_15, 'x2022_02_16', x2022_02_16, \
    'x2022_02_17', x2022_02_17) as (date, traffic_count)"
)

# Convert 'date' column to proper format
traffic_long_df = traffic_long_df.withColumn("date", to_date(expr("substring(date, 2, 10)"), "yyyy_MM_dd"))

# Save transformed data
traffic_long_df.write.mode("overwrite").parquet("hdfs://path/to/transformed_traffic.parquet")

print("Traffic Data Transformation Complete.")
