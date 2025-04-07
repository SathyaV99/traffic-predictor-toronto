from pyspark.sql import SparkSession

from traffic_data_transformation import run_transformation
from data_ingestion import run_ingestion
from data_preprocessing import run_preprocessing
from data_merge import run_merge
from data_saving import run_saving

print("Running Data Pipeline...")

# Create a single Spark session
spark = SparkSession.builder.appName("TrafficPipeline").getOrCreate()

# Call your steps in order
run_transformation(spark)
run_ingestion(spark)
run_preprocessing(spark)
run_merge(spark)
run_saving(spark)

# Stop Spark session
spark.stop()

print("✅ Data Pipeline Completed Successfully!")
