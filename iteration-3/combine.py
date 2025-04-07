from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("CombineCSVs").getOrCreate()

# Path to folder containing CSV files
input_folder = "/home/hdoop/bigdata_project/Traffic_Congestion_BigData/iteration-3/cleaned_csv"

# Load all CSV files in the folder into a single DataFrame
df = spark.read.option("header", "true").csv(input_folder)

# Show sample
df.show()

# Save as single combined CSV (coalesce to 1 partition)
output_path = "/home/hdoop/bigdata_project/Traffic_Congestion_BigData/iteration-3/combined.csv"

df.coalesce(1).write.option("header", "true").csv(output_path)
