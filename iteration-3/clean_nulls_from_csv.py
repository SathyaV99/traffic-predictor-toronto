from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def clean_nulls_from_csv(input_path, output_path, drop_high_null_cols=True):
    spark = SparkSession.builder.appName("RemoveNulls").getOrCreate()

    # Load the combined CSV
    df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

    # Drop rows with nulls in important columns (for modeling)
    important_cols = ["traffic_count", "Month", "Day", "Mean Temp (°C)", "Min Temp (°C)", "Max Temp (°C)", "Total Precip (mm)"]
    df_clean = df.dropna(subset=important_cols)

    if drop_high_null_cols:
        # Drop high-null columns (>25% missing)
        df_clean = df_clean.drop("Spd of Max Gust (km/h)", "Dir of Max Gust (10s deg)")

    # Write cleaned CSV
    df_clean.write.mode("overwrite").option("header", True).csv(output_path)

    print("✅ Cleaned CSV written to:", output_path)
    spark.stop()

if __name__ == "__main__":
    input_csv_path = "/home/hdoop/bigdata_project/Traffic_Congestion_BigData/iteration-3/cleaned_balanced_traffic_data_csv"

    output_csv_path = "/home/hdoop/bigdata_project/Traffic_Congestion_BigData/iteration-3/cleaned_csv"
    clean_nulls_from_csv(input_csv_path, output_csv_path)
