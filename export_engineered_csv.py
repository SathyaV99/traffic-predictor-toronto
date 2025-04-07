from pyspark.sql import SparkSession

def export_engineered_data(spark):
    print("📦 Loading predictions from parquet...")
    df = spark.read.parquet("traffic_optimized_predictions.parquet")

    print("🧹 Dropping unsupported vector columns for CSV export...")

    # Drop vector and struct fields before CSV export
    unsupported_types = ["vector", "struct"]
    columns_to_drop = [name for name, dtype in df.dtypes if any(t in dtype for t in unsupported_types)]
    print(f"⛔ Dropping columns: {columns_to_drop}")
    df_clean = df.drop(*columns_to_drop)

    # ✅ Export human-readable CSV
    df_clean.write.mode("overwrite") \
        .option("header", True) \
        .csv("traffic_engineered_data_csv")

    print("✅ CSV exported: traffic_engineered_data_csv/")

    # ✅ Also export full data with all fields (for advanced analysis)
    df.write.mode("overwrite") \
        .parquet("traffic_engineered_data_full.parquet")

    print("✅ Full Parquet exported: traffic_engineered_data_full.parquet/")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ExportEngineeredData").getOrCreate()
    export_engineered_data(spark)
    spark.stop()
