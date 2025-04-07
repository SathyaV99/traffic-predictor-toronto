from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, dayofweek, udf, mean
from pyspark.sql.types import IntegerType

def clean_and_engineer_data(spark):
    # Load input CSV
    df = spark.read.option("header", True).option("inferSchema", True).csv("final_traffic_weather.csv")

    # Drop unnecessary columns
    drop_cols = [
        "WKT", "CSDUID", "traffic_camera", "traffic_source", "camera_road",
        "Longitude (x)", "Latitude (y)", "Station Name", "Climate ID", "Data Quality",
        "Max Temp Flag", "Min Temp Flag", "Mean Temp Flag", "Heat Deg Days (°C)", "Heat Deg Days Flag",
        "Cool Deg Days (°C)", "Cool Deg Days Flag", "Total Rain Flag", "Total Snow Flag",
        "Total Precip Flag", "Snow on Grnd Flag", "Dir of Max Gust Flag", "Spd of Max Gust Flag",
        "Snow on Grnd (cm)"
    ]
    df = df.drop(*drop_cols)

    # Drop rows with essential nulls
    required_fields = ["traffic_count", "Month", "Day", "Mean Temp (°C)", "Total Precip (mm)", "Max Temp (°C)", "Min Temp (°C)"]
    df = df.dropna(subset=required_fields)

    # Feature engineering
    df = df.withColumn("day_of_week", dayofweek(col("date")))
    is_weekend_udf = udf(lambda dow: 1 if dow in [1, 7] else 0, IntegerType())
    df = df.withColumn("is_weekend", is_weekend_udf(col("day_of_week")))

    df = df.withColumn("temp_range", col("Max Temp (°C)") - col("Min Temp (°C)"))
    df = df.withColumn(
        "temp_range_cat",
        when(col("temp_range") < 5, "Low")
        .when(col("temp_range") < 10, "Medium")
        .otherwise("High")
    )

    df = df.withColumn(
        "season",
        when(col("Month").isin(12, 1, 2), "Winter")
        .when(col("Month").isin(3, 4, 5), "Spring")
        .when(col("Month").isin(6, 7, 8), "Summer")
        .otherwise("Fall")
    )

    # Binary traffic status (0: No Traffic, 1: Traffic)
    mean_count = df.select(mean("traffic_count")).first()[0]
    df = df.withColumn(
        "traffic_status",
        when(col("traffic_count") > mean_count * 0.75, 1).otherwise(0)
    )

    # Handle class imbalance (undersampling)
    class_0 = df.filter(col("traffic_status") == 0)
    class_1 = df.filter(col("traffic_status") == 1)
    minority_size = min(class_0.count(), class_1.count())
    df_balanced = class_0.limit(minority_size).union(class_1.limit(minority_size))

    # Save to CSV
    df_balanced.write.mode("overwrite").option("header", True).csv("cleaned_balanced_traffic_data_csv")
    print("✅ Cleaned and balanced CSV saved at: cleaned_balanced_traffic_data_csv")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("TrafficFeatureEngineering").getOrCreate()
    clean_and_engineer_data(spark)
    spark.stop()
