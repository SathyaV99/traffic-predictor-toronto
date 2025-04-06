from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month, sum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Step 1: Initialize Spark Session
spark = SparkSession.builder.appName("TrafficPredictionModel").getOrCreate()

# Step 2: Load Merged Traffic + Weather Data
HDFS_BASE_PATH = "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/"
df = spark.read.parquet(f"{HDFS_BASE_PATH}final_traffic_weather.parquet")

print("✅ Data Loaded Successfully!")

# Step 3: Clean Column Names (to avoid issues with spaces / special characters)
df = df.withColumnRenamed("Max Temp (°C)", "Max_Temp_C") \
       .withColumnRenamed("Min Temp (°C)", "Min_Temp_C") \
       .withColumnRenamed("Total Precip (mm)", "Total_Precip_mm")

# Step 4: Feature Engineering
df = df.withColumn("day_of_week", dayofweek(col("date")))
df = df.withColumn("month", month(col("date")))

# Step 5: Check for Null Values (Debugging aid)
print("✅ Checking for nulls in important columns:")
df.select([sum(col(c).isNull().cast("int")).alias(c) for c in [
    "Max_Temp_C", "Min_Temp_C", "Total_Precip_mm", "day_of_week", "month"
]]).show()

# Step 6: Handle Nulls Before VectorAssembler
df = df.dropna(subset=["Max_Temp_C", "Min_Temp_C", "Total_Precip_mm", "day_of_week", "month"])

print(f"✅ Data count after dropping nulls: {df.count()}")

# Step 7: Prepare Feature Columns
feature_columns = [
    "day_of_week",
    "month",
    "Max_Temp_C",
    "Min_Temp_C",
    "Total_Precip_mm"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

df = assembler.transform(df)

# Step 8: Prepare Target Column
df = df.withColumnRenamed("traffic_count", "label")

# Step 9: Train/Test Split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

print(f"✅ Training Data Count: {train_data.count()}")
print(f"✅ Test Data Count: {test_data.count()}")

# Step 10: Initialize and Train Model
rf = RandomForestRegressor(featuresCol="features", labelCol="label")
model = rf.fit(train_data)

print("✅ Model Training Complete!")

# Step 11: Predictions
predictions = model.transform(test_data)

print("✅ Predictions Sample:")
predictions.select("date", "label", "prediction").show(10)

# Step 12: Evaluation
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

print("📊 Model Evaluation Results:")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# Step 13: Save Model and Predictions
model.save(f"{HDFS_BASE_PATH}traffic_prediction_model")
predictions.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}traffic_predictions.parquet")

print("✅ Model and Predictions Saved as Parquet!")

# Step 14: Optional - Export predictions to CSV for local viewing
# Save predictions to HDFS as CSV (remove vector fields for clarity)
predictions.select("date", "label", "prediction") \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(f"{HDFS_BASE_PATH}traffic_predictions_csv")

print("✅ Predictions also saved as CSV for easy viewing!")

print("🎉 Traffic Prediction Pipeline Completed Successfully!")
