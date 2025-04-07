from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, dayofweek
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline

def run_final_pipeline():
    spark = SparkSession.builder.appName("FinalTrafficModel").getOrCreate()

    # STEP 2: Load CSV from HDFS
    print("📦 Loading data from HDFS...")
    df = spark.read.option("header", True).option("inferSchema", True).csv(
        "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/combined.csv"
    )

    # STEP 3: Rename problematic columns to avoid special characters
    df = df.withColumnRenamed("Max Temp (°C)", "Max_Temp_C") \
           .withColumnRenamed("Min Temp (°C)", "Min_Temp_C") \
           .withColumnRenamed("Mean Temp (°C)", "Mean_Temp_C") \
           .withColumnRenamed("Total Precip (mm)", "Total_Precip_mm")

    # STEP 4: Drop any remaining nulls
    df = df.dropna()

    # STEP 5: Feature Engineering
    df = df.withColumn("day_of_week", dayofweek(col("date")))
    df = df.withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
    df = df.withColumn("temp_range", col("Max_Temp_C") - col("Min_Temp_C"))
    
    # STEP 6: Use the existing binary label
    # Assuming combined.csv already has a balanced traffic_status column with 0 and 1, just rename it.
    df = df.withColumnRenamed("traffic_status", "traffic_label")

    # STEP 7: Show Class Distribution
    print("\n🚦 Class Distribution:")
    df.groupBy("traffic_label").count().show()

    # (Skip undersampling if already balanced)

    # STEP 8: Feature Encoding & Assembling
    temp_range_indexer = StringIndexer(inputCol="temp_range", outputCol="temp_range_idx", handleInvalid="keep")
    temp_range_encoder = OneHotEncoder(inputCol="temp_range_idx", outputCol="temp_range_vec")
    
    # Assemble features using the renamed columns and engineered features.
    feature_cols = ["Month", "Day", "day_of_week", "is_weekend", "Mean_Temp_C", "Total_Precip_mm", "temp_range_vec"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # STEP 9: Define the Random Forest Classifier
    rf = RandomForestClassifier(featuresCol="features", labelCol="traffic_label", seed=42, numTrees=100)

    # Build Pipeline
    pipeline = Pipeline(stages=[temp_range_indexer, temp_range_encoder, assembler, rf])

    # STEP 10: Split Data into Train and Test sets
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # STEP 11: Train Model
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)

    # Force cast the label and prediction columns to double to avoid type issues
    predictions = predictions.withColumn("traffic_label", col("traffic_label").cast("double")) \
                             .withColumn("prediction", col("prediction").cast("double"))

    # STEP 12: Evaluate Model
    evaluator = MulticlassClassificationEvaluator(labelCol="traffic_label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print(f"🎯 F1 Score: {f1:.4f}")
    print(f"📌 Precision: {precision:.4f}")
    print(f"🔄 Recall: {recall:.4f}")

    # STEP 13: Compute Confusion Matrix
    rdd = predictions.select("prediction", "traffic_label").rdd.map(tuple)
    metrics = MulticlassMetrics(rdd)
    print("\n📊 Confusion Matrix:")
    print(metrics.confusionMatrix().toArray())

    # STEP 14: Save Outputs to HDFS
    predictions.select("date", "traffic_label", "prediction").write.mode("overwrite").option("header", True).csv(
        "hdfs://localhost:9000/user/hdoop/toronto_traffic/output/final_predictions_csv"
    )
    model.write().overwrite().save("hdfs://localhost:9000/user/hdoop/toronto_traffic/output/final_rf_model")

    print("✅ Final pipeline completed and saved!")
    spark.stop()

if __name__ == "__main__":
    run_final_pipeline()
