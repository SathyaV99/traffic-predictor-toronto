from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean as _mean, max as _max, min as _min, dayofweek, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

def run_prediction(spark):
    HDFS_BASE_PATH = "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/"

    # Load data
    data = spark.read.parquet(f"{HDFS_BASE_PATH}final_traffic_weather.parquet")

    # Compute statistics for classification buckets
    summary = data.select(
        _min("traffic_count").alias("min_count"),
        _max("traffic_count").alias("max_count"),
        _mean("traffic_count").alias("mean_count")
    ).collect()[0]
    mean_count = summary['mean_count']

    # Create traffic status classes
    data = data.withColumn(
        "traffic_status",
        when(col("traffic_count") <= mean_count * 0.75, "Low")
        .when((col("traffic_count") > mean_count * 0.75) & (col("traffic_count") <= mean_count * 1.25), "Medium")
        .otherwise("High")
    )

    # Feature Engineering
    data = data.withColumnRenamed("Mean Temp (°C)", "Mean_Temp_C") \
               .withColumnRenamed("Total Precip (mm)", "Total_Precip_mm") \
               .withColumn("day_of_week", dayofweek(col("date")))

    is_weekend_udf = udf(lambda dow: 1 if dow in [1, 7] else 0, IntegerType())
    data = data.withColumn("is_weekend", is_weekend_udf(col("day_of_week")))

    # Drop rows with nulls in important features
    data = data.dropna(subset=["Month", "Day", "Mean_Temp_C", "Total_Precip_mm", "traffic_camera", "camera_road"])

    # Index + encode categorical features
    camera_indexer = StringIndexer(inputCol="traffic_camera", outputCol="camera_index", handleInvalid="keep")
    camera_encoder = OneHotEncoder(inputCol="camera_index", outputCol="camera_vec")

    road_indexer = StringIndexer(inputCol="camera_road", outputCol="road_index", handleInvalid="keep")
    road_encoder = OneHotEncoder(inputCol="road_index", outputCol="road_vec")

    # Assemble features
    feature_cols = ["Month", "Day", "day_of_week", "is_weekend", "Mean_Temp_C", "Total_Precip_mm", "camera_vec", "road_vec"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

    label_indexer = StringIndexer(inputCol="traffic_status", outputCol="label", handleInvalid="keep")

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10, seed=42)

    pipeline = Pipeline(stages=[
        camera_indexer, camera_encoder,
        road_indexer, road_encoder,
        assembler, label_indexer, rf
    ])

    # Hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 10]) \
        .addGrid(rf.numTrees, [50, 100]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2
    )

    # Split
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

    # Train
    model = crossval.fit(train_data)

    # Predict
    predictions = model.transform(test_data)

    # Evaluation
    accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
    f1_score = evaluator.evaluate(predictions)

    print(f"🚀 Test Accuracy: {accuracy:.2f}")
    print(f"🎯 Test F1 Score: {f1_score:.2f}")

    # Save
    model.write().overwrite().save(f"{HDFS_BASE_PATH}traffic_status_rf_model")
    predictions.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}traffic_status_predictions.parquet")
    predictions.select("Month", "Day", "traffic_count", "traffic_status", "prediction") \
        .write.mode("overwrite") \
        .option("header", "true") \
        .csv(f"{HDFS_BASE_PATH}traffic_status_predictions_csv")

    print("✅ Classification model and predictions saved!")

# Entry point
if __name__ == "__main__":
    spark = SparkSession.builder.appName("TrafficStatusPrediction").getOrCreate()
    run_prediction(spark)
    spark.stop()
