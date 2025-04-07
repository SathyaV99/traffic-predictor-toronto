from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, dayofweek, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

def run_optimized_prediction_pipeline(spark):
    HDFS_BASE_PATH = "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/"
    df = spark.read.csv(f"{HDFS_BASE_PATH}final_traffic_weather.csv", header=True, inferSchema=True)

    # Compute traffic_count mean
    mean_count = df.selectExpr("avg(traffic_count) as mean").first()["mean"]

    # Create traffic_status (classification label)
    df = df.withColumn(
        "traffic_status",
        when(col("traffic_count") <= mean_count * 0.75, "Low")
        .when((col("traffic_count") > mean_count * 0.75) & (col("traffic_count") <= mean_count * 1.25), "Medium")
        .otherwise("High")
    )

    # Rename relevant columns
    df = df.withColumnRenamed("Mean Temp (°C)", "Mean_Temp_C") \
           .withColumnRenamed("Max Temp (°C)", "Max_Temp_C") \
           .withColumnRenamed("Min Temp (°C)", "Min_Temp_C") \
           .withColumnRenamed("Total Precip (mm)", "Total_Precip_mm")

    # Feature Engineering
    df = df.withColumn("day_of_week", dayofweek(col("date")))

    is_weekend_udf = udf(lambda dow: 1 if dow in [1, 7] else 0, IntegerType())  # UDF for weekend
    df = df.withColumn("is_weekend", is_weekend_udf(col("day_of_week")))

    # Temperature range and classification
    df = df.withColumn("temp_range", col("Max_Temp_C") - col("Min_Temp_C"))
    df = df.withColumn(
        "temp_range_cat",
        when(col("temp_range") < 5, "Low")
        .when(col("temp_range") < 10, "Medium")
        .otherwise("High")
    )

    # Drop unnecessary columns
    drop_cols = ["traffic_camera", "camera_road", "WKT", "Longitude (x)", "Latitude (y)"]
    for colname in drop_cols:
        if colname in df.columns:
            df = df.drop(colname)

    # Drop nulls in essential columns
    df = df.dropna(subset=["Month", "Day", "Mean_Temp_C", "Total_Precip_mm", "Max_Temp_C", "Min_Temp_C"])

    # Categorical encoding for temp_range_cat
    temp_range_indexer = StringIndexer(inputCol="temp_range_cat", outputCol="temp_range_index", handleInvalid="keep")
    temp_range_encoder = OneHotEncoder(inputCol="temp_range_index", outputCol="temp_range_vec")

    # Assemble final features
    feature_cols = ["Month", "Day", "day_of_week", "is_weekend", "Mean_Temp_C", "Total_Precip_mm", "temp_range_vec"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

    # Label encoding
    label_indexer = StringIndexer(inputCol="traffic_status", outputCol="label", handleInvalid="keep")

    # Classifier
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10, seed=42)

    # Pipeline
    pipeline = Pipeline(stages=[
        temp_range_indexer, temp_range_encoder,
        assembler,
        label_indexer,
        rf
    ])

    # Grid search
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
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

    # Split and train
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    model = crossval.fit(train_data)

    # Predict
    predictions = model.transform(test_data)
    predictions.select("date", "traffic_count", "traffic_status", "prediction").show(10, False)

    # Save model and outputs
    model.write().overwrite().save(f"{HDFS_BASE_PATH}traffic_optimized_model")
    predictions.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}traffic_optimized_predictions.parquet")

    # Save simplified CSV (exclude vector fields)
    predictions.select("date", "traffic_count", "traffic_status", "prediction") \
        .write.mode("overwrite").option("header", True) \
        .csv("traffic_optimized_predictions_csv")

    print("✅ Optimized prediction model trained and outputs saved.")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("OptimizedTrafficPrediction").getOrCreate()
    run_optimized_prediction_pipeline(spark)
    spark.stop()
