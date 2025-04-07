from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, dayofweek, dayofyear, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import (
    RandomForestClassifier, DecisionTreeClassifier,
    LogisticRegression, GBTClassifier, NaiveBayes
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

def add_new_features(df):
    season_udf = udf(lambda m: "Winter" if m in [12, 1, 2]
                     else ("Spring" if m in [3, 4, 5]
                           else ("Summer" if m in [6, 7, 8]
                                 else "Fall")), StringType())
    df = df.withColumn("season", season_udf(col("Month")))
    df = df.withColumn("day_of_year", dayofyear(col("date")))
    return df

def evaluate_model(predictions, name):
    evaluator = MulticlassClassificationEvaluator(labelCol="traffic_label", predictionCol="prediction")
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)

    print(f"\n✅ {name} Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    rdd = predictions.select("prediction", "traffic_label").rdd.map(lambda row: (row["prediction"], row["traffic_label"]))
    metrics = MulticlassMetrics(rdd)
    print("\n📊 Confusion Matrix:")
    print(metrics.confusionMatrix().toArray())
    return accuracy, f1

def run_comparison_pipeline():
    spark = SparkSession.builder.appName("TrafficModelComparison").getOrCreate()

    print("📦 Loading data...")
    df = spark.read.option("header", True).option("inferSchema", True).csv(
        "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/combined.csv"
    )

    df = df.withColumnRenamed("Max Temp (°C)", "Max_Temp_C") \
           .withColumnRenamed("Min Temp (°C)", "Min_Temp_C") \
           .withColumnRenamed("Mean Temp (°C)", "Mean_Temp_C") \
           .withColumnRenamed("Total Precip (mm)", "Total_Precip_mm")

    df = df.dropna()
    df = df.withColumn("day_of_week", dayofweek(col("date"))) \
           .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0)) \
           .withColumn("temp_range", col("Max_Temp_C") - col("Min_Temp_C")) \
           .withColumnRenamed("traffic_status", "traffic_label")

    df = add_new_features(df)

    # Filter out invalid values
    df = df.filter(col("Mean_Temp_C").isNotNull() & (col("Mean_Temp_C") >= -50)) \
           .filter(col("temp_range").isNotNull() & (col("temp_range") >= 0)) \
           .filter(col("Total_Precip_mm").isNotNull() & (col("Total_Precip_mm") >= 0))

    season_indexer = StringIndexer(inputCol="season", outputCol="season_idx", handleInvalid="keep")
    season_encoder = OneHotEncoder(inputCol="season_idx", outputCol="season_vec")

    df.groupBy("traffic_label").count().show()

    all_features = ["Month", "Day", "day_of_week", "is_weekend", "Mean_Temp_C",
                    "Total_Precip_mm", "temp_range", "day_of_year", "season_vec"]
    assembler = VectorAssembler(inputCols=all_features, outputCol="features")

    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    classifiers = {
        "RandomForest": RandomForestClassifier(labelCol="traffic_label", featuresCol="features", numTrees=100),
        "DecisionTree": DecisionTreeClassifier(labelCol="traffic_label", featuresCol="features"),
        "LogisticRegression": LogisticRegression(labelCol="traffic_label", featuresCol="features", maxIter=10),
        "GBTClassifier": GBTClassifier(labelCol="traffic_label", featuresCol="features", maxIter=20)
    }

    nb_features = ["Month", "Day", "day_of_week", "is_weekend", "Total_Precip_mm", "day_of_year", "season_vec"]
    nb_assembler = VectorAssembler(inputCols=nb_features, outputCol="nb_features")
    nb_model = NaiveBayes(labelCol="traffic_label", featuresCol="nb_features", modelType="multinomial")
    classifiers["NaiveBayes"] = Pipeline(stages=[season_indexer, season_encoder, nb_assembler, nb_model])

    best_score = 0
    best_predictions = None

    for name, clf in classifiers.items():
        print(f"\n🔍 Training {name}")
        try:
            if name == "NaiveBayes":
                model = clf.fit(train_data)
                predictions = model.transform(test_data)
            else:
                pipeline = Pipeline(stages=[season_indexer, season_encoder, assembler, clf])
                model = pipeline.fit(train_data)
                predictions = model.transform(test_data)

            predictions = predictions.withColumn("prediction", col("prediction").cast("double")) \
                                     .withColumn("traffic_label", col("traffic_label").cast("double"))

            _, f1 = evaluate_model(predictions, name)
            if f1 > best_score:
                best_score = f1
                best_predictions = predictions

        except Exception as e:
            print(f"❌ {name} failed: {e}")

    if best_predictions:
        best_predictions.select("date", "traffic_label", "prediction") \
            .write.mode("overwrite").option("header", True) \
            .csv("hdfs://localhost:9000/user/hdoop/toronto_traffic/output/final_predictions_csv")
        print("✅ Best model predictions saved!")

    spark.stop()

if __name__ == "__main__":
    run_comparison_pipeline()
