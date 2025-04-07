from pyspark.sql.functions import col, expr, to_date

def run_transformation(spark):
    HDFS_BASE_PATH = "hdfs://localhost:9000/user/hdoop/toronto_traffic/input/"

    # Define file path
    file_path = f"{HDFS_BASE_PATH}tf-ft-eng.csv"

    # Read CSV (attempt with common delimiters)
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

    # If all data is in one column, try to split manually
    if len(df.columns) == 1:
        print("CSV appears malformed. Attempting to fix...")
        single_col = df.columns[0]
        df = df.withColumn("split_data", expr(f"split({single_col}, ',')"))
        header = df.first()["split_data"]
        
        # Recreate DataFrame with proper schema
        df = df.rdd.map(lambda row: row["split_data"]).toDF(header)

    # Identify columns
    date_columns = [col_name for col_name in df.columns if col_name.startswith("x20")]
    id_columns = [col_name for col_name in df.columns if not col_name.startswith("x20")]

    print(f"ID columns: {id_columns}")
    print(f"Number of date columns: {len(date_columns)}")

    # Optional: repartition to avoid memory pressure
    df = df.repartition(8)

    # Cast all date columns to string
    df = df.selectExpr(*id_columns, *[f"cast({col_name} as string) as {col_name}" for col_name in date_columns])

    # Melt using stack
    expr_str = ", ".join([f"'{c}', {c}" for c in date_columns])
    df_long = df.selectExpr(*id_columns, f"stack({len(date_columns)}, {expr_str}) as (date, traffic_count)")

    # Clean 'date' column
    df_long = df_long.withColumn("date", expr("replace(date, 'x', '')"))
    df_long = df_long.withColumn("date", to_date(col("date"), "yyyy_MM_dd"))

    # Filter out null counts
    df_long = df_long.filter(col("traffic_count").isNotNull())

    # Sort
    if 'traffic_camera' in df_long.columns:
        df_long = df_long.orderBy("traffic_camera", "date")
    else:
        sort_columns = [col for col in id_columns if col in df_long.columns]
        df_long = df_long.orderBy(*sort_columns, "date")

    # Show sample
    df_long.show(10)

    # Save transformed data
    output_path = f"{HDFS_BASE_PATH}transformed_traffic_data"
    df_long.write.mode("overwrite").option("header", "true").csv(output_path)

    print(f"✅ Transformed data saved to {output_path}")
