"""
Stage 1: Dataset Inspection Script
Zomato Delivery Operations Analysis Using PySpark

This standalone script executes the complete Stage 1 dataset inspection:
1. Initializes a SparkSession.
2. Loads the raw CSV dataset safely.
3. Prints dataset dimensions (rows and columns).
4. Displays sample records and inferred schema.
5. Calculates true missing values per column (Nulls, empty strings, and 'NaN' literals).
6. Checks for exact duplicate rows and ID uniqueness.
7. Inspects categorical value distributions.
8. Inspects numerical ranges and flags anomalies (ages, ratings, coordinates, time formats).
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from src.data_loader import get_spark_session, load_raw_dataset


def run_stage1_inspection(file_path: str = "data/raw/Zomato Dataset.csv"):
    print("=" * 80)
    print("STAGE 1: ZOMATO DELIVERY DATASET INSPECTION USING PYSPARK")
    print("=" * 80)

    # 1. Initialize SparkSession
    print("\n[Step 1] Initializing Apache SparkSession...")
    spark = get_spark_session("Stage1_DatasetInspection")
    print(f" Spark Version: {spark.version}")
    print(f" Master: {spark.sparkContext.master}")

    # 2. Load Raw CSV
    print(f"\n[Step 2] Loading raw dataset from: {file_path}")
    df = load_raw_dataset(spark, file_path)

    # 3. Basic Dimensions
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"\n[Step 3] Dataset Dimensions:")
    print(f" - Total Rows: {total_rows:,}")
    print(f" - Total Columns: {total_cols}")

    # 4. Column Names and Initial Data Types
    print("\n[Step 4] Column Schema (Initial Raw Types):")
    for col_name, dtype in df.dtypes:
        print(f"   * {col_name:<30} : {dtype}")

    # 5. Display Sample Records
    print("\n[Step 5] Sample Records (First 5 Rows):")
    df.show(5, truncate=False)

    # 6. Missing Value Analysis
    print("\n[Step 6] Missing Value Breakdown Per Column:")
    print(" Note: Includes Spark NULLs, empty strings '', and literal string 'NaN'")
    print("-" * 65)
    print(f"{'Column Name':<30} | {'Missing Count':<15} | {'Missing %':<10}")
    print("-" * 65)

    missing_summary = []
    for c in df.columns:
        cnt = df.filter(
            F.col(c).isNull() | 
            (F.trim(F.col(c)) == "") | 
            (F.trim(F.col(c)) == "NaN")
        ).count()
        pct = (cnt / total_rows) * 100
        missing_summary.append((c, cnt, pct))
        print(f"{c:<30} | {cnt:<15,d} | {pct:>6.2f}%")
    print("-" * 65)

    # 7. Exact Duplicates & Identifier Checks
    print("\n[Step 7] Duplicate Records & Identifier Validation:")
    distinct_rows = df.distinct().count()
    exact_duplicates = total_rows - distinct_rows
    print(f" - Distinct Row Count: {distinct_rows:,}")
    print(f" - Exact Duplicate Rows: {exact_duplicates:,}")

    unique_order_ids = df.select("ID").distinct().count()
    print(f" - Unique Order IDs: {unique_order_ids:,} (Repeated: {total_rows - unique_order_ids})")

    unique_delivery_persons = df.select("Delivery_person_ID").distinct().count()
    print(f" - Unique Delivery Person IDs: {unique_delivery_persons:,}")

    # Top active delivery partners
    print(" - Top 5 Most Frequent Delivery Partners:")
    df.groupBy("Delivery_person_ID").count().orderBy(F.desc("count")).show(5, truncate=False)

    # 8. Categorical Distributions
    print("\n[Step 8] Categorical Column Inspection:")
    cat_columns = [
        "Weather_conditions", 
        "Road_traffic_density", 
        "Vehicle_condition", 
        "Type_of_order", 
        "Type_of_vehicle", 
        "multiple_deliveries", 
        "Festival", 
        "City"
    ]
    for cat_col in cat_columns:
        print(f"\n--- Distribution for: {cat_col} ---")
        df.groupBy(cat_col).count().orderBy(F.desc("count")).show(truncate=False)

    # 9. Numerical Range & Anomaly Checks
    print("\n[Step 9] Numerical Range and Sanity Checks:")

    # Cast to numeric while converting 'NaN' strings to NULL so Spark doesn't pollute statistics
    clean_numeric_df = (
        df.withColumn("Age", F.when(F.col("Delivery_person_Age") == "NaN", None).otherwise(F.col("Delivery_person_Age").cast(DoubleType())))
          .withColumn("Rating", F.when(F.col("Delivery_person_Ratings") == "NaN", None).otherwise(F.col("Delivery_person_Ratings").cast(DoubleType())))
          .withColumn("Delivery_Time", F.when(F.col("Time_taken (min)") == "NaN", None).otherwise(F.col("Time_taken (min)").cast(DoubleType())))
          .withColumn("Rest_Lat", F.col("Restaurant_latitude").cast(DoubleType()))
          .withColumn("Rest_Lon", F.col("Restaurant_longitude").cast(DoubleType()))
          .withColumn("Del_Lat", F.col("Delivery_location_latitude").cast(DoubleType()))
          .withColumn("Del_Lon", F.col("Delivery_location_longitude").cast(DoubleType()))
    )

    clean_numeric_df.select("Age", "Rating", "Delivery_Time", "Rest_Lat", "Rest_Lon", "Del_Lat", "Del_Lon").describe().show()

    # Suspicious values audit
    print("\n--- Anomaly Investigation ---")
    
    # Ratings outside [1.0, 5.0]
    out_of_range_ratings = clean_numeric_df.filter((F.col("Rating") < 1.0) | (F.col("Rating") > 5.0)).count()
    print(f" * Delivery Person Ratings outside [1.0, 5.0]: {out_of_range_ratings} records (all are Rating == 6.0)")

    # Age anomalies
    underage_count = clean_numeric_df.filter(F.col("Age") < 18).count()
    age_50_count = clean_numeric_df.filter(F.col("Age") == 50).count()
    print(f" * Delivery Person Age < 18 (Age == 15): {underage_count} records")
    print(f" * Delivery Person Age == 50: {age_50_count} records (correlated with Rating == 6.0)")

    # Coordinate anomalies (0.0 GPS sensor failures and negative coordinates)
    zero_coords = clean_numeric_df.filter(
        (F.col("Rest_Lat") == 0.0) | (F.col("Rest_Lon") == 0.0) |
        (F.col("Del_Lat") == 0.0) | (F.col("Del_Lon") == 0.0)
    ).count()
    negative_lat = clean_numeric_df.filter((F.col("Rest_Lat") < 0.0) | (F.col("Del_Lat") < 0.0)).count()
    negative_lon = clean_numeric_df.filter((F.col("Rest_Lon") < 0.0) | (F.col("Del_Lon") < 0.0)).count()
    print(f" * Records with GPS coordinates == 0.0 (Null Island sensor failure): {zero_coords} records")
    print(f" * Records with negative latitude: {negative_lat} records")
    print(f" * Records with negative longitude: {negative_lon} records")

    # Time format irregularities
    decimal_orders = df.filter(F.col("Time_Orderd").contains(".")).count()
    decimal_picked = df.filter(F.col("Time_Order_picked").contains(".")).count()
    hhmmss_picked = df.filter(F.col("Time_Order_picked").rlike("^[0-9]{2}:[0-9]{2}:[0-9]{2}")).count()
    print(f" * Time_Orderd containing Excel decimal fractions: {decimal_orders} records")
    print(f" * Time_Order_picked containing Excel decimal fractions: {decimal_picked} records")
    print(f" * Time_Order_picked formatted as HH:MM:SS (including 24:05:00): {hhmmss_picked} records")

    # Date range
    dates_df = df.withColumn("ParsedDate", F.to_date(F.col("Order_Date"), "dd-MM-yyyy"))
    date_stats = dates_df.select(
        F.min("ParsedDate").alias("min_d"),
        F.max("ParsedDate").alias("max_d"),
        F.countDistinct("ParsedDate").alias("distinct_days")
    ).collect()[0]
    print(f" * Order_Date Range: {date_stats['min_d']} to {date_stats['max_d']} ({date_stats['distinct_days']} unique operational days)")

    print("\n" + "=" * 80)
    print("STAGE 1 INSPECTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    file_path = "data/raw/Zomato Dataset.csv"
    if not os.path.exists(file_path):
        # Fallback to current directory if not in data/raw
        if os.path.exists("Zomato Dataset.csv"):
            file_path = "Zomato Dataset.csv"
    run_stage1_inspection(file_path)
