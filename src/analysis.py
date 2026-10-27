"""
Stage 3 & Stage 4: Exploratory Analysis & Spark SQL Module
Project: Zomato Delivery Operations Analysis Using PySpark

This module implements comprehensive exploratory data analysis, Spark SQL queries,
window functions, and execution plan profiling on the cleaned Zomato delivery dataset.
"""

from typing import Dict, Any, Tuple
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from src.data_loader import get_spark_session


def load_cleaned_data(spark: SparkSession, parquet_path: str = "data/processed/zomato_cleaned.parquet") -> DataFrame:
    """Loads the preprocessed Parquet dataset."""
    if not os.path.exists(parquet_path):
        from src.preprocessing import run_preprocessing_pipeline
        run_preprocessing_pipeline()
    return spark.read.parquet(parquet_path)


# ==============================================================================
# STAGE 3: EXPLORATORY DATA ANALYSIS (Analyses A through I)
# ==============================================================================

def analyze_overall_delivery_performance(df: DataFrame) -> Dict[str, Any]:
    """Analysis A: Overall delivery performance metrics and percentiles."""
    stats = df.select(
        F.count("Time_taken (min)").alias("total_records"),
        F.round(F.mean("Time_taken (min)"), 2).alias("mean_duration"),
        F.round(F.stddev("Time_taken (min)"), 2).alias("stddev_duration"),
        F.min("Time_taken (min)").alias("min_duration"),
        F.max("Time_taken (min)").alias("max_duration")
    ).collect()[0].asDict()

    # Percentiles: 10th, 25th, 50th (median), 75th, 90th, 95th, 99th
    percentiles = df.approxQuantile("Time_taken (min)", [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], 0.001)
    stats["p10"] = percentiles[0]
    stats["p25"] = percentiles[1]
    stats["median_p50"] = percentiles[2]
    stats["p75"] = percentiles[3]
    stats["p90"] = percentiles[4]
    stats["p95"] = percentiles[5]
    stats["p99"] = percentiles[6]
    return stats


def analyze_traffic_impact(df: DataFrame) -> DataFrame:
    """Analysis B: Delivery duration grouped by Road_traffic_density."""
    return (
        df.groupBy("Road_traffic_density")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time"),
            F.round(F.percentile_approx("Time_taken (min)", 0.5), 2).alias("median_delivery_time"),
            F.max("Time_taken (min)").alias("max_delivery_time"),
            F.round(F.stddev("Time_taken (min)"), 2).alias("stddev_delivery_time")
        )
        .orderBy(F.desc("avg_delivery_time"))
    )


def analyze_weather_impact(df: DataFrame) -> DataFrame:
    """Analysis C: Delivery duration grouped by Weather_conditions."""
    return (
        df.groupBy("Weather_conditions")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time"),
            F.round(F.percentile_approx("Time_taken (min)", 0.5), 2).alias("median_delivery_time"),
            F.min("Time_taken (min)").alias("min_delivery_time"),
            F.max("Time_taken (min)").alias("max_delivery_time")
        )
        .orderBy(F.desc("avg_delivery_time"))
    )


def analyze_vehicle_performance(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Analysis D: Delivery duration by Type_of_vehicle and Vehicle_condition."""
    by_type = (
        df.groupBy("Type_of_vehicle")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time"),
            F.round(F.percentile_approx("Time_taken (min)", 0.5), 2).alias("median_delivery_time")
        )
        .orderBy(F.desc("avg_delivery_time"))
    )

    by_condition = (
        df.groupBy("Vehicle_condition")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time"),
            F.round(F.percentile_approx("Time_taken (min)", 0.5), 2).alias("median_delivery_time")
        )
        .orderBy("Vehicle_condition")
    )
    return by_type, by_condition


def analyze_delivery_person_factors(df: DataFrame) -> Dict[str, Any]:
    """Analysis E: Courier age, ratings, and stacked multiple deliveries."""
    # Pearson correlations with Time_taken (min)
    corr_age = df.stat.corr("Delivery_person_Age", "Time_taken (min)")
    corr_rating = df.stat.corr("Delivery_person_Ratings", "Time_taken (min)")
    corr_multiple = df.stat.corr("multiple_deliveries", "Time_taken (min)")

    # Grouped aggregates for rating bands
    rating_bands = (
        df.withColumn(
            "Rating_Band",
            F.when(F.col("Delivery_person_Ratings") >= 4.8, "4.8 - 5.0 (Top)")
             .when(F.col("Delivery_person_Ratings") >= 4.5, "4.5 - 4.7 (Good)")
             .when(F.col("Delivery_person_Ratings") >= 4.0, "4.0 - 4.4 (Average)")
             .otherwise("< 4.0 (Low)")
        )
        .groupBy("Rating_Band")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time")
        )
        .orderBy(F.desc("avg_delivery_time"))
    )

    # Multi-delivery impact
    multi_deliveries = (
        df.groupBy("multiple_deliveries")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time"),
            F.round(F.percentile_approx("Time_taken (min)", 0.5), 2).alias("median_delivery_time")
        )
        .orderBy("multiple_deliveries")
    )

    return {
        "correlations": {
            "Age_vs_Time": round(corr_age, 4),
            "Rating_vs_Time": round(corr_rating, 4),
            "MultipleDeliveries_vs_Time": round(corr_multiple, 4)
        },
        "rating_bands": rating_bands,
        "multi_deliveries": multi_deliveries
    }


def analyze_order_type_impact(df: DataFrame) -> DataFrame:
    """Analysis F: Delivery duration grouped by Type_of_order."""
    return (
        df.groupBy("Type_of_order")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time"),
            F.round(F.percentile_approx("Time_taken (min)", 0.5), 2).alias("median_delivery_time"),
            F.round(F.stddev("Time_taken (min)"), 2).alias("stddev_delivery_time")
        )
        .orderBy(F.desc("avg_delivery_time"))
    )


def analyze_festival_and_city(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Analysis G: Delivery duration by Festival and City category."""
    by_festival = (
        df.groupBy("Festival")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time"),
            F.round(F.percentile_approx("Time_taken (min)", 0.5), 2).alias("median_delivery_time")
        )
        .orderBy(F.desc("avg_delivery_time"))
    )

    by_city = (
        df.groupBy("City")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time"),
            F.round(F.percentile_approx("Time_taken (min)", 0.5), 2).alias("median_delivery_time")
        )
        .orderBy(F.desc("avg_delivery_time"))
    )
    return by_festival, by_city


def analyze_temporal_patterns(df: DataFrame) -> Dict[str, DataFrame]:
    """Analysis H: Date and Hour trends."""
    by_day_of_week = (
        df.groupBy("Day_of_Week", "Is_Weekend")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time")
        )
        .orderBy("Day_of_Week")
    )

    by_order_hour = (
        df.filter(F.col("Order_Hour").isNotNull())
        .groupBy("Order_Hour")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time")
        )
        .orderBy("Order_Hour")
    )

    by_month = (
        df.groupBy("Order_Month")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time")
        )
        .orderBy("Order_Month")
    )
    return {
        "by_day_of_week": by_day_of_week,
        "by_order_hour": by_order_hour,
        "by_month": by_month
    }


def analyze_geographic_distance(df: DataFrame) -> Dict[str, Any]:
    """Analysis I: Straight-line distance correlation and binned delivery times."""
    valid_dist = df.filter(F.col("Approx_Distance_Km").isNotNull())
    corr_dist = valid_dist.stat.corr("Approx_Distance_Km", "Time_taken (min)")

    binned_dist = (
        valid_dist.withColumn(
            "Distance_Bin",
            F.when(F.col("Approx_Distance_Km") < 5.0, "< 5 km (Short)")
             .when(F.col("Approx_Distance_Km") < 10.0, "5 - 10 km (Medium)")
             .when(F.col("Approx_Distance_Km") < 15.0, "10 - 15 km (Long)")
             .otherwise("15 - 21 km (Far)")
        )
        .groupBy("Distance_Bin")
        .agg(
            F.count("ID").alias("delivery_count"),
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_delivery_time"),
            F.round(F.percentile_approx("Time_taken (min)", 0.5), 2).alias("median_delivery_time")
        )
        .orderBy(F.desc("avg_delivery_time"))
    )

    return {
        "correlation": round(corr_dist, 4),
        "binned_dist": binned_dist
    }


# ==============================================================================
# STAGE 4: SPARK SQL & ADVANCED PYSPARK
# ==============================================================================

def run_spark_sql_queries(spark: SparkSession, df: DataFrame) -> Dict[str, DataFrame]:
    """
    Executes advanced Spark SQL queries using registered temporary views.
    Demonstrates filtering, joins, aggregations, and subqueries.
    """
    df.createOrReplaceTempView("zomato_deliveries")

    # Query 1: Operational Bottleneck - Traffic Impact by Urban Hierarchy
    query1 = """
        SELECT 
            City,
            Road_traffic_density,
            COUNT(ID) AS order_volume,
            ROUND(AVG(`Time_taken (min)`), 2) AS avg_duration_min,
            ROUND(AVG(Approx_Distance_Km), 2) AS avg_distance_km
        FROM zomato_deliveries
        WHERE City != 'Unknown'
        GROUP BY City, Road_traffic_density
        ORDER BY City, avg_duration_min DESC
    """
    res1 = spark.sql(query1)

    # Query 2: Adverse Weather & Vehicle Condition Matrix
    query2 = """
        SELECT 
            Weather_conditions,
            Vehicle_condition,
            COUNT(ID) AS total_orders,
            ROUND(AVG(`Time_taken (min)`), 2) AS avg_duration_min,
            ROUND(AVG(Order_to_Pickup_Minutes), 2) AS avg_kitchen_prep_min
        FROM zomato_deliveries
        WHERE Weather_conditions IN ('Fog', 'Stormy', 'Sandstorms', 'Sunny')
        GROUP BY Weather_conditions, Vehicle_condition
        HAVING COUNT(ID) > 500
        ORDER BY avg_duration_min DESC
    """
    res2 = spark.sql(query2)

    # Query 3: Courier Delivery Load Stacking during Festivals
    query3 = """
        SELECT 
            Festival,
            multiple_deliveries,
            COUNT(ID) AS volume,
            ROUND(AVG(`Time_taken (min)`), 2) AS avg_time,
            ROUND(MIN(`Time_taken (min)`), 2) AS min_time,
            ROUND(MAX(`Time_taken (min)`), 2) AS max_time
        FROM zomato_deliveries
        WHERE Festival != 'Unknown'
        GROUP BY Festival, multiple_deliveries
        ORDER BY Festival DESC, multiple_deliveries ASC
    """
    res3 = spark.sql(query3)

    return {
        "traffic_by_city": res1,
        "weather_vehicle_matrix": res2,
        "festival_stacking": res3
    }


def demonstrate_window_functions(df: DataFrame) -> Dict[str, DataFrame]:
    """
    Demonstrates advanced PySpark Window functions:
    1. DENSE_RANK(): Rank traffic densities within each city category by average delivery duration.
    2. ROLLING AVERAGE: Calculate 7-day rolling average delivery time over ordered operational dates.
    """
    # Window 1: Dense Rank of Traffic within City
    traffic_city_summary = (
        df.filter(F.col("City") != "Unknown")
        .groupBy("City", "Road_traffic_density")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_duration"))
    )

    city_window = Window.partitionBy("City").orderBy(F.desc("avg_duration"))
    window_ranking = traffic_city_summary.withColumn("traffic_severity_rank", F.dense_rank().over(city_window))

    # Window 2: Rolling Average across daily time series
    daily_delivery = (
        df.groupBy("Order_Date")
        .agg(
            F.count("ID").alias("daily_orders"),
            F.round(F.mean("Time_taken (min)"), 2).alias("daily_avg_time")
        )
        .orderBy("Order_Date")
    )

    # 7-day rolling window (3 preceding days, current day, 3 following days, or 6 preceding)
    rolling_window = Window.orderBy("Order_Date").rowsBetween(-6, 0)
    window_rolling = daily_delivery.withColumn(
        "7_day_rolling_avg_time",
        F.round(F.avg("daily_avg_time").over(rolling_window), 2)
    )

    return {
        "traffic_ranking_window": window_ranking,
        "rolling_daily_avg_window": window_rolling
    }


def profile_execution_and_caching(df: DataFrame) -> None:
    """
    Demonstrates Catalyst execution plan profiling using explain(),
    and measures the impact of DataFrame caching.
    """
    print("\n--- [Spark Plan Profiling: explain()] ---")
    filtered_agg = df.filter(F.col("Time_taken (min)") > 25).groupBy("City").count()
    filtered_agg.explain(mode="simple")

    print("\n--- [Caching Performance Experiment] ---")
    import time

    # Uncached evaluation
    t0 = time.time()
    count_1 = df.filter(F.col("Time_taken (min)") > 30).count()
    uncached_time = time.time() - t0

    # Cache the DataFrame in memory
    df.cache()
    df.count()  # Materialize cache into memory

    # Cached evaluation
    t0 = time.time()
    count_2 = df.filter(F.col("Time_taken (min)") > 30).count()
    cached_time = time.time() - t0

    print(f" Uncached Count Execution Time: {uncached_time:.4f} seconds")
    print(f" Cached Count Execution Time:   {cached_time:.4f} seconds")
    print(f" Cache Speedup Ratio:          {uncached_time / max(cached_time, 0.0001):.2f}x")
    df.unpersist()


def run_full_analysis():
    """Runs all Stage 3 and Stage 4 analytical routines and prints results."""
    print("=" * 80)
    print("STAGE 3 & 4: EXPLORATORY OPERATIONS ANALYSIS & SPARK SQL")
    print("=" * 80)

    spark = get_spark_session("Stage3_Analysis")
    df = load_cleaned_data(spark)

    print("\n>>> Analysis A: Overall Delivery Performance")
    stats_a = analyze_overall_delivery_performance(df)
    for k, v in stats_a.items():
        print(f"  {k:<20}: {v}")

    print("\n>>> Analysis B: Road Traffic Density Impact")
    analyze_traffic_impact(df).show(truncate=False)

    print("\n>>> Analysis C: Weather Conditions Impact")
    analyze_weather_impact(df).show(truncate=False)

    print("\n>>> Analysis D: Vehicle Type & Condition Impact")
    by_type, by_cond = analyze_vehicle_performance(df)
    print("--- By Vehicle Type ---")
    by_type.show(truncate=False)
    print("--- By Vehicle Condition ---")
    by_cond.show(truncate=False)

    print("\n>>> Analysis E: Delivery Partner Attributes (Correlations)")
    e_res = analyze_delivery_person_factors(df)
    for k, v in e_res["correlations"].items():
        print(f"  {k:<30}: {v}")
    print("\n--- Multi-Delivery Stacking ---")
    e_res["multi_deliveries"].show(truncate=False)

    print("\n>>> Analysis F: Type of Order Impact")
    analyze_order_type_impact(df).show(truncate=False)

    print("\n>>> Analysis G: Festival & City Hierarchy")
    fest, city = analyze_festival_and_city(df)
    print("--- Festival Impact ---")
    fest.show(truncate=False)
    print("--- City Hierarchy ---")
    city.show(truncate=False)

    print("\n>>> Analysis I: Straight-Line Distance vs. Delivery Duration")
    dist_res = analyze_geographic_distance(df)
    print(f"  Pearson Correlation (Distance vs Time): {dist_res['correlation']}")
    dist_res["binned_dist"].show(truncate=False)

    print("\n>>> Stage 4: Spark SQL Queries")
    sql_res = run_spark_sql_queries(spark, df)
    print("--- Query 1: Traffic by City Hierarchy ---")
    sql_res["traffic_by_city"].show(10, truncate=False)

    print("\n>>> Stage 4: Window Functions")
    wf_res = demonstrate_window_functions(df)
    print("--- Window Function 1: Dense Rank of Traffic per City ---")
    wf_res["traffic_ranking_window"].show(15, truncate=False)

    print("--- Window Function 2: 7-Day Rolling Average Daily Delivery Time ---")
    wf_res["rolling_daily_avg_window"].show(10, truncate=False)

    profile_execution_and_caching(df)

    print("\n" + "=" * 80)
    print("STAGE 3 & 4 ANALYTICS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_analysis()
