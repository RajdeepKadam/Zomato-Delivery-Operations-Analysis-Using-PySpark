"""
Stage 6: Analytical Visualizations Module
Project: Zomato Delivery Operations Analysis Using PySpark

This module generates all 11 required analytical visualizations from aggregated PySpark
queries and saves high-resolution figures to the `outputs/charts/` directory.

Design Rule: Only small aggregated DataFrames are converted to Pandas for plotting;
the full distributed dataset is never collected to the driver.
"""

from typing import Optional
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless terminal & scripts
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from src.data_loader import get_spark_session


# Aesthetic Color Palette
PRIMARY_COLOR = "#E23744"   # Zomato Red
SECONDARY_COLOR = "#2D3436" # Charcoal Black
ACCENT_COLOR = "#0984E3"    # Clean Blue
MUTED_GRAY = "#DFE6E9"
SUCCESS_COLOR = "#00B894"


def setup_plot_style():
    """Sets clean styling for Matplotlib charts."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["figure.dpi"] = 150


def generate_all_visualizations(df: DataFrame, output_dir: str = "outputs/charts"):
    """
    Generates all 11 required publication-quality charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()
    print("=" * 80)
    print("STAGE 6: GENERATING ANALYTICAL VISUALIZATIONS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Chart 1: Distribution of Delivery Durations
    # --------------------------------------------------------------------------
    print(" [1/11] Generating: Distribution of Delivery Durations...")
    # Compute frequency histogram in PySpark
    duration_hist = (
        df.groupBy("Time_taken (min)")
        .count()
        .orderBy("Time_taken (min)")
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        duration_hist["Time_taken (min)"], 
        duration_hist["count"], 
        color=PRIMARY_COLOR, 
        edgecolor="white", 
        alpha=0.85
    )
    mean_val = df.select(F.mean("Time_taken (min)")).collect()[0][0]
    median_val = df.approxQuantile("Time_taken (min)", [0.5], 0.001)[0]
    ax.axvline(mean_val, color="blue", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.1f} min")
    ax.axvline(median_val, color="black", linestyle="-", linewidth=1.8, label=f"Median: {median_val:.0f} min")

    ax.set_title("Distribution of Food Delivery Duration (N = 45,584)")
    ax.set_xlabel("Delivery Time (Minutes)")
    ax.set_ylabel("Number of Deliveries")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_delivery_duration_distribution.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 2: Average Delivery Time by Traffic Density
    # --------------------------------------------------------------------------
    print(" [2/11] Generating: Average Delivery Time by Traffic Density...")
    traffic_pdf = (
        df.groupBy("Road_traffic_density")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"))
        .orderBy(F.desc("avg_time"))
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(traffic_pdf["Road_traffic_density"], traffic_pdf["avg_time"], color=["#D63031", "#E17055", "#FDCB6E", "#00B894", "#B2BEC3"])
    ax.set_title("Average Delivery Duration by Road Traffic Density")
    ax.set_xlabel("Traffic Density Category")
    ax.set_ylabel("Average Delivery Time (Minutes)")
    ax.set_ylim(0, max(traffic_pdf["avg_time"]) * 1.15)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}m", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_avg_time_by_traffic_density.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 3: Average Delivery Time by Weather Conditions
    # --------------------------------------------------------------------------
    print(" [3/11] Generating: Average Delivery Time by Weather Conditions...")
    weather_pdf = (
        df.groupBy("Weather_conditions")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"))
        .orderBy(F.desc("avg_time"))
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(weather_pdf["Weather_conditions"], weather_pdf["avg_time"], color="#6C5CE7")
    ax.set_title("Average Delivery Duration Across Weather Conditions")
    ax.set_xlabel("Average Delivery Time (Minutes)")
    ax.set_ylabel("Weather Condition")
    ax.set_xlim(0, max(weather_pdf["avg_time"]) * 1.15)
    for b in bars:
        ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f} min", va="center", weight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_avg_time_by_weather.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 4: Average Delivery Time by Vehicle Type
    # --------------------------------------------------------------------------
    print(" [4/11] Generating: Average Delivery Time by Vehicle Type...")
    vehicle_pdf = (
        df.groupBy("Type_of_vehicle")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"))
        .orderBy(F.desc("avg_time"))
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(vehicle_pdf["Type_of_vehicle"], vehicle_pdf["avg_time"], color=["#E84393", "#0984E3", "#00CEC9", "#636E72"])
    ax.set_title("Delivery Duration by Transportation Vehicle Type")
    ax.set_xlabel("Vehicle Mode")
    ax.set_ylabel("Average Time (Minutes)")
    ax.set_ylim(0, max(vehicle_pdf["avg_time"]) * 1.15)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}m", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_avg_time_by_vehicle_type.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 5: Average Delivery Time by Vehicle Condition
    # --------------------------------------------------------------------------
    print(" [5/11] Generating: Average Delivery Time by Vehicle Condition...")
    cond_pdf = (
        df.groupBy("Vehicle_condition")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"))
        .orderBy("Vehicle_condition")
        .toPandas()
    )
    cond_labels = {0: "0 (Poor/Unrated)", 1: "1 (Fair)", 2: "2 (Good)", 3: "3 (Optimal)"}
    cond_pdf["Condition_Label"] = cond_pdf["Vehicle_condition"].map(cond_labels)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(cond_pdf["Condition_Label"], cond_pdf["avg_time"], color=["#D63031", "#E17055", "#00B894", "#0984E3"])
    ax.set_title("Delivery Duration by Fleet Vehicle Condition Tier")
    ax.set_xlabel("Vehicle Maintenance Tier")
    ax.set_ylabel("Average Time (Minutes)")
    ax.set_ylim(0, max(cond_pdf["avg_time"]) * 1.15)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}m", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_avg_time_by_vehicle_condition.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 6: Delivery Time by Order Type
    # --------------------------------------------------------------------------
    print(" [6/11] Generating: Delivery Time by Order Type...")
    order_pdf = (
        df.groupBy("Type_of_order")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"))
        .orderBy(F.desc("avg_time"))
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(order_pdf["Type_of_order"], order_pdf["avg_time"], color="#74B9FF")
    ax.set_title("Delivery Duration by Meal / Order Category")
    ax.set_xlabel("Type of Order")
    ax.set_ylabel("Average Time (Minutes)")
    ax.set_ylim(0, max(order_pdf["avg_time"]) * 1.15)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}m", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_avg_time_by_order_type.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 7: Delivery Time by Festival Status
    # --------------------------------------------------------------------------
    print(" [7/11] Generating: Delivery Time by Festival Status...")
    fest_pdf = (
        df.filter(F.col("Festival") != "Unknown")
        .groupBy("Festival")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"))
        .orderBy(F.desc("avg_time"))
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(fest_pdf["Festival"], fest_pdf["avg_time"], color=["#D63031", "#0984E3"], width=0.5)
    ax.set_title("Festival Impact on Average Delivery Duration")
    ax.set_xlabel("Festival Day Active")
    ax.set_ylabel("Average Time (Minutes)")
    ax.set_ylim(0, max(fest_pdf["avg_time"]) * 1.15)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f} min", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_avg_time_by_festival.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 8: Delivery Time by City Category
    # --------------------------------------------------------------------------
    print(" [8/11] Generating: Delivery Time by City Category...")
    city_pdf = (
        df.filter(F.col("City") != "Unknown")
        .groupBy("City")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"))
        .orderBy(F.desc("avg_time"))
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(city_pdf["City"], city_pdf["avg_time"], color=["#E17055", "#0984E3", "#00B894"])
    ax.set_title("Average Delivery Duration Across Urban City Categories")
    ax.set_xlabel("City Category")
    ax.set_ylabel("Average Delivery Time (Minutes)")
    ax.set_ylim(0, max(city_pdf["avg_time"]) * 1.15)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, f"{b.get_height():.1f}m", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "08_avg_time_by_city_category.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 9: Delivery Time by Courier Rating
    # --------------------------------------------------------------------------
    print(" [9/11] Generating: Delivery Time by Delivery-Person Rating...")
    rating_pdf = (
        df.withColumn("Rating_Bin", F.round(F.col("Delivery_person_Ratings"), 1))
        .groupBy("Rating_Bin")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"), F.count("ID").alias("cnt"))
        .filter(F.col("cnt") > 20)
        .orderBy("Rating_Bin")
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rating_pdf["Rating_Bin"], rating_pdf["avg_time"], marker="o", color=PRIMARY_COLOR, linewidth=2.2, markersize=6)
    ax.set_title("Courier Performance Rating vs. Delivery Duration")
    ax.set_xlabel("Courier Rating (1.0 to 5.0 scale)")
    ax.set_ylabel("Average Delivery Time (Minutes)")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_avg_time_by_courier_rating.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 10: Delivery Time by Approximate Distance
    # --------------------------------------------------------------------------
    print(" [10/11] Generating: Delivery Time by Approximate Straight-Line Distance...")
    dist_pdf = (
        df.filter(F.col("Approx_Distance_Km").isNotNull())
        .withColumn(
            "Dist_Band",
            F.when(F.col("Approx_Distance_Km") < 5.0, "< 5 km (Short)")
             .when(F.col("Approx_Distance_Km") < 10.0, "5 - 10 km (Medium)")
             .when(F.col("Approx_Distance_Km") < 15.0, "10 - 15 km (Long)")
             .otherwise("15 - 21 km (Far)")
        )
        .groupBy("Dist_Band")
        .agg(F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"))
        .toPandas()
    )
    # Order bands logically
    order_dict = {"< 5 km (Short)": 1, "5 - 10 km (Medium)": 2, "10 - 15 km (Long)": 3, "15 - 21 km (Far)": 4}
    dist_pdf["sort_key"] = dist_pdf["Dist_Band"].map(order_dict)
    dist_pdf = dist_pdf.sort_values("sort_key")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(dist_pdf["Dist_Band"], dist_pdf["avg_time"], color=["#00B894", "#0984E3", "#E17055", "#D63031"])
    ax.set_title("Delivery Duration by Approximate Straight-Line Distance")
    ax.set_xlabel("Haversine Distance Band (km)")
    ax.set_ylabel("Average Delivery Time (Minutes)")
    ax.set_ylim(0, max(dist_pdf["avg_time"]) * 1.15)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}m", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "10_avg_time_by_approx_distance.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Chart 11: Monthly Delivery-Time Trends
    # --------------------------------------------------------------------------
    print(" [11/11] Generating: Monthly Delivery-Time Trends...")
    month_pdf = (
        df.groupBy("Order_Month")
        .agg(
            F.round(F.mean("Time_taken (min)"), 2).alias("avg_time"),
            F.count("ID").alias("volume")
        )
        .orderBy("Order_Month")
        .toPandas()
    )
    month_names = {2: "Feb 2022", 3: "Mar 2022", 4: "Apr 2022"}
    month_pdf["Month_Name"] = month_pdf["Order_Month"].map(month_names)

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()

    ax1.bar(month_pdf["Month_Name"], month_pdf["avg_time"], color="#0984E3", alpha=0.8, width=0.45, label="Avg Delivery Time (min)")
    ax2.plot(month_pdf["Month_Name"], month_pdf["volume"], color=PRIMARY_COLOR, marker="s", linewidth=2.5, label="Order Volume")

    ax1.set_title("Operational Trends Across Observed Months (Feb - Apr 2022)")
    ax1.set_xlabel("Operational Month")
    ax1.set_ylabel("Average Delivery Time (min)", color="#0984E3")
    ax2.set_ylabel("Order Count", color=PRIMARY_COLOR)
    ax1.set_ylim(0, 35)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "11_monthly_delivery_trends.png"), dpi=200)
    plt.close(fig)

    print("\n" + "=" * 80)
    print(f"ALL 11 CHARTS SUCCESSFULLY SAVED TO: {output_dir}")
    print("=" * 80)


def run_visualizations():
    spark = get_spark_session("Stage6_Visualizations")
    parquet_path = "data/processed/zomato_cleaned.parquet"
    if not os.path.exists(parquet_path):
        from src.preprocessing import run_preprocessing_pipeline
        run_preprocessing_pipeline()
    df = spark.read.parquet(parquet_path)
    generate_all_visualizations(df)
    spark.stop()


if __name__ == "__main__":
    run_visualizations()
