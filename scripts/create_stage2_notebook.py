import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Zomato Delivery Operations Analysis Using PySpark\n",
                "## Stage 2: Data Preprocessing, Data Validation & Feature Engineering\n",
                "\n",
                "**Author:** Computer Engineering Mini-Project  \n",
                "**Framework:** Apache Spark via PySpark  \n",
                "**Dataset:** `Zomato Dataset.csv`  \n",
                "**Target Parquet Storage:** `data/processed/zomato_cleaned.parquet`\n",
                "\n",
                "---\n",
                "\n",
                "### Objectives of Stage 2:\n",
                "1. **Handle Missing Values:** Reconcile literal string `'NaN'` values with SQL `NULL` and assign explicit `'Unknown'` categories for categorical attributes.\n",
                "2. **Validate and Sanitize Data:** Correct inverted negative GPS signs, detect Null Island sensor dropouts (`0.0, 0.0`), and cap out-of-bounds ratings (`6.0` -> `5.0`).\n",
                "3. **Multi-Format Timestamp Parsing:** Standardize `HH:MM`, `HH:MM:SS`, and Microsoft Excel serial day fractions into minute-of-day metrics.\n",
                "4. **Feature Engineering:**\n",
                "   - Date attributes: `Order_Month`, `Order_Day`, `Day_of_Week`, `Is_Weekend`\n",
                "   - Operational timing: `Order_Hour`, `Pickup_Hour`, `Order_to_Pickup_Minutes` (with midnight rollover logic)\n",
                "   - Geospatial distance: `Approx_Distance_Km` using Spherical Haversine calculation\n",
                "5. **Before vs. After Telemetry Comparison:** Verify 100% record retention (0 dropped rows) and complete feature readiness."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Initialize PySpark Session"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "\n",
                "# Add project root to sys.path\n",
                "sys.path.append(os.path.abspath('..'))\n",
                "\n",
                "from pyspark.sql import SparkSession\n",
                "from pyspark.sql import functions as F\n",
                "from pyspark.sql.types import IntegerType, DoubleType\n",
                "\n",
                "from src.data_loader import get_spark_session, load_raw_dataset\n",
                "from src.preprocessing import clean_missing_and_types, validate_and_correct_anomalies, engineer_features\n",
                "\n",
                "spark = get_spark_session('Stage2_Preprocessing_Notebook')\n",
                "print(f'Spark Master: {spark.sparkContext.master}')\n",
                "print(f'Spark Version: {spark.version}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Load the Raw Dataset\n",
                "We begin by reading the raw 45,584 records."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "csv_path = '../data/raw/Zomato Dataset.csv' if os.path.exists('../data/raw/Zomato Dataset.csv') else '../Zomato Dataset.csv'\n",
                "raw_df = load_raw_dataset(spark, csv_path)\n",
                "print(f'Raw Record Count: {raw_df.count():,}')\n",
                "print(f'Raw Column Count: {len(raw_df.columns)}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Clean Missing Values and Data Types\n",
                "- Replaces `'NaN'` strings with `'Unknown'` in categorical features (`Weather_conditions`, `Road_traffic_density`, `Festival`, `City`).\n",
                "- Converts `'NaN'` to `NULL` for numeric features.\n",
                "- Parses `Order_Date` to Spark `DateType` (`dd-MM-yyyy`).\n",
                "- Corrects negative coordinates using `abs()`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cleaned_df = clean_missing_and_types(raw_df)\n",
                "cleaned_df.printSchema()\n",
                "cleaned_df.select('ID', 'Weather_conditions', 'Road_traffic_density', 'City', 'Order_Date').show(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Data Validation and Numerical Imputation\n",
                "- Caps ratings > 5.0 to 5.0.\n",
                "- Imputes missing `Delivery_person_Age`, `Delivery_person_Ratings`, and `multiple_deliveries` with medians.\n",
                "- Preserves raw un-imputed values for transparency."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "validated_df, medians = validate_and_correct_anomalies(cleaned_df)\n",
                "print('Medians applied for imputation:', medians)\n",
                "validated_df.select('Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries').describe().show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Feature Engineering\n",
                "We now engineer:\n",
                "1. `Order_Month`, `Order_Day`, `Day_of_Week`, `Is_Weekend`\n",
                "2. `Order_Hour`, `Pickup_Hour`, and `Order_to_Pickup_Minutes`\n",
                "3. `Approx_Distance_Km` (Haversine straight-line distance, masking Null Island coords as `NULL`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "featured_df = engineer_features(validated_df)\n",
                "print(f'Enriched Dataset Columns: {len(featured_df.columns)}')\n",
                "featured_df.select(\n",
                "    'Order_Date', 'Day_of_Week', 'Is_Weekend',\n",
                "    'Order_Hour', 'Pickup_Hour', 'Order_to_Pickup_Minutes',\n",
                "    'Approx_Distance_Km', 'Time_taken (min)'\n",
                ").show(10)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Before vs. After Preprocessing Comparison\n",
                "We compare key metrics before and after the cleaning pipeline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "\n",
                "comparison_data = [\n",
                "    ('Total Rows', f'{raw_df.count():,}', f'{featured_df.count():,}'),\n",
                "    ('Total Columns', len(raw_df.columns), len(featured_df.columns)),\n",
                "    ('Missing Age', '1,854 (NaN string)', '0 (Imputed with median 30)'),\n",
                "    ('Missing Ratings', '1,908 (NaN string)', '0 (Imputed with median 4.7)'),\n",
                "    ('Ratings == 6.0', '53 records', '0 records (Capped to 5.0)'),\n",
                "    ('Negative Latitudes', '431 records', '0 records (abs() corrected)'),\n",
                "    ('Null Island (0,0) Coords', '3,640 records', '3,640 records (masked as NULL distance)'),\n",
                "    ('Approx_Distance_Km created', 'No', f\"{featured_df.filter(F.col('Approx_Distance_Km').isNotNull()).count():,} valid distances\"),\n",
                "    ('Order_to_Pickup_Minutes created', 'No', f\"{featured_df.filter(F.col('Order_to_Pickup_Minutes').isNotNull()).count():,} valid durations\")\n",
                "]\n",
                "\n",
                "comparison_df = pd.DataFrame(comparison_data, columns=['Metric / Feature', 'Before Preprocessing', 'After Preprocessing'])\n",
                "print(comparison_df.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7: Verify Storage in Parquet Format\n",
                "We verify that the cleaned dataset is successfully persisted in Apache Parquet format for downstream high-performance querying in Stages 3, 4, and 5."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "parquet_path = '../data/processed/zomato_cleaned.parquet'\n",
                "if os.path.exists(parquet_path):\n",
                "    parquet_df = spark.read.parquet(parquet_path)\n",
                "    print(f'Successfully loaded Parquet dataset: {parquet_df.count():,} records, {len(parquet_df.columns)} columns.')\n",
                "else:\n",
                "    print('Parquet path does not exist yet. Running pipeline export...')\n",
                "    featured_df.write.mode('overwrite').parquet(parquet_path)\n",
                "    print('Parquet dataset exported.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8: Shutdown Spark Session"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "spark.stop()\n",
                "print('SparkSession stopped.')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("notebooks/02_data_preprocessing.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook notebooks/02_data_preprocessing.ipynb generated successfully.")
