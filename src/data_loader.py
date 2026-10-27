"""
Data Loader Module for Zomato Delivery Operations Analysis.

This module initializes the Apache Spark session with appropriate memory and local configurations,
and provides utility functions to load raw or processed delivery data.
"""

from typing import Optional
from pyspark.sql import SparkSession, DataFrame


def get_spark_session(app_name: str = "ZomatoDeliveryOperations") -> SparkSession:
    """
    Initializes and returns an Apache SparkSession configured for local execution.
    
    Parameters:
        app_name (str): Name of the Spark application.
        
    Returns:
        SparkSession: Active SparkSession instance.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    # Set log level to WARN or ERROR to suppress verbose INFO logs in terminal
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_raw_dataset(spark: SparkSession, file_path: str = "data/raw/Zomato Dataset.csv") -> DataFrame:
    """
    Loads the raw Zomato delivery CSV file into a PySpark DataFrame.
    
    Note on schema:
        In the raw inspection stage, columns are read as StringType (inferSchema=False)
        or with inferSchema=True to safely observe raw missing representations
        (such as literal 'NaN' strings, malformed decimals, and unusual time formats)
        without dropping or corrupting data during ingestion.
        
    Parameters:
        spark (SparkSession): The active SparkSession.
        file_path (str): Relative or absolute path to the CSV file.
        
    Returns:
        DataFrame: PySpark DataFrame containing raw delivery records.
    """
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(file_path)
    )
    return df
