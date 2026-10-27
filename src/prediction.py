"""
Stage 5: Delivery-Time Prediction Using Spark MLlib
Project: Zomato Delivery Operations Analysis Using PySpark

This module implements a production machine-learning regression pipeline:
1. Feature Preprocessing (Imputation, StringIndexing, OneHotEncoding, VectorAssembly).
2. Model Training (Median Baseline, Linear Regression, Random Forest Regressor).
3. Model Evaluation (RMSE, MAE, R-squared).
4. Feature Importance Profiling.
5. Model Persistence for downstream Streamlit Dashboard inference.
"""

from typing import Dict, Any, Tuple
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from src.data_loader import get_spark_session


def train_delivery_prediction_models(
    df: DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
    model_save_dir: str = "outputs/models/rf_pipeline_model"
) -> Dict[str, Any]:
    """
    Trains and evaluates regression models on the cleaned delivery dataset.
    """
    print("=" * 80)
    print("STAGE 5: DELIVERY-TIME PREDICTION USING SPARK MLLIB")
    print("=" * 80)

    # 1. Feature Specifications
    num_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Vehicle_condition",
        "multiple_deliveries",
        "Is_Weekend"
    ]
    impute_input_cols = ["Approx_Distance_Km", "Order_Hour"]
    impute_output_cols = ["Approx_Distance_Km_imp", "Order_Hour_imp"]

    cat_cols = [
        "Weather_conditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
        "Day_of_Week"
    ]

    # 2. Reproducible Train / Test Split
    print(f"\n[Step 1] Splitting data: {int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Test (seed={seed})")
    train_df, test_df = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f" Train records: {train_count:,} | Test records: {test_count:,}")

    # 3. Pipeline Stages
    imputer = Imputer(inputCols=impute_input_cols, outputCols=impute_output_cols, strategy="median")
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in cat_cols]

    assembler_inputs = num_cols + impute_output_cols + [f"{c}_vec" for c in cat_cols]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")

    # 4. Evaluation Metrics Engine
    eval_rmse = RegressionEvaluator(labelCol="Time_taken (min)", predictionCol="prediction", metricName="rmse")
    eval_mae = RegressionEvaluator(labelCol="Time_taken (min)", predictionCol="prediction", metricName="mae")
    eval_r2 = RegressionEvaluator(labelCol="Time_taken (min)", predictionCol="prediction", metricName="r2")

    # 5. Model 0: Simple Baseline (Training-set Median)
    print("\n[Step 2] Evaluating Baseline Model (Predicts Training Median)...")
    train_median = train_df.approxQuantile("Time_taken (min)", [0.5], 0.001)[0]
    test_baseline = test_df.withColumn("prediction", F.lit(train_median))
    base_rmse = eval_rmse.evaluate(test_baseline)
    base_mae = eval_mae.evaluate(test_baseline)

    print(f" * Baseline (Median={train_median} min): RMSE = {base_rmse:.3f} min | MAE = {base_mae:.3f} min")

    # 6. Model 1: Linear Regression
    print("\n[Step 3] Fitting Linear Regression Model...")
    lr = LinearRegression(featuresCol="features", labelCol="Time_taken (min)", maxIter=25, regParam=0.1)
    pipeline_lr = Pipeline(stages=[imputer] + indexers + encoders + [assembler, lr])
    model_lr = pipeline_lr.fit(train_df)
    pred_lr = model_lr.transform(test_df)
    lr_rmse = eval_rmse.evaluate(pred_lr)
    lr_mae = eval_mae.evaluate(pred_lr)
    lr_r2 = eval_r2.evaluate(pred_lr)
    print(f" * Linear Regression: RMSE = {lr_rmse:.3f} min | MAE = {lr_mae:.3f} min | R2 = {lr_r2:.3f}")

    # 7. Model 2: Random Forest Regressor
    print("\n[Step 4] Fitting Random Forest Regressor (30 trees, maxDepth=8)...")
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="Time_taken (min)",
        numTrees=30,
        maxDepth=8,
        seed=seed
    )
    pipeline_rf = Pipeline(stages=[imputer] + indexers + encoders + [assembler, rf])
    model_rf = pipeline_rf.fit(train_df)
    pred_rf = model_rf.transform(test_df)
    rf_rmse = eval_rmse.evaluate(pred_rf)
    rf_mae = eval_mae.evaluate(pred_rf)
    rf_r2 = eval_r2.evaluate(pred_rf)
    print(f" * Random Forest:     RMSE = {rf_rmse:.3f} min | MAE = {rf_mae:.3f} min | R2 = {rf_r2:.3f}")

    # 8. Feature Importance
    rf_stage: RandomForestRegressionModel = model_rf.stages[-1]
    importances = rf_stage.featureImportances.toArray()
    print(f"\n[Step 5] Extracted {len(importances)} feature importance weights from Random Forest.")

    # 9. Save Best Model
    print(f"\n[Step 6] Saving best model pipeline to: {model_save_dir}")
    os.makedirs(os.path.dirname(model_save_dir), exist_ok=True)
    model_rf.write().overwrite().save(model_save_dir)
    print(" Random Forest PipelineModel persisted successfully.")

    results = {
        "train_median": train_median,
        "baseline": {"rmse": round(base_rmse, 3), "mae": round(base_mae, 3)},
        "linear_regression": {"rmse": round(lr_rmse, 3), "mae": round(lr_mae, 3), "r2": round(lr_r2, 3)},
        "random_forest": {"rmse": round(rf_rmse, 3), "mae": round(rf_mae, 3), "r2": round(rf_r2, 3)},
        "model_path": model_save_dir
    }

    print("\n" + "=" * 80)
    print("STAGE 5 MODELING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return results


def run_prediction_pipeline():
    spark = get_spark_session("Stage5_Prediction")
    parquet_path = "data/processed/zomato_cleaned.parquet"
    if not os.path.exists(parquet_path):
        from src.preprocessing import run_preprocessing_pipeline
        run_preprocessing_pipeline()
    df = spark.read.parquet(parquet_path)
    return train_delivery_prediction_models(df)


if __name__ == "__main__":
    run_prediction_pipeline()
