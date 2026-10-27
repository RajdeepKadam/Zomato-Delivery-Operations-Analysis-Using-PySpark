# Zomato Delivery Operations Analysis Using PySpark

A complete college mini-project in Big Data Engineering investigating food delivery operations, courier performance, environmental impacts, and predictive modeling using **Apache Spark (PySpark)**, **Spark SQL**, and **Spark MLlib**.

---

## 1. Project Overview
- **Dataset:** `Zomato Dataset.csv` (45,584 records, 20 initial attributes -> 31 enriched features)
- **Time Window:** February 11, 2022 to April 6, 2022 (44 operational days)
- **Primary Metric:** `Time_taken (min)` (Delivery Duration in Minutes; Mean: 26.29 min, Median: 26.0 min)
- **Core Technology Stack:**
  - **Big Data Engine:** Apache Spark 4.2 / PySpark local mode
  - **SQL Engine:** Spark SQL with Temporary Views & Catalyst Optimizer
  - **Machine Learning:** Spark MLlib (`LinearRegression`, `RandomForestRegressor`, `Pipeline`)
  - **Visualizations:** Matplotlib (11 publication-grade charts)
  - **Interactive Dashboard:** Streamlit
  - **Testing:** PyTest (8 automated unit tests; 100% pass rate)

---

## 2. Directory Structure

```text
Zomato Delivery Operations Analysis Using PySpark/
├── data/
│   ├── raw/                      # Original raw dataset: Zomato Dataset.csv
│   └── processed/
│       ├── zomato_cleaned.parquet # High-performance columnar Parquet dataset
│       └── zomato_cleaned.csv     # Coalesced CSV mirror for inspection
├── notebooks/
│   ├── 01_dataset_inspection.ipynb     # Stage 1: Data ingestion & telemetry audit
│   ├── 02_data_preprocessing.ipynb     # Stage 2: Data cleaning & feature engineering
│   ├── 03_exploratory_analysis.ipynb   # Stage 3 & 4: Distributed EDA & Spark SQL
│   ├── 04_prediction_model.ipynb       # Stage 5: Spark MLlib regression modeling
│   └── 05_performance_comparison.ipynb # Stage 8: PySpark vs. Pandas benchmark
├── src/
│   ├── data_loader.py            # SparkSession factory & dataset ingestion
│   ├── stage1_inspect.py         # Stage 1 standalone inspection runner
│   ├── preprocessing.py          # Stage 2 cleaning, validation & Haversine pipeline
│   ├── analysis.py               # Stage 3 & 4 operations queries & window functions
│   ├── visualization.py          # Stage 6 Matplotlib chart generator (11 charts)
│   ├── prediction.py             # Stage 5 Spark MLlib training & model persistence
│   └── performance.py            # Stage 8 controlled PySpark vs. Pandas benchmark
├── dashboard/
│   └── app.py                    # Stage 7 interactive Streamlit decision dashboard
├── tests/
│   └── test_preprocessing.py     # Automated unit tests for preprocessing logic
├── outputs/
│   ├── charts/                   # 11 exported analytical charts (.png)
│   ├── models/                   # Persisted Spark MLlib Random Forest PipelineModel
│   └── reports/
│       ├── stage1_inspection_report.md       # Stage 1 audit report
│       ├── stage2_preprocessing_report.md    # Stage 2 cleaning & engineering report
│       ├── comprehensive_final_report.md     # Full college mini-project technical report
│       └── viva_questions_and_answers.md     # Extensive Viva preparation guide (19 Q&As)
├── requirements.txt              # Pinned, tested compatible dependencies
├── README.md                     # Complete project documentation
└── .gitignore                    # Prevents checking in .venv & Spark temp files
```

---

## 3. Quickstart & Execution Guide (macOS & VS Code)

### Prerequisites
- macOS (Apple Silicon or Intel)
- Python 3.10+ (Tested on Python 3.11)
- Java 17 or Java 21 (Tested on Oracle JDK 21)

### Step 1: Virtual Environment Setup
Open Terminal in the project root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Automated Unit Tests
```bash
pytest tests/
```
*Result: 8/8 passed in 0.04s.*

### Step 3: Run Each Stage in Terminal
```bash
# Stage 1: Dataset Inspection
python3 src/stage1_inspect.py

# Stage 2: Data Preprocessing & Feature Engineering (generates Parquet)
python3 src/preprocessing.py

# Stage 3 & 4: Exploratory Analysis & Spark SQL Queries
python3 src/analysis.py

# Stage 5: Train Spark MLlib Prediction Pipeline
python3 src/prediction.py

# Stage 6: Generate all 11 Analytical Visualizations
python3 src/visualization.py

# Stage 8: Run Controlled Performance Experiment (Spark vs. Pandas)
python3 src/performance.py
```

### Step 4: Launch the Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501` to view:
- **Operations Overview:** Top KPIs, distributions, and cleaning audit.
- **Interactive Insights:** Dynamic multi-factor filtering (traffic, weather, vehicle, city, festival).
- **Delivery Time Predictor:** Real-time ETA estimation using our trained Spark MLlib model.

---

## 4. Key Analytical Insights Summary

1. **Traffic Congestion Impact:** Deliveries during Traffic Jams average **31.14 min** vs. Low traffic at **21.27 min** (~10 min delay).
2. **Weather Conditions:** Sunny days average **21.86 min**, while Fog and Cloudy days average **28.92 min** (~7 min weather penalty).
3. **Multiple Deliveries Stacking:** 
   - 0 deliveries: 22.88 min
   - 1 delivery: 26.72 min
   - 2 deliveries: 40.45 min
   - 3 deliveries: 47.82 min (+25 min delay for 3 stacked orders!).
4. **Festival Surges:** Festival days average **45.52 min** vs. non-festival days **25.98 min** (+19.5 min surge delay).
5. **Order Type Invariance:** Meal (26.42 min), Snack (26.29 min), Buffet (26.28 min), Drinks (26.19 min) — virtually zero difference across meal categories (< 14 seconds).
6. **Machine Learning Model:** Random Forest Regressor achieves **RMSE = 4.79 min**, **MAE = 3.76 min**, and **$R^2 = 0.741$** (a 49.7% RMSE improvement over the median baseline).

---

## 5. College Report & Viva Preparation Guides

- **Comprehensive Technical Report:** [outputs/reports/comprehensive_final_report.md](outputs/reports/comprehensive_final_report.md)
- **Viva Preparation Guide (19 Q&As):** [outputs/reports/viva_questions_and_answers.md](outputs/reports/viva_questions_and_answers.md)
- **Pre-Executed Jupyter Notebooks:** Available in the `notebooks/` directory.
