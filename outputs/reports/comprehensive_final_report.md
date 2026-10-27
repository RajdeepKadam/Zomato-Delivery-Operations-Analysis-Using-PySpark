# Zomato Delivery Operations Analysis Using PySpark
## Comprehensive Mini-Project Technical Report

**Department:** Computer Engineering  
**Subject:** Big Data Engineering / Advanced Data Analytics  
**Frameworks:** Apache Spark (PySpark), Spark SQL, Spark MLlib, Pandas, Matplotlib, Streamlit  
**Dataset:** `Zomato Dataset.csv` (45,584 records, 20 initial attributes, 31 enriched features)  

---

## 1. Abstract
In modern on-demand food delivery platforms, delivery duration is a critical operational determinant of customer satisfaction, courier earnings, and restaurant throughput. This project presents an end-to-end Big Data analytics and predictive machine learning system developed using Apache Spark (`PySpark`) to investigate operational, environmental, and courier factors influencing delivery durations. 

Using an actual operational dataset of 45,584 delivery records spanning 44 days across multiple Indian metropolitan and urban hubs, we implemented a robust PySpark data cleaning and validation pipeline, formulated feature engineering routines (including spherical Haversine straight-line distance and multi-format timestamp parsing with midnight rollover), executed exploratory distributed aggregations and Spark SQL window functions, trained a Spark MLlib Random Forest predictive regression model (achieving an MAE of 3.76 minutes and an $R^2$ of 0.741), created 11 analytical visualizations, designed an interactive Streamlit operational dashboard, and conducted a controlled performance experiment benchmarking PySpark against Pandas.

---

## 2. Introduction & Background
On-demand logistics requires dynamic optimization under severe stochastic variability: traffic congestion surges during evening commute hours, adverse weather slows vehicular velocities, and order stacking increases courier routing latency. 

Single-node data tools such as standard spreadsheets or Pandas face memory constraints (`OutOfMemoryError`) and performance degradation as delivery platforms ingest hundreds of thousands of daily delivery pings. Apache Spark provides an in-memory distributed computing framework that couples resilient distributed datasets (RDDs) and DataFrames with an optimizing Catalyst query planner, making it an industry-standard engine for enterprise data platforms.

---

## 3. Problem Statement
Given observational delivery records containing courier demographic and rating scores, merchant and drop-off coordinates, transactional timestamps, atmospheric conditions, and road traffic categories:
1. What empirical factors exhibit the strongest association with extended delivery times?
2. How do adverse weather conditions, vehicle maintenance tiers, and festival surges interact with delivery durations?
3. Can an in-memory distributed pipeline accurately estimate delivery duration using operational features available at dispatch time?
4. How do distributed computing engines (PySpark) compare with single-node in-memory libraries (Pandas) across small and scaled workloads?

---

## 4. Project Objectives
- **Data Engineering:** Ingest and sanitize raw delivery logs using PySpark without data loss, addressing missing values, invalid GPS coordinates (0.0 sensor dropouts and sign inversions), and non-standard time representations.
- **Geospatial & Temporal Feature Engineering:** Compute straight-line customer-restaurant distance via the Haversine formula and extract calendar and hourly indicators.
- **Distributed Exploratory Analysis:** Conduct nine comprehensive operations analyses (Analyses A through I) using PySpark DataFrames and Spark SQL.
- **Spark SQL & Window Functions:** Demonstrate multi-dimensional ranking and rolling temporal aggregates.
- **Machine Learning Regression:** Build and evaluate a Spark MLlib pipeline comparing a median baseline, Linear Regression, and Random Forest Regressor.
- **Visual Analytics & Dashboard:** Deliver 11 publication-grade Matplotlib charts and an interactive Streamlit decision dashboard.
- **Scientific Benchmarking:** Empirically evaluate the execution tradeoffs between PySpark and Pandas on small and scaled datasets.

---

## 5. Scope & Limitations
- **Observational Data Only:** All findings represent empirical associations. Correlation does not imply direct physical causation.
- **Approximate Distance:** Geospatial distance is computed as spherical straight-line distance, not actual turn-by-turn road route distance or GPS odometer tracking.
- **Date Span:** The dataset spans February 11, 2022 to April 6, 2022 (44 days); conclusions represent seasonal patterns during this late-winter/early-spring period rather than multi-year trends.
- **No Financial Data:** The dataset contains no customer prices, order monetary values, restaurant revenues, or food margins.

---

## 6. Dataset Description
The source dataset `Zomato Dataset.csv` contains **45,584 records** and **20 columns**:
- **Identifiers:** `ID` (unique hex order key), `Delivery_person_ID` (1,320 unique couriers).
- **Courier Telemetry:** `Delivery_person_Age` (15 to 50 yrs), `Delivery_person_Ratings` (1.0 to 5.0 scale).
- **Geospatial Coordinates:** `Restaurant_latitude`, `Restaurant_longitude`, `Delivery_location_latitude`, `Delivery_location_longitude`.
- **Temporal Fields:** `Order_Date` (`dd-MM-yyyy`), `Time_Orderd`, `Time_Order_picked`.
- **Operational & Environmental Attributes:** `Weather_conditions`, `Road_traffic_density`, `Vehicle_condition` (0-3), `Type_of_order`, `Type_of_vehicle`, `multiple_deliveries` (0-3), `Festival` (`Yes`/`No`), `City` (`Metropolitian`, `Urban`, `Semi-Urban`).
- **Target Variable:** `Time_taken (min)` (Delivery duration in minutes; Mean: 26.29 min, Median: 26.0 min).

---

## 7. System Architecture
```text
                      +-----------------------------+
                      |   Raw CSV: 45,584 records   |
                      +--------------+--------------+
                                     |
                                     v
                 +---------------------------------------+
                 | Stage 1 & 2: PySpark Preprocessing    |
                 | - Schema mapping & 'NaN' string fix   |
                 | - abs() coordinate sign fix           |
                 | - Multi-format time parsing & rollover|
                 | - Haversine distance computation      |
                 +-------------------+-------------------+
                                     |
                                     v
                 +---------------------------------------+
                 | Columnar Parquet Store (31 Columns)   |
                 | data/processed/zomato_cleaned.parquet |
                 +---+---------------+---------------+---+
                     |               |               |
         +-----------+       +-------+-------+       +-----------+
         |                   |               |                   |
         v                   v               v                   v
+-----------------+ +-----------------+ +----------------+ +------------------+
| Stage 3 & 4:    | | Stage 5:        | | Stage 6:       | | Stage 7 & 8:     |
| PySpark EDA &   | | Spark MLlib     | | 11 Matplotlib  | | Streamlit App &  |
| Spark SQL       | | Random Forest   | | Charts         | | Pandas Benchmark |
| Window Functions| | Pipeline Model  | | outputs/charts/| | dashboard/app.py |
+-----------------+ +-----------------+ +----------------+ +------------------+
```

---

## 8. Data Preprocessing & Cleaning Rationale

### A. Missing Values
- **Categorical Columns (`Weather_conditions`, `Road_traffic_density`, `Festival`, `City`):** Mapped string `'NaN'` to `'Unknown'`. This avoids discarding records (preserving 45,584 records) and avoids fabricating data.
- **Numerical Columns (`Delivery_person_Age`, `Delivery_person_Ratings`, `multiple_deliveries`):** Converted to SQL `NULL` and imputed using training medians (Age: 30 yrs, Rating: 4.70, Multiple Deliveries: 1.0).

### B. Telemetry Anomalies
- **Out-of-Bounds Ratings (6.0):** 53 records had rating 6.0; capped at the 5-star maximum scale of 5.0.
- **Sensor Dropout (Null Island):** 3,640 records had coordinates `0.0, 0.0`. Kept records, but masked `Approx_Distance_Km` as `NULL`.
- **Negative Coordinates:** 431 negative latitudes and 162 negative longitudes corrected by applying `abs()`.

### C. Multi-Format Timestamps & Midnight Rollover
Standardized `HH:MM`, `24:MM:SS`, and Excel day fractions (e.g. `0.916666667`) into minute-of-day integers. Calculated kitchen latency `Order_to_Pickup_Minutes` with rollover math:
$$\text{If } \text{Pickup} < \text{Order} \implies \text{Order\_to\_Pickup} = (\text{Pickup} - \text{Order}) + 1440$$

---

## 9. Key Analytical Findings (Stage 3 & 4)

1. **Traffic Congestion:** Traffic Jam delivers average **31.14 min** (median 31 min) vs. Low traffic average **21.27 min** (median 20 min) — a ~10 minute operational penalty.
2. **Weather Conditions:** Sunny days average **21.86 min**, while Fog and Cloudy days average **28.92 min** and **28.98 min** (~7 minute weather delay).
3. **Multiple Deliveries Stacking:** 
   - 0 deliveries: 22.88 min
   - 1 delivery: 26.72 min
   - 2 deliveries: 40.45 min
   - 3 deliveries: 47.82 min (+25 min delay for stacked batches).
4. **Festival Surges:** Deliveries on festival days average **45.52 min** vs. non-festival days **25.98 min** (+19.5 min surge delay).
5. **Urban Hierarchy:** Semi-Urban deliveries average **49.73 min** compared to Urban hubs at **22.98 min**.
6. **Order Type Invariance:** Meal (26.42 min), Snack (26.29 min), Buffet (26.28 min), Drinks (26.19 min) — virtually identical duration across all four meal categories.
7. **Distance Correlation:** Pearson correlation between straight-line distance and delivery time is **+0.3205**.

---

## 10. Machine Learning Predictive Modeling (Stage 5)

An 80/20 train/test split (36,570 training rows, 9,014 testing rows, seed 42) was evaluated using a Spark MLlib Pipeline:

| Model Architecture | RMSE (min) | MAE (min) | $R^2$ Score | Performance vs. Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Train Median = 25.0 min)** | 9.522 | 7.594 | N/A | Naive operational reference |
| **Linear Regression (Regularized)** | 6.034 | 4.790 | 0.590 | 36.6% RMSE reduction |
| **Random Forest Regressor (30 trees, depth 8)** | **4.793** | **3.759** | **0.741** | **49.7% RMSE reduction** |

The Random Forest model captures non-linear interactions (e.g. Traffic Jam compounding with Multiple Deliveries during Bad Weather), achieving an average error of only **3.76 minutes** on unseen test data.

---

## 11. Controlled Performance Benchmark (Stage 8)

| Analytical Operation | Pandas Execution Time | PySpark Execution Time | Engineering Rationale |
| :--- | :---: | :---: | :--- |
| **GroupBy Aggregation (45,584 rows)** | **1.27 ms** | 71.55 ms | Pandas runs in compiled C without JVM/Py4J IPC overhead. |
| **Multi-Condition Filter (45,584 rows)** | **2.43 ms** | 87.23 ms | In-memory contiguous RAM vectorization dominates on small data. |
| **GroupBy Aggregation (455,840 rows - 10x)**| 10.96 ms | 261.42 ms | Spark coordinates parallel execution across 4 partitions. |

**Key Takeaway:** For small datasets fitting in memory, Pandas has significantly lower latency. PySpark's value proposition is horizontal scalability, out-of-core memory safety, and fault-tolerant distributed cluster processing on gigabyte- to terabyte-scale data where Pandas crashes with memory errors.

---

## 12. Conclusion & Future Work

### Conclusion
This project demonstrated that food delivery duration in urban operations is primarily driven by **order stacking (multiple deliveries)**, **road traffic congestion**, **festival events**, and **weather conditions**, while meal type has no measurable correlation. PySpark successfully ingested, cleaned, and modeled this real-world dataset, producing an MLlib predictive model capable of estimating delivery times within ±3.76 minutes.

### Future Work
1. **Turn-by-Turn Road Distances:** Incorporate actual OpenStreetMap or Google Distance Matrix road networks rather than straight-line Haversine estimates.
2. **Kitchen Backlog Telemetry:** Ingest live kitchen order queues and item preparation counts to model food prep latency directly.
3. **Real-Time Streaming:** Migrate batch ingestion to Spark Structured Streaming via Apache Kafka for live delivery tracking and dynamic ETA updating.
