# Stage 2 Data Preprocessing & Feature Engineering Report
**Project:** Zomato Delivery Operations Analysis Using PySpark  
**Dataset:** `Zomato Dataset.csv` -> `data/processed/zomato_cleaned.parquet`  
**Execution Engine:** Apache Spark Catalyst Native Expressions  
**Total Records Processed:** 45,584 (0 records dropped; 100% data retention)  
**Total Enriched Columns:** 31 columns  

---

## 1. Executive Summary

Stage 2 transforms raw delivery logs into an analytics- and machine-learning-ready dataset. The pipeline was designed in Apache Spark using native Catalyst expressions to maximize memory efficiency and execution speed across local CPU cores.

Every cleaning decision was made based on domain knowledge of food delivery operations and the telemetry findings uncovered in Stage 1.

---

## 2. Cleaning & Imputation Decisions Log

| Column / Feature | Observed Data Quality Issue | Action Taken & PySpark Strategy | Operational Rationale |
| :--- | :--- | :--- | :--- |
| `Weather_conditions` | 616 `'NaN'` literal string values | Replaced `'NaN'` and empty strings with `'Unknown'` category. | Preserves true sample count without fabricating weather data that could bias analysis. |
| `Road_traffic_density` | 601 `'NaN'` literal string values | Replaced `'NaN'` and empty strings with `'Unknown'` category. | Allows Spark SQL and downstream queries to isolate missing traffic telemetry. |
| `Festival` | 228 `'NaN'` literal string values | Replaced `'NaN'` with `'Unknown'`. | Prevents misclassifying unknown dates as non-festival days. |
| `City` | 1,200 `'NaN'` literal string values | Replaced `'NaN'` with `'Unknown'`. | Retains all 1,200 orders for overall performance analysis while categorizing location accurately. |
| `Delivery_person_Age` | 1,854 `'NaN'` values; 38 records with Age 15; 53 records with Age 50. | Converted `'NaN'` to `NULL`. Preserved raw column as `Delivery_person_Age_Raw` and imputed missing values using training median (**30.0 years**). | Median is robust to skew and outliers. Retaining the raw column avoids data leakage. |
| `Delivery_person_Ratings` | 1,908 `'NaN'` values; 53 records with out-of-bounds `6.0` rating. | Converted `'NaN'` to `NULL`. Capped values > 5.0 to 5.0 (5-star rating scale upper bound). Imputed missing values using median (**4.70**). | Eliminates impossible rating scores while preserving courier records. |
| `multiple_deliveries` | 993 `'NaN'` values. | Converted `'NaN'` to `NULL`. Imputed missing values using median (**1.0 order**). | 1 order is both the median and the mode (62% of all deliveries). |
| `Restaurant_latitude` & `longitude` | 431 negative latitudes and 162 negative longitudes (sign inversions). | Applied `abs()` transformation to flip negative coordinates back to Northern and Eastern hemispheres. | Rectifies data entry sign inversion without deleting valid geospatial locations. |
| `Coordinates at 0.0, 0.0` | 3,640 records recorded at Null Island (0.0, 0.0). | Kept rows, but set derived `Approx_Distance_Km` to `NULL` for any record where restaurant or delivery coordinate is 0.0. | Prevents false ~7,000 km straight-line distance calculations that would distort downstream analytics. |
| `Time_Orderd` & `Time_Order_picked` | 3,638 order times stored as Excel day decimals; 880 times stored as `24:05:00`. | Implemented native Catalyst parser converting `HH:MM`, `24:MM:SS`, and Excel day fractions into minute-of-day [0, 1439]. | Standardizes all timestamps across formats without dropping records. |

---

## 3. Feature Engineering Summary

| Feature Name | Type | Formula / Logic | Analytical & Operational Purpose |
| :--- | :--- | :--- | :--- |
| `Order_Month` | Integer | `month(Order_Date)` | Captures month-level patterns (February, March, April 2022). |
| `Order_Day` | Integer | `dayofmonth(Order_Date)` | Captures day-of-month delivery volume cycles. |
| `Day_of_Week` | String | `date_format(Order_Date, 'E')` | Categorizes days (Mon, Tue, Wed, Thu, Fri, Sat, Sun). |
| `Is_Weekend` | Integer | `when(dayofweek(Order_Date).isin([1, 7]), 1).otherwise(0)` | Binary indicator (1 = Sat/Sun, 0 = Mon-Fri) for weekend surge analysis. |
| `Order_Hour` | Integer | `floor(Order_Minute_of_Day / 60)` | Peak hour segmentation (lunch 12-14, dinner 19-22). |
| `Pickup_Hour` | Integer | `floor(Pickup_Minute_of_Day / 60)` | Identifies courier pickup peak periods. |
| `Order_to_Pickup_Minutes`| Integer | `Pickup_Min - Order_Min` (+ 1440 if rollover past midnight) | Measures kitchen food preparation and courier dispatch latency. |
| `Approx_Distance_Km` | Double | Spherical Haversine distance on corrected coordinates | Approximates straight-line customer-restaurant distance (1.47 km to 20.97 km). |

> [!NOTE]  
> **Important Distinction:** `Approx_Distance_Km` is straight-line spherical distance calculated from coordinates. It is not actual road distance, route distance, or GPS-tracked travel distance.

---

## 4. Before vs. After Preprocessing Telemetry

| Metric / Attribute | Before Preprocessing (Raw CSV) | After Preprocessing (Parquet) |
| :--- | :--- | :--- |
| **Total Rows** | 45,584 | **45,584** (0 dropped rows) |
| **Total Columns** | 20 | **31** (11 engineered/audit columns added) |
| **Missing Age Records** | 1,854 (4.07% stored as `'NaN'`) | 0 in imputed column (median: 30 yrs) |
| **Missing Ratings Records** | 1,908 (4.19% stored as `'NaN'`) | 0 in imputed column (median: 4.7) |
| **Ratings == 6.0** | 53 records | 0 (Capped to 5.0) |
| **Missing Multiple Deliveries**| 993 (2.18% stored as `'NaN'`) | 0 in imputed column (median: 1.0) |
| **Negative Latitudes** | 431 records | 0 (`abs()` applied) |
| **Negative Longitudes** | 162 records | 0 (`abs()` applied) |
| **Valid Approx_Distance_Km** | None (feature did not exist) | **41,944 valid distances** (Mean: 9.72 km) |
| **Valid Order_to_Pickup** | None (feature did not exist) | **43,853 valid durations** (Mean: 9.99 min) |
| **Storage Format** | Unindexed raw CSV (6.4 MB) | Optimized Columnar Parquet (`snappy` compressed) |

---

## 5. Storage & Artifact Verification

The cleaned and enriched dataset has been persisted in two formats:
1. **Primary Analytics Store:** `data/processed/zomato_cleaned.parquet` (Columnar format optimized for Spark Catalyst predicate pushdown and column projection).
2. **Inspection Mirror:** `data/processed/zomato_cleaned.csv` (Single coalesced CSV for spreadsheet or lightweight inspection).
3. **Interactive Notebook:** `notebooks/02_data_preprocessing.ipynb` (Fully executed notebook with intermediate tables).
4. **Unit Test Suite:** `tests/test_preprocessing.py` (8 automated tests covering timestamp parsing, midnight rollover, coordinate sign rectification, and null handling; 100% pass rate).
