# Stage 1 Dataset Inspection & Telemetry Audit Report
**Project:** Zomato Delivery Operations Analysis Using PySpark  
**Dataset:** `Zomato Dataset.csv`  
**Execution Engine:** Apache Spark local mode via PySpark  
**Total Records:** 45,584  
**Total Columns:** 20  

---

## 1. Executive Summary

A comprehensive dataset inspection was conducted using Apache Spark. The dataset captures real-world food delivery operations across multiple Indian cities between **February 11, 2022** and **April 6, 2022** (44 unique operational days).

Every record represents an individual completed food delivery containing delivery partner attributes, environmental conditions, operational metadata, geospatial coordinates, and the measured delivery duration `Time_taken (min)`.

### Key Verification Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Total Rows** | 45,584 | Complete dataset as specified |
| **Total Columns** | 20 | Identifiers, rider info, GPS coordinates, timestamps, environment, and target |
| **Exact Duplicate Rows** | 0 | Every record has distinct attributes |
| **Unique Order IDs** | 45,584 | `ID` serves as a globally unique primary key |
| **Unique Delivery Personnel** | 1,320 | Couriers handle multiple orders across the observed period |
| **Target Variable** | `Time_taken (min)` | Ranges from 10 to 54 minutes (Mean: 26.29 min, Std: 9.38 min) |

---

## 2. Column-by-Column Data Dictionary & Operational Significance

| Column Name | Raw Type | Missing Count | Missing % | Operational Significance & Analytical Role |
| :--- | :--- | :--- | :--- | :--- |
| `ID` | String | 0 | 0.00% | Unique hexadecimal order identifier (e.g. `0xcdcd`). Primary key; not used as predictive feature. |
| `Delivery_person_ID` | String | 0 | 0.00% | Coded courier identifier (e.g. `DEHRES17DEL01`). Encodes city (`DEH`), restaurant hub (`RES17`), and rider ID (`DEL01`). Useful for courier repeat analysis. |
| `Delivery_person_Age` | String/Numeric | 1,854 | 4.07% | Courier age. Correlates with fitness/experience. Median is 29.5 years. |
| `Delivery_person_Ratings` | String/Numeric | 1,908 | 4.19% | Courier performance rating. Ranges mostly from 1.0 to 5.0 (with 53 anomalous 6.0 values). |
| `Restaurant_latitude` | Float/Double | 0 | 0.00% | Geospatial latitude of the merchant restaurant. |
| `Restaurant_longitude` | Float/Double | 0 | 0.00% | Geospatial longitude of the merchant restaurant. |
| `Delivery_location_latitude` | Float/Double | 0 | 0.00% | Geospatial latitude of the customer delivery drop-off point. |
| `Delivery_location_longitude` | Float/Double | 0 | 0.00% | Geospatial longitude of the customer delivery drop-off point. |
| `Order_Date` | String (Date) | 0 | 0.00% | Order placement date formatted as `dd-MM-yyyy`. Spans Feb 11 to Apr 6, 2022. |
| `Time_Orderd` | String (Time) | 1,731 | 3.80% | Timestamp when customer placed the order. Useful for calculating order-to-pickup kitchen prep time. |
| `Time_Order_picked` | String (Time) | 0 | 0.00% | Timestamp when courier picked up package from restaurant. |
| `Weather_conditions` | String | 616 | 1.35% | Ambient weather (Fog, Stormy, Cloudy, Sandstorms, Windy, Sunny). Directly impacts road safety and speed. |
| `Road_traffic_density` | String | 601 | 1.32% | Congestion category (Low, Medium, High, Jam). Major operational bottleneck. |
| `Vehicle_condition` | Integer (0-3) | 0 | 0.00% | Fleet maintenance tier (0 = poor/unrated, 1 = average, 2 = good, 3 = optimal). |
| `Type_of_order` | String | 0 | 0.00% | Meal category (Snack, Meal, Drinks, Buffet). Affects kitchen prep load. |
| `Type_of_vehicle` | String | 0 | 0.00% | Transportation mode (motorcycle, scooter, electric_scooter, bicycle). |
| `multiple_deliveries` | Float/Integer | 993 | 2.18% | Number of stacked orders assigned to the courier simultaneously (0, 1, 2, 3). |
| `Festival` | String | 228 | 0.50% | Binary indicator (`Yes`, `No`) for festival dates with surge volume and road diversions. |
| `City` | String | 1,200 | 2.63% | Urban categorization (`Metropolitian`, `Urban`, `Semi-Urban`). |
| `Time_taken (min)` | Integer | 0 | 0.00% | Total delivery duration in minutes. The primary analytical and ML regression target. |

---

## 3. Discovered Anomalies & Data Quality Telemetry

The PySpark inspection uncovered several critical anomalies that must be addressed during Stage 2 Data Preprocessing:

### A. Missing Values Embedded as Literal `'NaN'` Strings
In standard database systems, missing values are stored as SQL `NULL`. However, in this dataset, missing values in columns such as `Delivery_person_Age`, `Delivery_person_Ratings`, `multiple_deliveries`, `Weather_conditions`, `Road_traffic_density`, `Festival`, and `City` are stored as literal string `'NaN'` values.  
*Action:* A strict missing-value parser must map both SQL `NULL`, empty strings, and string `'NaN'` to a unified missing representation.

### B. Out-of-Bounds Ratings and Correlated Anomalies
- Exactly **53 records** exhibit a `Delivery_person_Ratings` of **6.0** (on a 1.0 to 5.0 scale).
- These exact **53 records** also record a `Delivery_person_Age` of **50.0**.
- Exactly **38 records** exhibit a `Delivery_person_Age` of **15.0** (underage delivery courier).
*Action:* In Stage 2, decide whether to cap/impute or isolate these records.

### C. Geospatial Telemetry Inconsistencies (Null Island & Negative Coordinates)
1. **Sensor Dropouts (0.0, 0.0):** Exactly **3,640 records** have restaurant or customer coordinates recorded as `0.0, 0.0`. In GPS telemetry, this represents a hardware timeout ("Null Island" off the coast of Africa). Straight-line distance cannot be calculated for these rows.
2. **Inverted Hemisphere Signs:** Exactly **431 records** have negative latitudes (e.g. `-27.16°`) and **162 records** have negative longitudes (e.g. `-78.05°`). Because all recorded Indian cities are located in the Northern and Eastern hemispheres (positive latitudes and longitudes), these reflect inadvertent negative sign errors in mobile GPS exports.
*Action:* For negative coordinates, take the absolute value `abs(coordinate)`; for `0.0` records, mask the derived distance as `NULL` so downstream models do not consume false 7,000 km distances.

### D. Multi-Format Timestamps & Excel Fractional Days
- **3,638 records** in `Time_Orderd` and **3,687 records** in `Time_Order_picked` are stored as decimal numbers (e.g., `0.916666667`, `0.458333333`). These occur when raw CSVs are saved through Microsoft Excel, which converts times to day fractions (`0.916666667 * 24 hours = 22:00:00`).
- **880 records** use `HH:MM:SS` format (e.g., `24:05:00`), where `24:05:00` represents five minutes past midnight.
*Action:* Implement a multi-format time parser in PySpark to convert Excel fractions, `HH:mm`, and `HH:mm:ss` into consistent hour, minute, and elapsed duration metrics.

---

## 4. Categorical Breakdown Summary

- **Weather:** Fog (7,653), Stormy (7,584), Cloudy (7,533), Sandstorms (7,494), Windy (7,422), Sunny (7,282), Missing (616).
- **Traffic Density:** Low (15,476), Jam (14,139), Medium (10,945), High (4,423), Missing (601).
- **Vehicle Type:** Motorcycle (26,429), Scooter (15,273), Electric Scooter (3,814), Bicycle (68).
- **Vehicle Condition:** 2 (15,031), 1 (15,028), 0 (15,005), 3 (520).
- **Multiple Deliveries:** 1 (28,151), 0 (14,094), 2 (1,985), 3 (361), Missing (993).
- **Festival:** No (44,460), Yes (896), Missing (228).
- **City Category:** Metropolitian (34,087), Urban (10,133), Semi-Urban (164), Missing (1,200).

---

## 5. Next Steps for Stage 2 (Preprocessing)

With Stage 1 verification complete, the preprocessing pipeline will implement:
1. Unified null handling (`'NaN'` string -> `None`).
2. Robust time parsing supporting both `HH:mm:ss` and Excel decimal fractions.
3. Coordinate rectification (taking `abs()` for negative coords; setting `0.0` coords to `None` for distance computation).
4. Feature engineering (`Approx_Distance_Km` via Haversine, `Order_to_Pickup_Minutes`, `Order_Hour`, `Day_of_Week`, `Is_Weekend`).
5. Parquet checkpointing for fast Stage 3 exploratory analytics.
