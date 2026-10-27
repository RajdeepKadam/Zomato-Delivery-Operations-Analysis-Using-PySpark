# College Project Viva Questions & Answers
## Zomato Delivery Operations Analysis Using PySpark

This guide prepares Computer Engineering students for academic viva examinations and technical presentations.

---

### Category 1: Apache Spark & Big Data Architecture

#### Q1: What is Apache Spark, and how does it differ from Apache Hadoop MapReduce?
**Answer:**  
Apache Spark is a unified, in-memory distributed data processing engine. Unlike Hadoop MapReduce, which writes intermediate calculation states to disk after every map and reduce phase, Spark executes computations in RAM using Resilient Distributed Datasets (RDDs) and Directed Acyclic Graphs (DAGs). This makes Spark up to 100 times faster for iterative algorithms and analytical queries.

#### Q2: What is the role of the `SparkSession` in PySpark?
**Answer:**  
`SparkSession` (introduced in Spark 2.0) is the unified entry point for programming with Spark. Prior to Spark 2.0, developers had to manage separate context objects (`SparkContext` for RDDs, `SQLContext` for DataFrames, and `HiveContext` for Hive). `SparkSession` encapsulates all of them into a single builder API. In our project, we initialized it using:
```python
spark = SparkSession.builder.appName("ZomatoAnalytics").master("local[*]").getOrCreate()
```

#### Q3: What is Lazy Evaluation in Apache Spark? How did our project handle it?
**Answer:**  
In Spark, operations on DataFrames are divided into two types:
1. **Transformations** (e.g., `filter()`, `select()`, `withColumn()`, `groupBy()`): These are lazily evaluated. Spark does not compute the data immediately; instead, it records the operations in a logical execution plan (DAG).
2. **Actions** (e.g., `count()`, `collect()`, `show()`, `write`): Actions trigger the actual execution of the DAG.  
In our Stage 8 performance benchmark, we specifically called `.collect()` or `.count()` at the end of each timing block to ensure Spark actually finished executing the computation before we stopped the timer.

#### Q4: What is the Catalyst Optimizer in Apache Spark?
**Answer:**  
The Catalyst Optimizer is Spark SQL's query execution engine. It converts DataFrame code and SQL queries into an optimized physical execution plan through four phases:
1. Analysis (resolving column names and types)
2. Logical Optimization (predicate pushdown, constant folding, projection pruning)
3. Physical Planning (selecting algorithms like HashAggregate or SortAggregate)
4. Code Generation (compiling Java bytecode via WholeStageCodegen).

#### Q5: What is Predicate Pushdown and how does Parquet enable it?
**Answer:**  
Predicate pushdown is an optimization where filter conditions (e.g., `WHERE City = 'Urban'`) are pushed down to the storage layer so that non-matching records or blocks are filtered out *before* data is read into memory. Apache Parquet stores data in columnar chunks with min/max metadata per block, allowing Spark to skip reading entire row groups that do not meet the filter criteria.

---

### Category 2: Data Preprocessing & Feature Engineering

#### Q6: Why did our dataset have missing values represented as literal `'NaN'` strings instead of SQL `NULL`?
**Answer:**  
When tabular data is exported from Python/Pandas or certain web scrapers into CSV files, missing floating-point values are often written out as the characters `"NaN"`. PySpark's default CSV reader parses these as valid 3-character strings rather than missing values. In our preprocessing pipeline, we explicitly handled this by checking:
```python
F.col(c).isNull() | (F.trim(F.col(c)) == "") | (F.trim(F.col(c)) == "NaN")
```
and replacing them with `'Unknown'` for categorical columns and `None` (SQL `NULL`) for numerical columns before casting.

#### Q7: Why did we use Median Imputation instead of Mean Imputation for numerical features?
**Answer:**  
The mean is sensitive to extreme values and skewness, whereas the median represents the 50th percentile and is robust to outliers. In our dataset, features like `multiple_deliveries` and courier ratings were skewed. Using median imputation (Age: 30, Rating: 4.70, Multiple Deliveries: 1.0) prevented distorted feature distributions.

#### Q8: How were the negative GPS coordinates in the dataset identified and resolved?
**Answer:**  
All Indian cities in the dataset (Agra, Pune, Dehradun, Mumbai, etc.) are located in the Northern Hemisphere (positive latitude: ~8°N to 37°N) and Eastern Hemisphere (positive longitude: ~68°E to 97°E). We discovered 431 negative latitudes (e.g., `-18.55°`) and 162 negative longitudes caused by sign inversion during mobile GPS exports. We resolved this by applying `abs()` to convert all coordinates back to positive values, perfectly restoring their true locations.

#### Q9: What is "Null Island" (coordinates 0.0, 0.0), and why did we mask distance calculations for these records?
**Answer:**  
Coordinates `(0.0, 0.0)` refer to the intersection of the Equator and the Prime Meridian in the Gulf of Guinea off Africa. When mobile phone GPS sensors fail to establish a satellite lock within the timeout window, they default to `(0.0, 0.0)`. Calculating the Haversine distance between an Indian restaurant and `(0.0, 0.0)` would yield a false distance of over 7,000 kilometers! We retained these records for non-geographic operational analysis, but masked `Approx_Distance_Km` as `NULL`.

#### Q10: How did we calculate the Haversine straight-line distance in PySpark?
**Answer:**  
We used the spherical law of Haversine:
$$d = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
where $R = 6371.0 \text{ km}$. We implemented this directly using native PySpark SQL functions (`F.radians`, `F.sin`, `F.cos`, `F.asin`, `F.sqrt`), which allowed Spark's Catalyst engine to compile the expression directly into JVM bytecode with WholeStageCodegen, avoiding Python UDF serialization bottlenecks.

#### Q11: How did we handle orders where pickup occurred after midnight?
**Answer:**  
When an order is placed at 23:55 (1,435 minutes into the day) and collected at 00:10 (10 minutes into the next day), a simple subtraction gives $10 - 1435 = -1425$ minutes. We implemented rollover logic in PySpark:
$$\text{If } \text{Pickup\_Minutes} < \text{Order\_Minutes} \implies \text{Order\_to\_Pickup} = (\text{Pickup\_Minutes} - \text{Order\_Minutes}) + 1440$$
which yields the correct kitchen latency of 15 minutes.

---

### Category 3: Spark SQL & Window Functions

#### Q12: What is the difference between a `groupBy()` aggregation and a Window Function in PySpark?
**Answer:**  
A `groupBy()` aggregation collapses rows belonging to the same key into a single summary row. In contrast, a **Window Function** computes aggregations, rankings, or cumulative statistics over a group of rows (defined by `Window.partitionBy()`) while **retaining the original row count and individual row identities**.

#### Q13: Explain the two Window Functions implemented in our project.
**Answer:**  
1. **`DENSE_RANK()` by Traffic Severity within City Hierarchy:**  
   We partitioned deliveries by `City` and ordered them by average delivery duration descending. This ranked traffic conditions (Jam, High, Medium, Low) within each city category.
2. **7-Day Rolling Average Daily Delivery Time:**  
   We created a window over ordered operational dates using `Window.orderBy("Order_Date").rowsBetween(-6, 0)`. This smoothed daily delivery times over a moving 7-day window to evaluate trends over the 44-day timeline.

---

### Category 4: Spark MLlib & Machine Learning

#### Q14: What is a Spark ML Pipeline, and why is it preferred over individual transformations?
**Answer:**  
A Spark ML `Pipeline` chains multiple data transformers (e.g. `StringIndexer`, `OneHotEncoder`, `VectorAssembler`) and estimators (e.g. `RandomForestRegressor`) into a single workflow.  
Benefits:
- Eliminates manual repetitive transformation code.
- Prevents data leakage by fitting transformers (like `StringIndexer` and `Imputer`) strictly on training data and applying the learned mappings to test data.
- Enables single-call model persistence and deployment (`PipelineModel.save()`).

#### Q15: Why did our Random Forest model outperform Linear Regression?
**Answer:**  
- **Linear Regression** assumes linear additive relationships between independent features and delivery time ($R^2 = 0.590$, RMSE = 6.03 min).
- **Random Forest** builds an ensemble of decorrelated decision trees ($R^2 = 0.741$, RMSE = 4.79 min, MAE = 3.76 min). In delivery operations, features have complex non-linear interactions: for example, a Traffic Jam during Heavy Rain combined with Multiple Deliveries compounds delivery delays far more than the simple sum of individual delays. Random Forest splits effectively capture these interactions.

#### Q16: Why did we exclude `ID` and `Delivery_person_ID` from the ML model?
**Answer:**  
`ID` is an arbitrary hexadecimal transaction primary key; using it would cause severe overfitting with zero generalization capability. `Delivery_person_ID` is a high-cardinality identifier (1,320 categories). Including it would cause the model to memorize individual courier identities rather than learning generalizable operational patterns based on age, ratings, vehicle type, and traffic conditions.

---

### Category 5: Operational Domain Insights

#### Q17: What was the single most impactful operational factor causing delivery delays?
**Answer:**  
**Order Stacking (Multiple Deliveries)** and **Festival Surges**:
- Couriers with 0 stacked deliveries averaged **22.88 minutes**, while couriers carrying 3 stacked deliveries averaged **47.82 minutes** (+25 minute delay!).
- Festival days averaged **45.52 minutes** vs non-festival days **25.98 minutes** (+19.5 minute surge delay).

#### Q18: Did meal/order type (Meal, Snack, Drinks, Buffet) affect delivery time?
**Answer:**  
**No.** All four meal categories exhibited virtually identical average delivery durations:
- Meal: 26.42 min
- Snack: 26.29 min
- Buffet: 26.28 min
- Drinks: 26.19 min  
The difference between the highest and lowest meal category was less than 14 seconds (0.23 minutes). Delivery performance is dictated by road conditions, routing, and logistics dispatch, not the type of food in the bag.

#### Q19: Why was Pandas faster than PySpark on our 45,584-row dataset?
**Answer:**  
At 45,584 records (~6 megabytes), the data comfortably fits inside CPU L3 cache and RAM. Pandas runs compiled C/Cython operations directly in memory with zero coordination overhead (~1 to 2 milliseconds). PySpark, being a distributed framework, incurs ~70 milliseconds of fixed baseline overhead:
1. JVM process communication (Py4J socket serialization)
2. DAG compilation and task scheduling
3. Thread context switching across partitions.  
PySpark's advantage appears at scale (gigabytes/terabytes across cluster nodes) where Pandas crashes with `OutOfMemoryError`.
