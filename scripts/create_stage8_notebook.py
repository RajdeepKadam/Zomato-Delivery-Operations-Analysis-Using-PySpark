import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Zomato Delivery Operations Analysis Using PySpark\n",
                "## Stage 8: Controlled Performance Experiment (PySpark vs. Pandas)\n",
                "\n",
                "**Author:** Computer Engineering Mini-Project  \n",
                "**Frameworks:** Apache Spark 4.2 / PySpark vs. Pandas 2.3  \n",
                "**Dataset:** `zomato_cleaned.parquet` (45,584 baseline rows & 455,840 10x scaled rows)\n",
                "\n",
                "---\n",
                "\n",
                "### Objectives of Stage 8:\n",
                "1. Conduct a scientifically fair, controlled benchmark comparing PySpark and Pandas on the exact same dataset.\n",
                "2. Measure equivalent operations: GroupBy Aggregation and Multi-Condition Filtering.\n",
                "3. Account for JVM warmup, task scheduling latency, and lazy evaluation.\n",
                "4. Benchmark horizontal scaling on a 10x synthesized dataset (455,840 records).\n",
                "5. Provide an objective Big Data Engineering interpretation without biased assumptions that distributed computing is automatically faster for small datasets."
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
                "sys.path.append(os.path.abspath('..'))\n",
                "\n",
                "from src.performance import run_benchmark\n",
                "\n",
                "# Run controlled benchmark and display metrics table\n",
                "summary_results = run_benchmark()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Architectural Analysis & Viva Takeaways\n",
                "1. **Why Pandas is Faster on Small Datasets:** At 45,584 records (~6 MB), Pandas operates purely in contiguous RAM using highly optimized C/Cython vectorization. PySpark incurs JVM startup, Py4J gateway socket serialization, task DAG compilation, and thread pool scheduling overhead (~70 ms minimum baseline).\n",
                "2. **When PySpark Becomes Essential:** When dataset size scales to tens of gigabytes or terabytes (exceeding driver RAM), Pandas crashes with an `OutOfMemoryError`. PySpark partitions the workload across worker nodes, streaming data out-of-core and executing resilient distributed computations with fault tolerance."
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

with open("notebooks/05_performance_comparison.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook notebooks/05_performance_comparison.ipynb generated successfully.")
