# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Using Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Basic operations
# MAGIC What is:
# MAGIC - A notebook
# MAGIC - A code cell
# MAGIC 
# MAGIC How to:
# MAGIC - Create a notebook
# MAGIC - Run a code cell
# MAGIC - Autosave and revision history
# MAGIC - Cluster is off

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Write Python or R code in Databricks

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC message = "Hello World"
# MAGIC print(message)

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC message <- "Hello World"
# MAGIC print(message)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data analysis in the normal way

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Reading data from an external source**
# MAGIC 
# MAGIC Python: pandas read_csv function, document at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# MAGIC 
# MAGIC R: SparkR read.csv function, document at: https://stat.ethz.ch/R-manual/R-devel/library/utils/html/read.table.html

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC import pandas as pd
# MAGIC 
# MAGIC data = pd.read_csv("https://github.com/lindayi/MMA865-Fall19/releases/download/dataset/sample_data.csv", header = 0)
# MAGIC print(data.head(5))

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC data <- read.csv("https://github.com/lindayi/MMA865-Fall19/releases/download/dataset/sample_data.csv", header=TRUE)
# MAGIC print(head(data, 5))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Get some basic statistics**

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC import numpy as np
# MAGIC 
# MAGIC print("Count of records:", len(data))
# MAGIC print("Median of x:", np.median(data["x"]))
# MAGIC print("Max of y:", np.max(data["y"]))

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC print(paste("Count of records: ", nrow(data)))
# MAGIC print(paste("Median of x: ", median(data[,"x"])))
# MAGIC print(paste("Max of y: ", max(data[,"y"])))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Plot some figures**
# MAGIC 
# MAGIC _Reference: https://blog.arinti.be/azure-databricks-plotting-data-made-simple-e3b70a0b1a14_

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC display(data)

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC import matplotlib.pyplot as plt
# MAGIC 
# MAGIC fig, ax = plt.subplots() 
# MAGIC ax.scatter(data["x"], data["y"], alpha=1)
# MAGIC display(fig)

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC plot(data[,"x"], data[,"y"])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data Analysis in the Spark way

# COMMAND ----------

# MAGIC %md 
# MAGIC **Reading data from an external source**
# MAGIC 
# MAGIC Python: Pyspark spark.read.csv function, document at: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=read%20csv#pyspark.sql.DataFrameReader.csv
# MAGIC 
# MAGIC R: SparkR read.df function, document at: https://spark.apache.org/docs/latest/api/R/read.df.html

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC df = spark.sql("select * from tmp.sample_data")
# MAGIC display(df)

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC library(SparkR)
# MAGIC 
# MAGIC df = sql("select * from tmp.sample_data")
# MAGIC display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Basic stats**

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC print("Count of records:", df.count())
# MAGIC print("Median of x:", df.approxQuantile("x", [0.5], 0)[0])
# MAGIC print("Max of y:", df.agg({"y": "max"}).collect()[0]["max(y)"])

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC print(paste("Count of records: ", count(df)))
# MAGIC print(paste("Median of x: ", approxQuantile(df, "x", c(0.5), 0)))
# MAGIC print(paste("Max of y: ", first(agg(df, max = max(df$y)))$max))

# COMMAND ----------

# MAGIC %md
# MAGIC **Plot some figures**
# MAGIC 
# MAGIC We first collect data back to the master node, then do plotting exactly as if we are writing normal python / R code

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC display(df)

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC import matplotlib.pyplot as plt
# MAGIC 
# MAGIC pdDF = df.toPandas()
# MAGIC 
# MAGIC fig, ax = plt.subplots() 
# MAGIC ax.scatter(pdDF["x"], pdDF["y"], alpha=1)
# MAGIC display(fig)

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC converted_df <- collect(df)
# MAGIC 
# MAGIC plot(converted_df[,"x"], converted_df[,"y"])

# COMMAND ----------


