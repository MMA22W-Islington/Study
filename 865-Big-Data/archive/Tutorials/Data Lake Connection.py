# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Azure Data Lake Connection
# MAGIC 
# MAGIC **NOTE: This notebook contains the Access Key to your group's Azure Data Lake, which has the confidential dataset for your group project.** 
# MAGIC 
# MAGIC **DO NOT SHARE THE ACCESS KEY OR THIS FIILE OUTSIDE OF YOUR GROUP**

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Structure of your data lake
# MAGIC 
# MAGIC Each of the group has a Azure Data Lake. Within each data lake, there are two blob containers:
# MAGIC 
# MAGIC - **data**: where your group project dataset is stored; and
# MAGIC - **workspace**: where you can store any temporary files or result datasets for your group project
# MAGIC 
# MAGIC The data lake is programmatically accessible using the access key of the data lake. Once set, you can use the abfss:// path to read from and write to data lake.

# COMMAND ----------

# This is the access key for the sample datalake "ssbdatalakegen2".

spark.conf.set(
  "fs.azure.account.key.ssbdatalakegen2.dfs.core.windows.net",
  "PIaF7byqwANFUKfbdpp8taw/lsyFNsRKNu6bf7XHA6pk7DEABggBFaeSZ2tIDqDCNZ3xEbwz3G4/ziWnGWSfjQ==")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Below are some examples of filesystem operations in data lake. Read more about dbutils.fs here: https://docs.databricks.com/user-guide/databricks-file-system.html#dbfs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Files in "data" blob container

# COMMAND ----------

# Listing files in data

display(dbutils.fs.ls("abfss://data@ssbdatalakegen2.dfs.core.windows.net/"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Files in "workspace" blob container

# COMMAND ----------

# Listing files in workspace

display(dbutils.fs.ls("abfss://workspace@ssbdatalakegen2.dfs.core.windows.net/"))
