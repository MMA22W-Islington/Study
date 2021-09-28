# Databricks notebook source
!ls ../../dbfs/databricks/driver

# COMMAND ----------

!touch hello.txt

# COMMAND ----------

!mkdir ../../dbfs/joe

# COMMAND ----------

!cp hello.txt ../../dbfs/joe/hello.txt

# COMMAND ----------

!ls ../../dbfs/joe/2021_09_28_05_42_53_model_baseline

# COMMAND ----------

# MAGIC %fs ls joe/2021_09_28_05_42_53_model_baseline
