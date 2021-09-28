# Databricks notebook source
!ls ../../dbfs/databricks/driver

# COMMAND ----------

!touch hello.txt

# COMMAND ----------

!mkdir ../../dbfs/joe

# COMMAND ----------

!cp hello.txt ../../dbfs/joe/hello.txt
