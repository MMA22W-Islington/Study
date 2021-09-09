# Databricks notebook source
# DBTITLE 1,Load Data
from pyspark.sql.functions import lit

# Load in one of the tables
df1 = spark.sql("select * from default.video_games_5")
df1 = df1.withColumn('category', lit("video_games"))

df2 = spark.sql("select * from default.home_and_kitchen_5_small")
df2 = df2.withColumn('category', lit("home_and_kitchen"))

df3 = spark.sql("select * from default.books_5_small")
df3 = df3.withColumn('category', lit("books"))

df = df1.union(df2).union(df3)

# Take a sample (useful for code development purposes)
# TODO: remove all the rest in this data frame when doing real analysis
df = df.sample(False, 0.15, seed=0)

df = df.cache()

print((df.count(), len(df.columns)))

# COMMAND ----------

from pyspark.sql.functions import to_date

# seems like changing the name will slow down the program
df_transformed = df.withColumn(
  "reviewTime",
  to_date(col("reviewTime"), "M d, y")
)

drop_list = [
  "reviewID",
  "reviewerID",
  "unixReviewTime", # reviewTime is the same as unixReviewTime
  # TODO: "category" is not part of Test
]
df_transformed = df_transformed.select([column for column in df.columns if column not in drop_list])

df_transformed.show()

# COMMAND ----------

df.printSchema()
