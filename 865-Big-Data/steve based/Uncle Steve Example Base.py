# Databricks notebook source
"""
TODO:
1. proper missing value handle
2. imbalance data
3. topic
4. sentiment analytis
5. [RIGHT-NOW] note down all the auc after adding new features


For Imbalance
+-----+-------+
|label|  count|
+-----+-------+
|    1| 626166|
|    0|2861165|
+-----+-------+
"""


# COMMAND ----------

"""
Pipeline: regexTokenizer, stopwordsRemover, countVectors, lr
Pipeline Name: 2021_09_25_02_18_52  or 2021_09_25_02_19_04
pipeline summary: Basic Steve Model

kaggle AUC: 0.72735

Training Accuracy:  0.8414306146723333
Training Precision: [0.8509137584299493, 0.6849417313747381]
Training Recall:    [0.978054907511774, 0.2177783627585299]
Training FMeasure:  [0.9100652108473223, 0.33048004539436354]
Training AUC:       0.842636473214598

Test Area Under ROC 0.8412387227224285

---------------------------------------------------------------------------

Pipeline: regexTokenizer, stopwordsRemover, countVectors, lr
pipeline summary: Add weight 
https://www.datatrigger.org/post/spark_3_weighted_random_forest/

<Please Follow above Format>

---------------------------------------------------------------------------

<Please Follow above Format>

---------------------------------------------------------------------------

<Please Follow above Format>

---------------------------------------------------------------------------

<Please Follow above Format>

---------------------------------------------------------------------------


"""

# COMMAND ----------

# DBTITLE 1,Load Data
# Load in one of the tables
df1 = spark.sql("select * from default.video_games_5")
df2 = spark.sql("select * from default.books_5_small")
df3 = spark.sql("select * from default.home_and_kitchen_5_small")
df = df1.union(df2).union(df3)
print((df.count(), len(df.columns)))

# COMMAND ----------

# DBTITLE 1,Data Wrangling/Prep
# For our intitial modeling efforts, we are not going to use the following features
drop_list = ['overall', 'summary', 'asin', 'reviewID', 'reviewerID', 'summary', 'unixReviewTime','reviewTime', 'image', 'style', 'verified', 'reviewerName']
df = df.select([column for column in df.columns if column not in drop_list])
df = df.na.drop(subset=["reviewText", "label"])




# class Weight
from pyspark.sql.functions import col, lit, when
import pandas as pd
counts = df.groupBy('label').count().toPandas()
# Counts
count_fraud = counts[counts['label']==1]['count'].values[0]
count_total = counts['count'].sum()
# Weights
c = 2
weight_fraud = count_total / (c * count_fraud)
weight_no_fraud = count_total / (c * (count_total - count_fraud))
df = df.withColumn("classWeightCol", when(col("label") ==1, weight_fraud).otherwise(weight_no_fraud))



# Split
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)

# COMMAND ----------

# DBTITLE 1,Create a Data Transformation/Preprocessing Pipeline
# In Spark's MLLib, it's considered good practices to combine all the preprocessing steps into a pipeline.
# That way, you can run the same steps on both the training data, and testing data and beyond (new data)
# without copying and pasting any code.

# It is possible to run all of these steps one-by-one, outside of a Pipeline, if desired. But that's
# not how I am going to do it here.

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# We'll tokenize the text using a simple RegexTokenizer
regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")


# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")


# Vectorize the sentences using simple BOW method. Other methods are possible:
# https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

# Check everything seems ok
# df.select('label', 'classWeightCol').where(col('label')==1).show(3)

from pyspark.ml.classification import LogisticRegression

# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0, weightCol="classWeightCol")




pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, lr])


# COMMAND ----------

# DBTITLE 1,Transform Training Data
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(trainingData)
import datetime
now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
pipelineFit.save(f"file:///databricks/driver/models/{now}")
now # name

# COMMAND ----------

# import datetime
# now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# pipelineFit.save(f"file:///databricks/driver/models/{now}") # need now

# COMMAND ----------

# DBTITLE 1,Show Training Metrics
# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = pipelineFit.stages[-1].summary

print("Training Accuracy:  " + str(trainingSummary.accuracy))
print("Training Precision: " + str(trainingSummary.precisionByLabel))
print("Training Recall:    " + str(trainingSummary.recallByLabel))
print("Training FMeasure:  " + str(trainingSummary.fMeasureByLabel()))
print("Training AUC:       " + str(trainingSummary.areaUnderROC))

# move to next cell to check

# trainingSummary.roc.show()

# Obtain the objective per iteration
# objectiveHistory = trainingSummary.objectiveHistory
# for objective in objectiveHistory:
#     print(objective)

# COMMAND ----------

# DBTITLE 1,Transform Testing Data
predictions = pipelineFit.transform(testingData)

# COMMAND ----------

# DBTITLE 1,Use Model to Predict Test Data; Evaluate
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

# DBTITLE 1,Make Predictions on Kaggle Test Data
# Load in the tables
test_df = spark.sql("select * from default.reviews_holdout")
print((test_df.count(), len(test_df.columns)))

# COMMAND ----------

test_df = test_df.withColumn("classWeightCol", when(col("label") ==1, weight_fraud).otherwise(weight_no_fraud))
submit_predictions = pipelineFit.transform(test_df)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

lastElement=udf(lambda v:float(v[1]),FloatType())
submit_predictions.select('reviewID', lastElement('probability').alias("label")).display()

# COMMAND ----------

!ls models
