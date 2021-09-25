# Databricks notebook source
"""
TODO:
1. proper missing value handle
2. imbalance data
3. topic
4. sentiment analytis
5. [RIGHT-NOW] note down all the auc after adding new features
6. change the count vector values
7. [RIGHT-NOW] count vector value add idf and summaryintion of features


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
Pipeline Name: 2021_09_25_02_18_52  or 2021_09_25_02_19_04
Pipeline: regexTokenizer, stopwordsRemover, countVectors, lr
pipeline summary: Basic Steve Model

kaggle AUC: 0.72735

Training Accuracy:  0.8414306146723333
Training Precision: [0.8509137584299493, 0.6849417313747381]
Training Recall:    [0.978054907511774, 0.2177783627585299]
Training FMeasure:  [0.9100652108473223, 0.33048004539436354]
Training AUC:       0.842636473214598

Test Area Under ROC 0.8412387227224285

---------------------------------------------------------------------------
Pipeline Name; 2021_09_25_03_06_11
Pipeline: regexTokenizer, stopwordsRemover, countVectors, lr
pipeline summary: Added weight 
https://www.datatrigger.org/post/spark_3_weighted_random_forest/

kaggle AUC: 0.72431

Training Accuracy:  0.7420567701313684
Training Precision: [0.6958397646011562, 0.8166218072466968]
Training Recall:    [0.8595904654558897, 0.6246417823832509]
Training FMeasure:  [0.7690955409993384, 0.7078456777536136]
Training AUC:       0.8426555831776456

Test Area Under ROC 0.8414762990598188

---------------------------------------------------------------------------
Pipeline Name: 2021_09_25_12_46_32
Pipeline:  [regexTokenizer, stopwordsRemover, countVectors, idf, assembler] + [lr]
Pipeline Summary: added idf and assembler and make sure it is run as fast, and check if idf working

Training Accuracy:  0.8414273893063349
Training Precision: [0.8509121813978698, 0.6849123467243704]
Training Recall:    [0.9780527230945328, 0.21777038574827348]
Training FMeasure:  [0.9100633632567908, 0.3304674401854508]
Training AUC:       0.842635567782776

Test Accuracy       = 0.841263
Test Precision      = 0.820209
Test Recall         = 0.841263
Test areaUnderPR    = 0.540858
Test areaUnderROC   = 0.84124
Test Area Under ROC 0.841239327241676

---------------------------------------------------------------------------

Pipeline name: 2021_09_25_13_20_43
Pipeline object: [regexTokenizer, stopwordsRemover, countVectors, assembler] + [lr]
Pipeline summary: No IDF, add features

"reviewTime_year",
"reviewTime_month",
"reviewTime_day",
"reviewTime_dayofy",
"reviewTime_week_no",

Training Accuracy:  0.8392459667694125
Training Precision: [0.8474593999012964, 0.6861175642379443]
Training Recall:    [0.9805206776936048, 0.19436583765587576]
Training FMeasure:  [0.9091471786742109, 0.3029195435522796]
Training AUC:       0.8411366650327537


Test Accuracy       = 0.839367
Test Precision      = 0.81804
Test Recall         = 0.839367
Test areaUnderPR    = 0.536398
Test areaUnderROC   = 0.840357

---------------------------------------------------------------------------
Pipeline name: wrong
Pipeline object: [regexTokenizer, stopwordsRemover, countVectors, assembler] + [lr]
Pipeline summary: No IDF, add features

"reviewTime_year",

Training Accuracy:  0.8413657489783654
Training Precision: [0.851759243946919, 0.6770653254224016]
Training Recall:    [0.9765778045732959, 0.22415997096368268]
Training FMeasure:  [0.9099079112923153, 0.33681021670342315]
Training AUC:       0.8415424937340857


Test Accuracy       = 0.841392
Test Precision      = 0.819891
Test Recall         = 0.841392
Test areaUnderPR    = 0.539182
Test areaUnderROC   = 0.840581

---------------------------------------------------------------------------

Pipeline name: 2021_09_25_13_48_17
Pipeline object: [regexTokenizer, stopwordsRemover, countVectors, assembler] + [lr]
Pipeline summary: No IDF, add features

"reviewerName_Shorthand",
"reviewerName_isAmazon",
"reviewerName_capsName",


Training Accuracy:  0.8414854458943062
Training Precision: [0.8509848583914436, 0.6850559314925289]
Training Recall:    [0.978019519952467, 0.2182450178585317]
Training FMeasure:  [0.9100905525927865, 0.33103041485805895]
Training AUC:       0.8428357728999406


Test Accuracy       = 0.841299
Test Precision      = 0.820262
Test Recall         = 0.841299
Test areaUnderPR    = 0.541338
Test areaUnderROC   = 0.841344

---------------------------------------------------------------------------

Pipeline name: 2021_09_25_14_28_52
Pipeline object: [regexTokenizer, stopwordsRemover, countVectors, assembler] + [lr]
Pipeline summary: No IDF, add features

"reviewerName_Shorthand",
"reviewerName_isAmazon",
"reviewerName_capsName",
"reviewTextHasCapsWord",
"summaryHasCapsWord",
"reviewTextHasSwearWord",
"summaryHasSwearWord",


Training Accuracy:  0.8415879408582556
Training Precision: [0.8511668357622764, 0.684844902149902]
Training Recall:    [0.9778731639973088, 0.21948344870084416]
Training FMeasure:  [0.9101312239052612, 0.3324281955943251]
Training AUC:       0.8428372046535535


Test Accuracy       = 0.841481
Test Precision      = 0.820516
Test Recall         = 0.841481
Test areaUnderPR    = 0.541822
Test areaUnderROC   = 0.841414

---------------------------------------------------------------------------

Pipeline Name; 2021_09_25_03_06_11
Pipeline: regexTokenizer, stopwordsRemover, countVectors, lr
pipeline summary: remove weight , the previous one is def a bust, need to 
find a different way, this version focus on adding more features.

interal test:

1------>
"reviewTime_year",
"reviewTime_month",
"reviewTime_day",
"reviewTime_dayofy",
"reviewTime_week_no",
"reviewerName_Shorthand",
"reviewerName_isAmazon",
"reviewerName_capsName",
"reviewTextHasCapsWord",
"summaryHasCapsWord",
"reviewTextHasSwearWord",
"summaryHasSwearWord",
"reviewTextNumberExclamation",
"summaryNumberExclamation",
"reviewTextNumberComma",
"summaryNumberComma",
"reviewTextNumberPeriod",
"summaryNumberPeriod"
Pipeline name: 2021_09_25_13_34_06
Pipeline object: <insert here>
Pipeline summary: <insert here>



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
from pyspark.sql.functions import to_date, year, month, dayofmonth, dayofyear, weekofyear
from pyspark.sql.functions import col, lit, when, split, size

def featureEngineering(df): 
  res = df.withColumn(
    "reviewTime",
    to_date(col("reviewTime"), "M d, y")
  )

  # Dates
#   res = res.withColumn('reviewTime_year', year(col('reviewTime')))
#   res = res.withColumn('reviewTime_month', month(col('reviewTime')))
#   res = res.withColumn('reviewTime_day', dayofmonth(col('reviewTime')))
#   res = res.withColumn('reviewTime_dayofy', dayofyear(col('reviewTime')))
#   res = res.withColumn('reviewTime_week_no', weekofyear(col('reviewTime')))

#   #Reviewer Name
  res = res.withColumn('reviewerName_Shorthand', when(col('reviewerName').rlike('\\. '),True).otherwise(False))
  res = res.withColumn('reviewerName_isAmazon', when(col('reviewerName').rlike('Amazon'),True).otherwise(False))
  res = res.withColumn('reviewerName_capsName', when(col('reviewerName').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))

#   # check if review contains all caps words
  res = res.withColumn('reviewTextHasCapsWord', when(col('reviewText').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
  res = res.withColumn('summaryHasCapsWord', when(col('summary').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
#   # check if review contians swear
  res = res.withColumn('reviewTextHasSwearWord', when(col('reviewText').rlike('\\*{2,}'),True).otherwise(False))
  res = res.withColumn('summaryHasSwearWord', when(col('summary').rlike('\\*{2,}'),True).otherwise(False))
  ## Number of Exclaimation
  res = res.withColumn('reviewTextNumberExclamation', size(split(col('reviewText'), r"!")) - 1)
  res = res.withColumn('summaryNumberExclamation', size(split(col('summary'), r"!")) - 1)
  ## Number of Exclaimation
  res = res.withColumn('reviewTextNumberComma', size(split(col('reviewText'), r",")) - 1)
  res = res.withColumn('summaryNumberComma', size(split(col('summary'), r",")) - 1)
  ## Number of Exclaimation
  res = res.withColumn('reviewTextNumberPeriod', size(split(col('reviewText'), r"\.")) - 1)
  res = res.withColumn('summaryNumberPeriod', size(split(col('summary'), r"\.")) - 1)
  
  return (res, [
#     "reviewTime_year",
#     "reviewTime_month",
#     "reviewTime_day",
#     "reviewTime_dayofy",
#     "reviewTime_week_no",
    "reviewerName_Shorthand",
    "reviewerName_isAmazon",
    "reviewerName_capsName",
    "reviewTextHasCapsWord",
    "summaryHasCapsWord",
    "reviewTextHasSwearWord",
    "summaryHasSwearWord",
    "reviewTextNumberExclamation",
    "summaryNumberExclamation",
    "reviewTextNumberComma",
    "summaryNumberComma",
    "reviewTextNumberPeriod",
    "summaryNumberPeriod"
  ])

df, featureList = featureEngineering(df)
# For our intitial modeling efforts, we are not going to use the following features
drop_list = ['overall', 'summary', 'asin', 'reviewID', 'reviewerID', 'summary', 'unixReviewTime','reviewTime', 'image', 'style', 'verified', 'reviewerName']
df = df.select([column for column in df.columns if column not in drop_list])
df = df.na.drop(subset=["reviewText", "label"])


# class Weight - garbage
# import pandas as pd
# counts = df.groupBy('label').count().toPandas()
# # Counts
# count_fraud = counts[counts['label']==1]['count'].values[0]
# count_total = counts['count'].sum()
# # Weights
# c = 2
# weight_fraud = count_total / (c * count_fraud)
# weight_no_fraud = count_total / (c * (count_total - count_fraud))
# df = df.withColumn("classWeightCol", when(col("label") ==1, weight_fraud).otherwise(weight_no_fraud))



# Split
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)

# COMMAND ----------

# DBTITLE 1,Create a Data Transformation/Preprocessing Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression


# We'll tokenize the text using a simple RegexTokenizer
regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")


# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")


# Vectorize the sentences using simple BOW method. Other methods are possible:
# https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
countVectors = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=10000, minDF=5)


# IDF
idf = IDF(inputCol=f"rawFeatures", outputCol=f"idfFeatures", minDocFreq=5)


# Check everything seems ok
# df.select('label', 'classWeightCol').where(col('label')==1).show(3)


# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html

# lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0, weightCol="classWeightCol") # garbage
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)


# Build up the pipeline
cols = ["rawFeatures"] + featureList
assembler = VectorAssembler(inputCols=cols, outputCol="features")

pipelineStages = [regexTokenizer, stopwordsRemover, countVectors, assembler] + [lr]

# COMMAND ----------

# DBTITLE 1,Transform Training Data
# Fit the pipeline to training documents.
pipeline = Pipeline(stages=pipelineStages)
pipelineFit = pipeline.fit(trainingData)
import datetime
now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
pipelineFit.save(f"file:///databricks/driver/models/{now}")
comment = [f"Pipeline name: {now}"]
comment +=  ["Pipeline object: <insert here>"]
comment +=  ["Pipeline summary: <insert here>\n\n\n"]

# COMMAND ----------

# import datetime
# now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# pipelineFit.save(f"file:///databricks/driver/models/{now}") # need now

# COMMAND ----------

# DBTITLE 1,Show Training Metrics
# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = pipelineFit.stages[-1].summary

comment += ["Training Accuracy:  " + str(trainingSummary.accuracy)]
comment += ["Training Precision: " + str(trainingSummary.precisionByLabel)]
comment += ["Training Recall:    " + str(trainingSummary.recallByLabel)]
comment += ["Training FMeasure:  " + str(trainingSummary.fMeasureByLabel())]
comment += ["Training AUC:       " + str(trainingSummary.areaUnderROC)]
comment += ["\n"]

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
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
pre_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
pr_evaluator  = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

comment += ["Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions))]
comment += ["Test Precision      = %g" % (pre_evaluator.evaluate(predictions))]
comment += ["Test Recall         = %g" % (rec_evaluator.evaluate(predictions))]
comment += ["Test areaUnderPR    = %g" % (pr_evaluator.evaluate(predictions))]
comment += ["Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions))]

# COMMAND ----------

# DBTITLE 1,Make Predictions on Kaggle Test Data
# Load in the tables
test_df = spark.sql("select * from default.reviews_holdout")
print((test_df.count(), len(test_df.columns)))

# COMMAND ----------

test_df, _ = featureEngineering(test_df)
submit_predictions = pipelineFit.transform(test_df)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

lastElement=udf(lambda v:float(v[1]),FloatType())
submit_predictions.select('reviewID', lastElement('probability').alias("label")).display()

# COMMAND ----------

!ls models

# COMMAND ----------

comment += ["---------------------------------------------------------------------------"]
# generate a template to put it on top
for line in comment:
  print(line)
