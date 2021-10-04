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
8. [Done]"verified", "overall",


For Imbalance
+-----+-------+
|label|  count|
+-----+-------+
|    1| 626166|
|    0|2861165|
+-----+-------+
"""


# COMMAND ----------



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
#     "reviewerName_Shorthand",
#     "reviewerName_isAmazon",
#     "reviewerName_capsName",
    "reviewTextHasCapsWord",
#     "summaryHasCapsWord",
#     "reviewTextHasSwearWord",
#     "summaryHasSwearWord",
#     "reviewTextNumberExclamation",
#     "summaryNumberExclamation",
#     "reviewTextNumberComma",
#     "summaryNumberComma",
#     "reviewTextNumberPeriod",
#     "summaryNumberPeriod",
    
    "overall", 
    "verified"
  ])

df, featureList = featureEngineering(df)
# For our intitial modeling efforts, we are not going to use the following features
drop_list = ['asin', 'reviewID', 'unixReviewTime','reviewTime', 'image', 'style', 'reviewerName']
df = df.select([column for column in df.columns if column not in drop_list])
df = df.na.drop(subset=["reviewText", "label", "summary"])
print((df.count(), len(df.columns)))

# class Weight - garbage
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
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler, SQLTransformer
from sparknlp.annotator import ContextSpellCheckerModel, LemmatizerModel
from pyspark.ml.classification import LogisticRegression
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel, DocumentNormalizer
from pyspark.ml.clustering import LDA
# , SymmetricDeleteModel, ContextSpellCheckerApproach, NormalizerModel, ContextSpellCheckerModel, NorvigSweetingModel, AlbertEmbeddings, DocumentNormalizer

def NLPPipe(fieldname):
  document_assembler = DocumentAssembler() \
      .setInputCol(fieldname) \
      .setOutputCol(f"{fieldname}_document")

  documentNormalizer = DocumentNormalizer() \
      .setInputCols(f"{fieldname}_document") \
      .setOutputCol(f"{fieldname}_removedHTML") \
      .setAction("clean") \
      .setPatterns(["<[^>]*>"]) \
      .setReplacement(" ") \
      .setPolicy("pretty_all") \
      .setLowercase(True) \
      .setEncoding("UTF-8")

  # convert document to array of tokens
  tokenizer = Tokenizer() \
      .setInputCols([f"{fieldname}_removedHTML"]) \
      .setOutputCol(f"{fieldname}_token")

#   spellChecker = ContextSpellCheckerModel.pretrained() \
#       .setInputCols(f"{fieldname}_token") \
#       .setOutputCol(f"{fieldname}_corrected")


#       .setInputCols([f"{fieldname}_corrected"]) \
  lemmatizer = LemmatizerModel.pretrained() \
      .setInputCols([f"{fieldname}_token"]) \
      .setOutputCol(f"{fieldname}_lemma")

  # remove stopwords
  stopwords_cleaner = StopWordsCleaner()\
        .setInputCols(f"{fieldname}_lemma")\
        .setOutputCol(f"{fieldname}_cleanTokens")\
        .setCaseSensitive(False)

  # clean tokens , also need comtraction expand, and remove punctality
  normalizer = Normalizer() \
      .setInputCols([f"{fieldname}_cleanTokens"]) \
      .setOutputCol(f"{fieldname}_normalized") \
      .setLowercase(True) \
      .setCleanupPatterns(["""[^\w\d\s]"""])
  
  ## sentiment
  # https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.SentimentDLModel.html
  # https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.ViveknSentimentApproach.html

  # # Convert custom document structure to array of tokens.
  finisher = Finisher() \
      .setInputCols([f"{fieldname}_normalized"]) \
      .setOutputCols([f"{fieldname}_token_features"]) \
      .setOutputAsArray(True) \
      .setCleanAnnotations(False) 


  # Vectorize the sentences using simple BOW method. Other methods are possible:
  # https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
  countVectors = CountVectorizer(inputCol=f"{fieldname}_token_features", outputCol=f"{fieldname}_rawFeatures", vocabSize=10000, minDF=5)


  # IDF
#   idf = IDF(inputCol=f"rawFeatures", outputCol=f"idfFeatures", minDocFreq=5)


#   cleaned_token_size = SQLTransformer(
#       statement = f"SELECT * , size({fieldname}_filtered) AS {fieldname}_tokenSize FROM __THIS__"
#   )
  cleaned_token_size = SQLTransformer(
      statement = f"SELECT * , size({fieldname}_token_features) AS {fieldname}_tokenSize FROM __THIS__"
  )
  
  return [
    document_assembler, 
    documentNormalizer, 
    tokenizer, 
#     spellChecker,
    lemmatizer,
    stopwords_cleaner,
    normalizer,
    finisher,
    countVectors,
    cleaned_token_size], [f"{fieldname}_rawFeatures", f"{fieldname}_tokenSize"]


# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html

# lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0, weightCol="classWeightCol")
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)  # garbage


# Topic modeling
lda = LDA(k=10, maxIter=10)

# Build up the pipeline
cols = featureList
pipelineObj = []

for field in [ "reviewText", "summary"]: #, "summary"
  nlpPipe, output = NLPPipe(field)
  cols += output
  pipelineObj += nlpPipe
  
assembler = VectorAssembler(inputCols=cols, outputCol="features")

pipelineStages = pipelineObj + [assembler, lda]

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

ldaPipeline = pipelineFit.stages[-1]
ldaPipeline

# COMMAND ----------

#ldaPipeline.describeTopics().show()
ldaPipeline.transform(trainingData).show()
#ldaPipeline.logLikelihood(pipelineFit)

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
!mkdir answers
!ls

# COMMAND ----------

comment += ["---------------------------------------------------------------------------"]
# generate a template to put it on top
for line in comment:
  print(line)
