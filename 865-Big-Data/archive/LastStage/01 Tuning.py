# Databricks notebook source
# DBTITLE 1,Make Sure to Update
"""
kaggle: 
trainingAUC xxxx
xxx



TODO:
1. [DONE]proper missing value handle
2. [MB]imbalance data
3. [MB] topic
4. [Done] sentiment analytis
5. [RIGHT-NOW] note down all the auc after adding new features
6. [LAST] change the count vector values
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

# DBTITLE 1,[Don't Touch]Load Data
from pyspark.sql.functions import to_date, year, month, dayofmonth, dayofyear, weekofyear
from pyspark.sql.functions import col, lit, when, split, size
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler, SQLTransformer
from sparknlp.annotator import ContextSpellCheckerModel, LemmatizerModel
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel, DocumentNormalizer, UniversalSentenceEncoder, MultiClassifierDLModel
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import to_date, year, month, dayofmonth, dayofyear, weekofyear
from pyspark.sql.functions import col, lit, when, split, size

# Load in one of the tables
df1 = spark.sql("select * from default.video_games_5")
df2 = spark.sql("select * from default.books_5_small")
df3 = spark.sql("select * from default.home_and_kitchen_5_small")
df = df1.union(df2).union(df3)
# print((df.count(), len(df.columns)))

def resample(base_features, ratio, class_field, base_class):
    pos = base_features.filter(col(class_field)==base_class)
    neg = base_features.filter(col(class_field)!=base_class)
    total_pos = pos.count()
    total_neg = neg.count()
    fraction=float(total_pos*ratio)/float(total_neg)
    sampled = neg.sample(False,fraction)
    return sampled.union(pos)

# Resample
df = resample(df,1,'label',1)
print((df.count(), len(df.columns)))

# COMMAND ----------

# DBTITLE 1,[Don't touch]Transform Data
from pyspark.sql.functions import to_date, year, month, dayofmonth, dayofyear, weekofyear
from pyspark.sql.functions import col, lit, when, split, size

def featureEngineering(df): 
  res = df.withColumn(
    "reviewTime",
    to_date(col("reviewTime"), "M d, y")
  )

  # Dates
  res = res.withColumn('reviewTime_year', year(col('reviewTime')))
  res = res.withColumn('reviewTime_month', month(col('reviewTime')))
  res = res.withColumn('reviewTime_day', dayofmonth(col('reviewTime')))
  res = res.withColumn('reviewTime_dayofy', dayofyear(col('reviewTime')))
  res = res.withColumn('reviewTime_week_no', weekofyear(col('reviewTime')))

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
    "summaryNumberPeriod",
    
    "overall", 
    "verified"
  ])

df, featureList = featureEngineering(df)
# For our intitial modeling efforts, we are not going to use the following features
drop_list = ['unixReviewTime', 'image', 'style']
df = df.select([column for column in df.columns if column not in drop_list])
df = df.na.drop(subset=["reviewText", "label", "summary"])
print((df.count(), len(df.columns)))

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
trainingData.columns

# COMMAND ----------

# DBTITLE 1,[Don't Touch] NLP tuning
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler, SQLTransformer
from sparknlp.annotator import ContextSpellCheckerModel, LemmatizerModel
from pyspark.ml.classification import LogisticRegression
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel, DocumentNormalizer, UniversalSentenceEncoder, MultiClassifierDLModel
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
  idf = IDF(inputCol=f"{fieldname}_rawFeatures", outputCol=f"{fieldname}_idfFeatures", minDocFreq=5)

  cleaned_token_size = SQLTransformer(
      statement = f"SELECT * , size({fieldname}_token_features) AS {fieldname}_tokenSize FROM __THIS__"
  )
  
  # sentimental
#   useEmbeddings = UniversalSentenceEncoder.pretrained() \
#     .setInputCols(f"{fieldname}_document") \
#     .setOutputCol(f"{fieldname}_sentence_embeddings")
#   multiClassifierDl = MultiClassifierDLModel.pretrained() \
#       .setInputCols(f"{fieldname}_sentence_embeddings") \
#       .setOutputCol(f"{fieldname}_sentimental_classifications")
#   sentimentalFinisher = Finisher() \
#       .setInputCols([f"{fieldname}_sentimental_classifications"]) \
#       .setOutputCols([f"{fieldname}_sentimental_token_features"]) \
#       .setOutputAsArray(True) \
#       .setCleanAnnotations(False) 
#   countVectorsSentimental = CountVectorizer(
#     inputCol=f"{fieldname}_sentimental_token_features", 
#     outputCol=f"{fieldname}_sentimental_rawFeatures", 
#     vocabSize=10000, minDF=5)
  
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
    idf,
    cleaned_token_size,
#     useEmbeddings,
#     multiClassifierDl,
#     sentimentalFinisher,
#     countVectorsSentimental
  ]


# COMMAND ----------

# DBTITLE 1,Topic Modeling
# for interpertation check the bottom cell named TOPIC Lib

# Topic modeling
# lda = LDA(k=10, maxIter=10)

# COMMAND ----------

# DBTITLE 1,Control Features
# control the Features   _idfFeatures
assembler = VectorAssembler(inputCols=[
#   "summary_rawFeatures", 
  "summary_idfFeatures", 
  "summary_tokenSize", 
#   "summary_sentimental_rawFeatures",
#   "reviewText_rawFeatures",  # no idf
  "reviewText_idfFeatures", 
  "reviewText_tokenSize", 
#   "reviewText_sentimental_rawFeatures",
#   'asin', 
#   'reviewID',
#   'reviewTime',
#   'reviewerName',
#   'reviewText',
#   'reviewerID',
#   'summary',
  'overall',
  'verified',
#   'reviewTime_year',
#   'reviewTime_month',
#   'reviewTime_day',
#   'reviewTime_dayofy',
#   'reviewTime_week_no',
#   'reviewerName_Shorthand',
#   'reviewerName_isAmazon',
#   'reviewerName_capsName',
  'reviewTextHasCapsWord',
  'summaryHasCapsWord',
#   'reviewTextHasSwearWord',
#   'summaryHasSwearWord',
#   'reviewTextNumberExclamation',
#   'summaryNumberExclamation',
  'reviewTextNumberComma',
  'summaryNumberComma',
  'reviewTextNumberPeriod',
  'summaryNumberPeriod',
#   'classWeightCol'  # should be used in LR directly not as feature
  ], outputCol="features")
prePipelineStages = NLPPipe("reviewText") + NLPPipe("summary") + [assembler]

# COMMAND ----------

# DBTITLE 1,[Don't touch]Transform
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)
preprocess = Pipeline(stages=prePipelineStages).fit(trainingData)
trainingData = preprocess.transform(trainingData)
testingData = preprocess.transform(testingData)

# COMMAND ----------

# DBTITLE 1,LDA and Regression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit,CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# other Regression?
# lr = LogisticRegression(maxIter=20, weightCol="classWeightCol")
# lr = LogisticRegression(maxIter=20)  # garbage
gbt = GBTClassifier(maxIter=20)

# paramGrid = ParamGridBuilder().build()

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 5, 10])
             .addGrid(gbt.maxBins, [10, 20, 40])
             .addGrid(gbt.maxIter, [5, 10, 20])
             .build())


# Create 5-fold CrossValidator
gbcv = CrossValidator(estimator = gbt,
                      estimatorParamMaps = paramGrid,
                      evaluator = BinaryClassificationEvaluator(),
                      numFolds = 5,
                      seed = 530,
                      parallelism = 10)

# In this case the estimator is simply the linear regression.
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
# tvs = TrainValidationSplit(estimator=gbt,
#                            estimatorParamMaps=paramGrid,
#                            evaluator=BinaryClassificationEvaluator(),
#                            # 80% of the data will be used for training, 20% for validation.
#                            trainRatio=0.9, 
#                            seed=42,
#                            parallelism = 10)


# COMMAND ----------

# DBTITLE 1,[Don't Touch]Transform Training Data
# Fit the pipeline to training documents.
# pipelineFit = tvs.fit(trainingData)
pipelineFit = gbcv.fit(trainingData)

import datetime

now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
pipelineFit.save(f"file:///dbfs/joe/{now}_model")
comment = [f"Pipeline name: {now}"]
comment +=  ["Pipeline summary: <insert here>\n\n\n"]

# COMMAND ----------

# DBTITLE 1,Show Training Metrics
# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
# trainingSummary = pipelineFit.summary

# comment += ["Training Accuracy:  " + str(trainingSummary.accuracy)]
# comment += ["Training Precision: " + str(trainingSummary.precisionByLabel)]
# comment += ["Training Recall:    " + str(trainingSummary.recallByLabel)]
# comment += ["Training FMeasure:  " + str(trainingSummary.fMeasureByLabel())]
# comment += ["Training AUC:       " + str(trainingSummary.areaUnderROC)]
# comment += ["\n"]

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

# acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
# pre_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
# rec_evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
# pr_evaluator  = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# comment += ["Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions))]
# comment += ["Test Precision      = %g" % (pre_evaluator.evaluate(predictions))]
# comment += ["Test Recall         = %g" % (rec_evaluator.evaluate(predictions))]
# comment += ["Test areaUnderPR    = %g" % (pr_evaluator.evaluate(predictions))]
comment += ["Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions))]

# COMMAND ----------

# DBTITLE 1,Make Predictions on Kaggle Test Data
# Load in the tables
test_df = spark.sql("select * from default.reviews_holdout")
print((test_df.count(), len(test_df.columns)))

# COMMAND ----------

test_df, _ = featureEngineering(test_df)
submit_tranformed = preprocess.transform(test_df)
submit_predictions = pipelineFit.transform(submit_tranformed)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

lastElement=udf(lambda v:float(v[1]),FloatType())
submission = submit_predictions.select('reviewID', lastElement('probability').alias("label"))

# COMMAND ----------

submission.write.csv(f"file:///dbfs/joe/{now}_sub.csv", header=True)
now

# COMMAND ----------

comment += ["---------------------------------------------------------------------------"]
# generate a template to put it on top
for line in comment:
  print(line)

# COMMAND ----------

# DBTITLE 1,LDA Lib
# # Fit the pipeline to training documents.
# pipeline = Pipeline(stages=pipelineStages)
# pipelineFit = pipeline.fit(trainingData)
# PipelineTransformed = pipelineFit.transform(trainingData)

# # Topic modeling
# lda = LDA(k=10, maxIter=10)
# ldaModel = lda.fit(PipelineTransformed)

# transformed = ldaModel.transform(PipelineTransformed).select("topicDistribution")
# transformed.show(truncate=False)  

# ll = ldaModel.logLikelihood(PipelineTransformed)  
# lp = ldaModel.logPerplexity(PipelineTransformed) 

# topicIndices = ldaModel.describeTopics(maxTermsPerTopic = wordNumbers)  
# vocab_broadcast = sc.broadcast(vocabArray)  
# udf_to_word = udf(to_word, ArrayType(StringType()))  
  
# topics = topicIndices.withColumn("words", udf_to_word(topicIndices.termIndices))  
# topics.show(truncate=False)  
# exit()
