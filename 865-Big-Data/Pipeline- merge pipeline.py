# Databricks notebook source
# TODO: Save not working after restart
# TODO: dictionary
# TODO: summary?
# TODO: Sentiment https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.SentimentDetectorModel.html
# TODO: save model 
# TODO: RF Ensemble
# TODO: html anchor

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sparknlp.base import DocumentAssembler, Finisher, EmbeddingsFinisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Lemmatizer, LemmatizerModel, SymmetricDeleteModel, ContextSpellCheckerApproach, NormalizerModel, ContextSpellCheckerModel, NorvigSweetingModel, AlbertEmbeddings, DocumentNormalizer
from sparknlp.pretrained import PretrainedPipeline

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, RegexTokenizer, StopWordsRemover, VectorSizeHint
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes

from pyspark.sql.functions import lit, col
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_date, when
from pyspark.sql.functions import col, size, split
from pyspark.sql.functions import year, month, dayofmonth, dayofyear, weekofyear
import datetime

now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
now

# COMMAND ----------

# DBTITLE 1,Load Data
# Load in one of the tables
df1 = spark.sql("select * from default.video_games_5")
df1 = df1.withColumn('category', lit("video_games"))

df2 = spark.sql("select * from default.home_and_kitchen_5_small")
df2 = df2.withColumn('category', lit("home_and_kitchen"))

df3 = spark.sql("select * from default.books_5_small")
df3 = df3.withColumn('category', lit("books"))

df = df1.union(df2).union(df3)

# Test
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)
# (trainingData, testingData) = df.sample(False, 0.30, seed=0).randomSplit([0.8, 0.2], seed = 47)
# (trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)

# Kaggle
# (trainingData, testingData) = df, spark.sql("select * from default.reviews_holdout")

# COMMAND ----------

# seems like changing the name will slow down the program
df = df.withColumn(
  "reviewTime",
  to_date(col("reviewTime"), "M d, y")
)

# Dates
df = df.withColumn('reviewTime_year', year(col('reviewTime')))
df = df.withColumn('reviewTime_month', month(col('reviewTime')))
df = df.withColumn('reviewTime_day', dayofmonth(col('reviewTime')))
df = df.withColumn('reviewTime_dayofy', dayofyear(col('reviewTime')))
df = df.withColumn('reviewTime_week_no', weekofyear(col('reviewTime')))

#Reviewer Name
df = df.withColumn('reviewerName_Shorthand', when(col('reviewerName').rlike('\\. '),True).otherwise(False))
df = df.withColumn('reviewerName_isAmazon', when(col('reviewerName').rlike('Amazon'),True).otherwise(False))
df = df.withColumn('reviewerName_capsName', when(col('reviewerName').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))

# check if review contains all caps words
df = df.withColumn('reviewTextHasCapsWord', when(col('reviewText').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
df = df.withColumn('summaryHasCapsWord', when(col('summary').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
# check if review contians swear
df.withColumn('reviewTextHasSwearWord', when(col('reviewText').rlike('\\*{2,}'),True).otherwise(False))
df.withColumn('summaryHasSwearWord', when(col('summary').rlike('\\*{2,}'),True).otherwise(False))
## Number of Exclaimation
df = df.withColumn('reviewTextNumberExclamation', size(split(col('reviewText'), r"!")) - 1)
df = df.withColumn('summaryNumberExclamation', size(split(col('summary'), r"!")) - 1)
## Number of Exclaimation
df = df.withColumn('reviewTextNumberComma', size(split(col('reviewText'), r",")) - 1)
df = df.withColumn('summaryNumberComma', size(split(col('summary'), r",")) - 1)
## Number of Exclaimation
df = df.withColumn('reviewTextNumberPeriod', size(split(col('reviewText'), r"\.")) - 1)
df = df.withColumn('summaryNumberPeriod', size(split(col('summary'), r"\.")) - 1)

drop_list = [
  "asin",
  "reviewTime",
  "reviewID",
  "reviewerID",
  "unixReviewTime", # reviewTime is the same as unixReviewTime
  "category"  # TODO: "category" is not part of Test
]
df = df.select([column for column in df.columns if column not in drop_list])

# df.show()

# COMMAND ----------

# df = df.withColumn('matched',F.when(df.email.rlike('^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'),True).otherwise(False))
# ndf = df.withColumn('hasSwearWord', when(df.reviewText.rlike('\\*{2,}'),True).otherwise(False))

# ndf = df.withColumn('reviewTextNumberExclamation', size(split(df.reviewText, r"!")) - 1)

# ndf.select(['reviewText', 'NumberComma']).show(40, truncate = 220)
df.select(['reviewText', 'reviewTextNumberExclamation', 'reviewTextNumberComma', 'reviewTextNumberPeriod']).show(20, truncate = 140)

# COMMAND ----------

# DBTITLE 1,NLP Preprocessing
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
  
  # check how many caps
#   regexTokenizer = RegexTokenizer() \
#     .setInputCol(fieldname) \
#     .setOutputCol(f"{fieldname}_regexCapsToken") \
#     .setPattern("\\*+")
#     .setPattern("\b[A-Z].*?\b")

  # convert document to array of tokens
  tokenizer = Tokenizer() \
      .setInputCols([f"{fieldname}_removedHTML"]) \
      .setOutputCol(f"{fieldname}_token")

  spellChecker = ContextSpellCheckerModel.pretrained() \
      .setInputCols(f"{fieldname}_token") \
      .setOutputCol(f"{fieldname}_corrected")

  lemmatizer = LemmatizerModel.pretrained() \
      .setInputCols([f"{fieldname}_corrected"]) \
      .setOutputCol(f"{fieldname}_lemma")

  # # remove stopwords
  stopwords_cleaner = StopWordsCleaner()\
        .setInputCols(f"{fieldname}_lemma")\
        .setOutputCol(f"{fieldname}_cleanTokens")\
        .setCaseSensitive(False)

  # clean tokens , also need comtraction expand, and remove punctality
  normalizer = Normalizer() \
      .setInputCols([f"{fieldname}_cleanTokens"]) \
      .setOutputCol(f"{fieldname}_normalized")
      
  
#    \.setLowercase(True)
#       .setCleanupPatterns(["""[^\w\d\s]"""])

#   regexTokenizer = SQLTransformer(
#       statement=f"SELECT {fieldname}_regexCapsToken, {fieldname}_token_features, size({fieldname}_token_features) AS {fieldname}_TokenSize FROM __THIS__")
  
  ## sentiment
  # https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.SentimentDLModel.html
  # https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.ViveknSentimentApproach.html

  # # Convert custom document structure to array of tokens.
  finisher = Finisher() \
      .setInputCols([f"{fieldname}_normalized"]) \
      .setOutputCols([f"{fieldname}_token_features"]) \
      .setOutputAsArray(True) \
      .setCleanAnnotations(False) 
# {fieldname}_regexCapsToken,
  cleaned_token_size = SQLTransformer(
      statement = f"SELECT * , size({fieldname}_token_features) AS {fieldname}_TokenSize FROM __THIS__"
  )
#   cleaned_token_size = SQLTransformer(
#       statement=f"SELECT *, {fieldname}_token_features, size({fieldname}_token_features) AS {fieldname}_TokenSize FROM __THIS__")
  
  
#   SELECT * FROM Email Addresses
# WHERE Email Address ~* '%@chartio.com'
# regexTokenizer,
  return ([
    document_assembler, documentNormalizer, tokenizer,
    spellChecker, lemmatizer, stopwords_cleaner,
    normalizer, finisher, cleaned_token_size])


# COMMAND ----------

pipeline_obj = NLPPipe("reviewText")
%time
eda = Pipeline(stages=pipeline_obj).fit(trainingData).transform(trainingData)
# eda.show()
eda.selectExpr("reviewText").show(10, False)

# COMMAND ----------


# eda.selectExpr("reviewText_regexCapsToken").show(10, False)

# COMMAND ----------

# DBTITLE 1,Convert text to vector
def getVector(field):
  # hashingTF = HashingTF(inputCol="token_features", outputCol="rawFeatures", numFeatures=20)
  
  tf = CountVectorizer(inputCol=f"{field}_token_features", outputCol=f"{field}_rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.40)
  idf = IDF(inputCol=f"{field}_rawFeatures", outputCol=f"{field}_idfFeatures", minDocFreq=5)
  return [tf, idf]
  
#   tf = CountVectorizer(inputCol=f"{field}_token_features", outputCol=f"{field}_idfFeatures", vocabSize=10000, minDF=5)
#   return [tf]

cols = [
  "verified", "overall", "summary_TokenSize", "summary_idfFeatures", "reviewText_TokenSize", "reviewText_idfFeatures",
  'reviewTextHasCapsWord', 'summaryHasCapsWord', 'reviewTextHasSwearWord', 'summaryHasSwearWord',
  'reviewTextNumberExclamation',
  'summaryNumberExclamation',
  'reviewTextNumberComma',
  'summaryNumberComma',
  'reviewTextNumberPeriod',
  'summaryNumberPeriod',
  'reviewTime_year',
  'reviewTime_month',
  'reviewTime_day',
  'reviewTime_dayofy',
  'reviewTime_week_no',
  'reviewerName_Shorthand',
  'reviewerName_isAmazon',
  'reviewerName_capsName'
]
# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=cols, outputCol="features")

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html
%time
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)

# COMMAND ----------

# DBTITLE 1,Review Text
pipeline_obj = NLPPipe("reviewText") + NLPPipe("summary") + getVector("reviewText") + getVector("summary") + [assembler, lr]
%time
model_pipeline = Pipeline(stages=pipeline_obj).fit(trainingData)
import datetime
now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
pipelineFit.save(f"file:///databricks/driver/models/{now}")
now # name

# COMMAND ----------

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

predictions = pipelineFit.transform(testingData)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
pre_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
pr_evaluator  = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

print("Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions)))
print("Test Precision      = %g" % (pre_evaluator.evaluate(predictions)))
print("Test Recall         = %g" % (rec_evaluator.evaluate(predictions)))
print("Test areaUnderPR    = %g" % (pr_evaluator.evaluate(predictions)))
print("Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions)))
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

# Load in the tables
test_df = spark.sql("select * from default.reviews_holdout")
print((test_df.count(), len(test_df.columns)))

# COMMAND ----------

# features
test_df = test_df.withColumn(
  "reviewTime",
  to_date(col("reviewTime"), "M d, y")
)

# Dates
test_df = test_df.withColumn('reviewTime_year', year(col('reviewTime')))
test_df = test_df.withColumn('reviewTime_month', month(col('reviewTime')))
test_df = test_df.withColumn('reviewTime_day', dayofmonth(col('reviewTime')))
test_df = test_df.withColumn('reviewTime_dayofy', dayofyear(col('reviewTime')))
test_df = test_df.withColumn('reviewTime_week_no', weekofyear(col('reviewTime')))

#Reviewer Name
test_df = test_df.withColumn('reviewerName_Shorthand', when(col('reviewerName').rlike('\\. '),True).otherwise(False))
test_df = test_df.withColumn('reviewerName_isAmazon', when(col('reviewerName').rlike('Amazon'),True).otherwise(False))
test_df = test_df.withColumn('reviewerName_capsName', when(col('reviewerName').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))

# check if review contains all caps words
test_df = test_df.withColumn('reviewTextHasCapsWord', when(col('reviewText').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
test_df = test_df.withColumn('summaryHasCapsWord', when(col('summary').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
# check if review contians swear
test_df.withColumn('reviewTextHasSwearWord', when(col('reviewText').rlike('\\*{2,}'),True).otherwise(False))
test_df.withColumn('summaryHasSwearWord', when(col('summary').rlike('\\*{2,}'),True).otherwise(False))
## Number of Exclaimation
test_df = test_df.withColumn('reviewTextNumberExclamation', size(split(col('reviewText'), r"!")) - 1)
test_df = test_df.withColumn('summaryNumberExclamation', size(split(col('summary'), r"!")) - 1)
## Number of Exclaimation
test_df = test_df.withColumn('reviewTextNumberComma', size(split(col('reviewText'), r",")) - 1)
test_df = test_df.withColumn('summaryNumberComma', size(split(col('summary'), r",")) - 1)
## Number of Exclaimation
test_df = test_df.withColumn('reviewTextNumberPeriod', size(split(col('reviewText'), r"\.")) - 1)
test_df = test_df.withColumn('summaryNumberPeriod', size(split(col('summary'), r"\.")) - 1)



submit_predictions = pipelineFit.transform(test_df)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

lastElement=udf(lambda v:float(v[1]),FloatType())
submit_predictions.select('reviewID', lastElement('probability').alias("label")).display()

# COMMAND ----------

!ls models
