# Databricks notebook source
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

df = df.sample(False, 0.30, seed=530)

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
(trainingData, testingData) = randomSplit([0.8, 0.2], seed = 47)
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
  useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols(f"{fieldname}_document") \
    .setOutputCol(f"{fieldname}_sentence_embeddings")
  multiClassifierDl = MultiClassifierDLModel.pretrained() \
      .setInputCols(f"{fieldname}_sentence_embeddings") \
      .setOutputCol(f"{fieldname}_sentimental_classifications")
  sentimentalFinisher = Finisher() \
      .setInputCols([f"{fieldname}_sentimental_classifications"]) \
      .setOutputCols([f"{fieldname}_sentimental_token_features"]) \
      .setOutputAsArray(True) \
      .setCleanAnnotations(False) 
  countVectorsSentimental = CountVectorizer(
    inputCol=f"{fieldname}_sentimental_token_features", 
    outputCol=f"{fieldname}_sentimental_rawFeatures", 
    vocabSize=10000, minDF=5)
  
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
    useEmbeddings,
    multiClassifierDl,
    sentimentalFinisher,
    countVectorsSentimental
  ]


# COMMAND ----------

# DBTITLE 1,Control Features
# control the Features   _idfFeatures
assembler = VectorAssembler(inputCols=[
#   "summary_rawFeatures", 
  "summary_idfFeatures", 
  "summary_tokenSize", 
  "summary_sentimental_rawFeatures",
#   "reviewText_rawFeatures",  # no idf
  "reviewText_idfFeatures", 
  "reviewText_tokenSize", 
  "reviewText_sentimental_rawFeatures",
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
