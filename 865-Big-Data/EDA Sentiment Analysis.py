# Databricks notebook source
from pyspark.ml import Pipeline
# from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler, IDF
# from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sparknlp.base import DocumentAssembler, Finisher, EmbeddingsFinisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Lemmatizer, LemmatizerModel, SymmetricDeleteModel, ContextSpellCheckerApproach, NormalizerModel, ContextSpellCheckerModel, NorvigSweetingModel, AlbertEmbeddings, DocumentNormalizer
from sparknlp.pretrained import PretrainedPipeline

# from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, RegexTokenizer, StopWordsRemover, VectorSizeHint
# from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes

from pyspark.sql.functions import lit, col
from pyspark.sql.functions import udf
# from pyspark.ml.feature import SQLTransformer
# from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import to_date, when
from pyspark.sql.functions import size, split
from pyspark.sql.functions import year, month, dayofmonth, dayofyear, weekofyear
import datetime

from pyspark.sql.functions import lit
from pyspark.sql.functions import col
import pyspark.sql.functions as F

now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
now

# COMMAND ----------

# Load in one of the tables
df1 = spark.sql("select * from default.video_games_5")
df1 = df1.withColumn('category', lit("video_games"))

df2 = spark.sql("select * from default.home_and_kitchen_5_small")
df2 = df2.withColumn('category', lit("home_and_kitchen"))

df3 = spark.sql("select * from default.books_5_small")
df3 = df3.withColumn('category', lit("books"))

df = df1.union(df2).union(df3)

# COMMAND ----------

df = df.withColumn('reviewTime_year', year(col('reviewTime')))
df = df.withColumn('reviewTime_month', month(col('reviewTime')))
df = df.withColumn('reviewTime_day', dayofmonth(col('reviewTime')))
df = df.withColumn('reviewTime_dayofy', dayofyear(col('reviewTime')))
df = df.withColumn('reviewTime_week_no', weekofyear(col('reviewTime')))

# #Reviewer Name
# df = df.withColumn('reviewerName_Shorthand', when(col('reviewerName').rlike('\\. '),True).otherwise(False))
# df = df.withColumn('reviewerName_isAmazon', when(col('reviewerName').rlike('Amazon'),True).otherwise(False))
# df = df.withColumn('reviewerName_capsName', when(col('reviewerName').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))

# # check if review contains all caps words
# df = df.withColumn('reviewTextHasCapsWord', when(col('reviewText').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
# df = df.withColumn('summaryHasCapsWord', when(col('summary').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
# # check if review contians swear
# df = df.withColumn('reviewTextHasSwearWord', when(col('reviewText').rlike('\\*{2,}'),True).otherwise(False))
# df = df.withColumn('summaryHasSwearWord', when(col('summary').rlike('\\*{2,}'),True).otherwise(False))
# ## Number of Exclaimation
# df = df.withColumn('reviewTextNumberExclamation', size(split(col('reviewText'), r"!")) - 1)
# df = df.withColumn('summaryNumberExclamation', size(split(col('summary'), r"!")) - 1)
# ## Number of Exclaimation
# df = df.withColumn('reviewTextNumberComma', size(split(col('reviewText'), r",")) - 1)
# df = df.withColumn('summaryNumberComma', size(split(col('summary'), r",")) - 1)
# ## Number of Exclaimation
# df = df.withColumn('reviewTextNumberPeriod', size(split(col('reviewText'), r"\.")) - 1)
# df = df.withColumn('summaryNumberPeriod', size(split(col('summary'), r"\.")) - 1)

# COMMAND ----------

# Test
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)
# (trainingData, testingData) = df.sample(False, 0.30, seed=0).randomSplit([0.8, 0.2], seed = 47)
# (trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)

# Kaggle
# (trainingData, testingData) = df, spark.sql("select * from default.reviews_holdout")

# COMMAND ----------

df.display(2)

# COMMAND ----------

drop_list = [
  "asin",
  "reviewID",
  "reviewerID",
  "unixReviewTime", # reviewTime is the same as unixReviewTime
  "category"  # TODO: "category" is not part of Test
]
df = df.select([column for column in df.columns if column not in drop_list])
df.display()

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofmonth, dayofyear, weekofyear


# temp = df.withColumn('reviewTime_year', year(col('reviewTime')))

temp = df.withColumn('reviewerName_Shorthand', when(col('reviewerName').rlike('\\. '),True).otherwise(False))
temp = temp.withColumn('reviewerName_isAmazon', when(col('reviewerName').rlike('Amazon'),True).otherwise(False))
temp = temp.withColumn('reviewerName_capsName', when(col('reviewerName').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))

temp.select([
    'reviewerName',
    'reviewerName_Shorthand',
    'reviewerName_isAmazon',
    'reviewerName_capsName'
    ]).show(120, truncate=20)

# COMMAND ----------

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
eda = Pipeline(stages=pipeline_obj).fit(df).transform(df)
# eda.show()
eda.selectExpr("reviewText").show(10, False)

# COMMAND ----------

eda.columns

# COMMAND ----------

review = eda.select("reviewText_token_features")
review.display()

# COMMAND ----------

counts = eda.select(F.explode('reviewText_token_features').alias('col')).groupBy('col').count().sort(F.desc('count')).collect()

# COMMAND ----------

cols = [
  "summary", "reviewText",
  "verified", "overall", "summary_TokenSize",  "reviewText_TokenSize", 
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

eda.select(cols).show(20, truncate=20)

# COMMAND ----------

# DBTITLE 1,nltk polarized score
import nltk
nltk.download('vader_lexicon')

# COMMAND ----------

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()

# COMMAND ----------


sent.polarity_scores(df['reviewText'])

df = df.withColumn('reviewText', polarity_scores('reviewText'))

# COMMAND ----------

# DBTITLE 1,Sentiment DL Model - User pretrained twitter model
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")
useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")
sentiment = SentimentDLModel.pretrained("sentimentdl_use_twitter") \
    .setInputCols(["sentence_embeddings"]) \
    .setThreshold(0.7) \
    .setOutputCol("sentiment")
pipeline = Pipeline().setStages([
    documentAssembler,
    useEmbeddings,
    sentiment
])

# COMMAND ----------

result = pipeline.fit(df).transform(df)
result.select("reviewText", "sentiment.result").show(truncate=False)

# COMMAND ----------

result.display(2)

# COMMAND ----------

# extract results from "sentiments" column
sent_result = result\
.selectExpr("reviewText","explode(sentiment) sentiments")\
.selectExpr("reviewText","sentiments.result result")\
.createOrReplaceTempView("result_tbl_")

# COMMAND ----------

# sentiment labels in training data are float, so we map them to 
# categorical classes
spark.sql("""
    SELECT
        reviewText,
        CASE WHEN SentimentHeadline>0 THEN 'positive' 
        WHEN SentimentHeadline<0 THEN 'negative'
        ELSE 'neutral'
        END AS label,
        result
    FROM
    result_tbl_""").createOrReplaceTempView("result_tbl")

# COMMAND ----------

sent_result.show(2)

# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")
lemmatizer = Lemmatizer() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma") \
    .setDictionary("lemmas_small.txt", "->", "\t")
sentimentDetector = SentimentDetector() \
    .setInputCols(["lemma", "document"]) \
    .setOutputCol("sentimentScore") \
    .setDictionary("default-sentiment-dict.txt", ",", ReadAs.TEXT)
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    lemmatizer,
    sentimentDetector,
])

# COMMAND ----------

result = pipeline.fit(df).transform(df)
result.selectExpr("sentimentScore.result").show(truncate=False)

# COMMAND ----------

pip uninstall -nltk

# COMMAND ----------


