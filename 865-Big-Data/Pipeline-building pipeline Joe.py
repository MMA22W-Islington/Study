# Databricks notebook source
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

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Lemmatizer, LemmatizerModel, SymmetricDeleteModel, ContextSpellCheckerApproach, NormalizerModel, ContextSpellCheckerModel, NorvigSweetingModel, AlbertEmbeddings, DocumentNormalizer
from sparknlp.pretrained import PretrainedPipeline

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, RegexTokenizer, StopWordsRemover, VectorSizeHint
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes

from pyspark.sql.functions import lit, col

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

# Take a sample (useful for code development purposes)
# TODO: remove all the rest in this data frame when doing real analysis
df = df.sample(False, 0.30, seed=0)

# df = df.cache()

# print((df.count(), len(df.columns)))

# COMMAND ----------

from pyspark.sql.functions import to_date

# seems like changing the name will slow down the program
df = df.withColumn(
  "reviewTime",
  to_date(col("reviewTime"), "M d, y")
)

drop_list = [
  "reviewID",
  "reviewerID",
  "unixReviewTime", # reviewTime is the same as unixReviewTime
  "category"  # TODO: "category" is not part of Test
]
df = df.select([column for column in df.columns if column not in drop_list])

df.show()

# COMMAND ----------

# DBTITLE 1,[SKIP]Develop pipeline(Joe)
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

documentNormalizer = DocumentNormalizer() \
    .setInputCols("document") \
    .setOutputCol("removedHTML") \
    .setAction("clean") \
    .setPatterns(["<[^>]*>"]) \
    .setReplacement(" ") \
    .setPolicy("pretty_all") \
    .setLowercase(True) \
    .setEncoding("UTF-8")

# convert document to array of tokens
tokenizer = Tokenizer() \
    .setInputCols(["removedHTML"]) \
    .setOutputCol("token")

spellChecker = ContextSpellCheckerModel.pretrained() \
    .setInputCols("token") \
    .setOutputCol("corrected")

lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["corrected"]) \
    .setOutputCol("lemma")
  
# # remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("lemma")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# clean tokens , also need comtraction expand, and remove punctality
normalizer = Normalizer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("normalized") \
    .setLowercase(True) \
    .setCleanupPatterns(["""[^\w\d\s]"""])

## sentiment
# https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.SentimentDLModel.html
# https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.ViveknSentimentApproach.html

# # Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["normalized"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False) 

sqlTrans = SQLTransformer(
    statement="SELECT *, size(token_features) AS reviewTextTokenSize FROM __THIS__")

# pick and choose what pipeline you want.
pipeline_test = [document_assembler, documentNormalizer, tokenizer, spellChecker, lemmatizer, stopwords_cleaner, normalizer, finisher, sqlTrans]
# pipeline_test = [document_assembler, tokenizer, normalizer, spellChecker, lemmatizer, stopwords_cleaner, finisher] # move normalizer to the back


eda = Pipeline(stages=pipeline_test).fit(df).transform(df)
eda.selectExpr("token_features").show(10, truncate=1000)

# COMMAND ----------

eda.selectExpr("reviewTextTokenSize").show(10, truncate=1000)
