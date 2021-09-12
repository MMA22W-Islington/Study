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

print((df.count(), len(df.columns)))

# Take a sample (useful for code development purposes)
df_sample = df.sample(False, 0.15, seed=0)

df = df.cache()

print((df_sample.count(), len(df_sample.columns)))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# DBTITLE 1,Describe Data
# Let's look at some quick summary statistics
df.describe().show()

# COMMAND ----------

# Let's look at df1 - video_games
df1.describe().show()

# COMMAND ----------

# Let's look at df2 - home_and_kitchen
df2.describe().show()

# COMMAND ----------

# Let's look at df3 - books
df3.describe().show()

# COMMAND ----------

display(df.groupBy("category").count().orderBy("category"))

# COMMAND ----------

from pyspark.sql.functions import col
display(df.groupBy("overall").count().orderBy("overall"))

# COMMAND ----------

# The most common product IDs
display(df.groupBy("asin").count().orderBy(col("count").desc()).head(50))

# COMMAND ----------

display(df.groupBy("label").count().orderBy("label"))

# COMMAND ----------

import pyspark.sql.functions as func

df_transformed = df.withColumn("reviewTime", func.to_timestamp("reviewTime_trans", "MM/dd/yyyy hh:mm:ss aaa"))

# COMMAND ----------

df_transformed = df.withColumn(
  "reviewTime",
  to_date(("reviewTime"), "M d, y")
)

# COMMAND ----------

# DBTITLE 1,Create a Data Transformation Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import functions as f

# We'll tokenize the text using a simple RegexTokenizer
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")

# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

pipeline = Pipeline(stages=[tokenizer, stopwordsRemover])

pipelineFit = pipeline.fit(df)
df = pipelineFit.transform(df)

# COMMAND ----------

# DBTITLE 1,Get Term Frequencies
counts = df.select(f.explode('filtered').alias('col')).groupBy('col').count().sort(f.desc('count')).collect()
display(counts)

# COMMAND ----------

df = df.na.drop(subset=["reviewText", "label"])
df.show(5)
print((df.count(), len(df.columns)))

# COMMAND ----------

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier


# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")


# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
 
# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# stems tokens to bring it to root form
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Generate Term Frequency
tf = CountVectorizer(inputCol="token_features", outputCol="rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.40)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=5)

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "idfFeatures"], outputCol="features")

# Machine Learning Algorithm
#ml_alg  = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)
ml_alg  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)

nlp_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            stemmer, 
            finisher,
            tf,
            idf,
            assembler,
            ml_alg])

# COMMAND ----------

drop_list = ['overall', 'summary', 'asin', 'reviewID', 'reviewerID', 'summary', 'unixReviewTime','reviewTime', 'image', 'style', 'verified', 'reviewerName']
df = df.select([column for column in df.columns if column not in drop_list])
df.show(5)
print((df.count(), len(df.columns)))

# COMMAND ----------

# set seed for reproducibility
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testingData.count()))

# COMMAND ----------

# Fit the pipeline to training documents.
pipelineFit = nlp_pipeline.fit(trainingData)
trainingDataTransformed = pipelineFit.transform(trainingData)
trainingDataTransformed.show(5)

# COMMAND ----------


