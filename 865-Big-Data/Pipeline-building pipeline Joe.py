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
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Lemmatizer, LemmatizerModel, SymmetricDeleteModel, ContextSpellCheckerApproach, NormalizerModel, ContextSpellCheckerModel, NorvigSweetingModel
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

df = df.cache()

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

!echo -e "I'm\t->\tI am\t\nok\t->\tjoey" > Contraction.txt
!cat Contraction.txt
# !echo Contraction.txt

# COMMAND ----------

# !realpath Contraction.txt
!cat Contraction.txt

# COMMAND ----------

# DBTITLE 1,[SKIP]Develop pipeline(Joe)
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

# convert document to array of tokens
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
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

# pick and choose what pipeline you want.
pipeline_test = [document_assembler, tokenizer, spellChecker, lemmatizer, stopwords_cleaner, normalizer, finisher]
# pipeline_test = [document_assembler, tokenizer, normalizer, spellChecker, lemmatizer, stopwords_cleaner, finisher] # move normalizer to the back


eda = Pipeline(stages=pipeline_test).fit(df).transform(df)
# .select(["reviewText", "result"])
eda.selectExpr("token_features").show(10, truncate=False)

# eda.select('sentence').show(10, truncate=False)

# COMMAND ----------

# DBTITLE 1,[SKIP]Develop pipeline(Yujun)


# pick and choose what pipeline you want.
pipeline_test = [tokenizer, stopwordsRemover, counter]

jxhdkajhdakjh
eda = Pipeline(stages=pipeline_test).fit(df).transform(df)
eda.show()

# COMMAND ----------

# DBTITLE 1,Pipeline Libs
# We'll tokenize the text using a simple RegexTokenizer
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")

# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

# TODO: insert other clearning steps here (and put into the pipeline, of course!)
# E.g., n-grams? document length?
# convert text column to spark nlp document
# document_assembler = DocumentAssembler() \
#     .setInputCol("reviewText") \
#     .setOutputCol("document")


# # convert document to array of tokens
# tokenizer = Tokenizer() \
#   .setInputCols(["document"]) \
#   .setOutputCol("token")
 
# # clean tokens 
# normalizer = Normalizer() \
#     .setInputCols(["token"]) \
#     .setOutputCol("normalized")

# # remove stopwords
# stopwords_cleaner = StopWordsCleaner()\
#       .setInputCols("normalized")\
#       .setOutputCol("cleanTokens")\
#       .setCaseSensitive(False)

# # stems tokens to bring it to root form
# stemmer = Stemmer() \
#     .setInputCols(["cleanTokens"]) \
#     .setOutputCol("stem")

# # Convert custom document structure to array of tokens.
# finisher = Finisher() \
#     .setInputCols(["stem"]) \
#     .setOutputCols(["token_features"]) \
#     .setOutputAsArray(True) \
#     .setCleanAnnotations(False)

# Vectorize the sentences using simple BOW method. Other methods are possible:
# https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
tf = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=2000, minTF=1, maxDF=0.40)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=100)

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "idfFeatures"], outputCol="features")

# Machine Learning Algorithm
ml_lr  = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)
ml_rf  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
ml_nb = NaiveBayes(smoothing=1.0, modelType="multinomial")


# COMMAND ----------

# DBTITLE 1,Split Data-Shuffle
# TODO: MAKE SURE TO DISABLE THE SAMPLE!!

# set seed for reproducibility
(trainingData, testData) = df.randomSplit([0.9, 0.1], seed = 47)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count:     " + str(testData.count()))

# COMMAND ----------

# DBTITLE 1,Train and Predict (only for feature selection and that sort)
fit_transform = [];
# TODO: This is for feature selection, not for tuning.
for name, ml in [("lr", ml_lr), ("rf", ml_rf), ("nb", ml_nb)]:
  pipeline = Pipeline(stages=pipeline_pre + [ml])
  
  pipelineFit = pipeline.fit(trainingData)
  predictions = pipelineFit.transform(testData)
  fit_transform += [(name, pipelineFit, predictions)]

# COMMAND ----------

# DBTITLE 1,Metrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import datetime

now = datetime.datetime.now()
acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
pre_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
pr_evaluator  = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

Notes = "Testing" # dates, and model name
files = []

for (name, pipelineFit, predictions) in fit_transform:
  file_name = "_".join([name,  now.strftime("%Y_%m_%d_%H_%M_%S"), Notes])
  pipelineFit.save(file_name)
  predictions.groupBy("prediction").count().show()
  files += "Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions))
  files += "Test Precision      = %g" % (pre_evaluator.evaluate(predictions))
  files += "Test Recall         = %g" % (rec_evaluator.evaluate(predictions))
  files += "Test areaUnderPR    = %g" % (pr_evaluator.evaluate(predictions))
  files += "Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions))
  files += "*************************************************************"

# COMMAND ----------

# need to do shuffling of train
# need to do auto tune
# need to dealt with imbalance
# Check Steve's code for generating for Kaggle

text_file_name = "_".join([now.strftime("%Y_%m_%d_%H_%M_%S"), Notes])
with open(text_file_name, "w") as text_file:
  for file in files:
    text_file.write(file)
