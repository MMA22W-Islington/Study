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
(trainingData, testingData) = df.sample(False, 0.1, seed=0).randomSplit([0.8, 0.2], seed = 47)

# Kaggle
# (trainingData, testingData) = df, spark.sql("select * from default.reviews_holdout")

# COMMAND ----------

from pyspark.sql.functions import to_date

# seems like changing the name will slow down the program
df = df.withColumn(
  "reviewTime",
  to_date(col("reviewTime"), "M d, y")
)

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

  sqlTrans = SQLTransformer(
      statement=f"SELECT *, size({fieldname}_token_features) AS {fieldname}_TokenSize FROM __THIS__")

  return ([
    document_assembler, documentNormalizer, tokenizer, 
    spellChecker, lemmatizer, stopwords_cleaner, 
    normalizer, finisher, sqlTrans])


# COMMAND ----------

# DBTITLE 1,Review Text
pipeline_pre_1 = NLPPipe("reviewText") + NLPPipe("summary")
%time
tranformData_cleaned_p = Pipeline(stages=pipeline_pre_1).fit(trainingData)
tranformData_cleaned_p.save(f"file:///databricks/driver/pipeline_pre_1_{now}")
tranformData_cleaned = tranformData_cleaned_p.transform(trainingData)
tranformData_cleaned.show(5)

# COMMAND ----------

tranformData_cleaned.selectExpr("reviewText_token_features").show(10, False)

# COMMAND ----------

# nlp_pretrained = Pipeline.load("file:///databricks/driver/Spark_NLP_Example_all")
# eda = nlp_pretrained.transform(trainingData)
# eda.show(5)

# COMMAND ----------

# DBTITLE 1,Convert text to vector
def getVector(field):
  # hashingTF = HashingTF(inputCol="token_features", outputCol="rawFeatures", numFeatures=20)
  tf = CountVectorizer(inputCol=f"{field}_token_features", outputCol=f"{field}_rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.40)
  idf = IDF(inputCol=f"{field}_rawFeatures", outputCol=f"{field}_idfFeatures", minDocFreq=5)
  return [tf, idf]

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "summary_TokenSize", "summary_idfFeatures", "reviewText_TokenSize", "reviewText_idfFeatures"], outputCol="features")
%time
tranformData_features_p = Pipeline(stages=getVector("reviewText") + getVector("summary") + [assembler]).fit(tranformData_cleaned)
tranformData_features_p.save("file:///databricks/driver/tranformData_features_{now}")
tranformData_features = tranformData_features_p.transform(tranformData_cleaned)
tranformData_features.show(5)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html
%time
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = Pipeline(stages=[lr]).fit(tranformData_features)
lrModel.save("file:///databricks/driver/lr_model_{now}")

# COMMAND ----------

# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
# trainingSummary = lrModel.summary

# print("Training Accuracy:  " + str(trainingSummary.accuracy))
# print("Training Precision: " + str(trainingSummary.precisionByLabel))
# print("Training Recall:    " + str(trainingSummary.recallByLabel))
# print("Training FMeasure:  " + str(trainingSummary.fMeasureByLabel()))
# print("Training AUC:       " + str(trainingSummary.areaUnderROC))

# COMMAND ----------

testingDataTransform = lrModel.transform(tranformData_features_p.transform(tranformData_cleaned_p.transform(testingData)))
testingDataTransform.show(5)


# COMMAND ----------

# evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
# print('Test Area Under ROC', evaluator.evaluate(testingDataTransform))

# COMMAND ----------

testingDataTransform.write.format("csv").save(f"file:///databricks/driver/dataframe_kaggle_{now")

# COMMAND ----------

display(testingDataTransform.select('reviewID', 'prediction'))

# COMMAND ----------

pM = pip
