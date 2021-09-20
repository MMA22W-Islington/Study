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

# Take a sample (useful for code development purposes)
# TODO: remove all the rest in this data frame when doing real analysis
df = df.sample(False, 0.01, seed=0)

(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)
# print((df.count(), len(df.columns)))

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

pipeline_pre_1 = [
  document_assembler, documentNormalizer, tokenizer, 
  spellChecker, lemmatizer, stopwords_cleaner, 
  normalizer, finisher, sqlTrans]

%time
tranformData_cleaned_p = Pipeline(stages=pipeline_pre_1).fit(trainingData)
tranformData_cleaned_p.save(f"file:///databricks/driver/pipeline_pre_1_{now}")
tranformData_cleaned = tranformData_cleaned_p.transform(trainingData)
tranformData_cleaned.show(5)

# COMMAND ----------

# DBTITLE 1,Convert text to vector
hashingTF = HashingTF(inputCol="token_features", outputCol="rawFeatures", numFeatures=20)
tf = CountVectorizer(inputCol="token_features", outputCol="rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.40)


# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=5)

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "reviewTextTokenSize", "idfFeatures"], outputCol="features")

%time
tranformData_features_p = Pipeline(stages=[tf, idf, assembler]).fit(tranformData_cleaned)
tranformData_features_p.save("file:///databricks/driver/tranformData_features_{now}")
tranformData_features = tranformData_features_p.transform(tranformData_cleaned)
tranformData_features.show(5)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html
%time
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = Pipeline(stages=[lr]).fit(tranformData_features)

# COMMAND ----------

# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = lrModel.summary

print("Training Accuracy:  " + str(trainingSummary.accuracy))
print("Training Precision: " + str(trainingSummary.precisionByLabel))
print("Training Recall:    " + str(trainingSummary.recallByLabel))
print("Training FMeasure:  " + str(trainingSummary.fMeasureByLabel()))
print("Training AUC:       " + str(trainingSummary.areaUnderROC))

# COMMAND ----------

testingDataTransform = lrModel.transform(testingData)
testingDataTransform.show(5)

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = lrModel.transform(testingDataTransform)
predictions.show(5)

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

!ls
