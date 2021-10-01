# Databricks notebook source
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, AlbertEmbeddings, LemmatizerModel, ContextSpellCheckerModel
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

df = df.sample(False, 0.30, seed=0)

df = df.cache()

# COMMAND ----------

from pyspark.sql.functions import to_date

# seems like changing the name will slow down the program
df = df.withColumn(
  "reviewTime",
  to_date(col("reviewTime"), "M d, y")
)

drop_list = [
  "asin",
  "reviewID",
  "reviewerID",
  "unixReviewTime", # reviewTime is the same as unixReviewTime
  "category"  # TODO: "category" is not part of Test
]
df = df.select([column for column in df.columns if column not in drop_list])

df.show()

# COMMAND ----------

# DBTITLE 1,build pre-proccessing Pipeline
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

# embeddings = AlbertEmbeddings.pretrained() \
#    .setInputCols(["document", "normalized"]) \
#    .setOutputCol("embeddings")

# method one provided by prof
# tf = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=2000, minTF=1, maxDF=0.40)
# idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures")

# method two
# hashingTF = HashingTF(inputCol="token_features", outputCol="rawFeatures", numFeatures=20)

from pyspark.ml.feature import Word2Vec
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="token_features", outputCol="w2v")

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "w2v"], outputCol="features")

# pick and choose what pipeline you want.

pipeline_pre = [document_assembler, tokenizer, spellChecker, lemmatizer, stopwords_cleaner, normalizer, finisher, word2Vec,assembler]
# pipeline_pre = [document_assembler, tokenizer, spellChecker, lemmatizer, stopwords_cleaner, normalizer, finisher, tf, idf, assembler]
# pipeline_test = [document_assembler, tokenizer, normalizer, spellChecker, lemmatizer, stopwords_cleaner, finisher] # move normalizer to the back

# COMMAND ----------

eda = Pipeline(stages=pipeline_pre).fit(df).transform(df)
# .select(["reviewText", "result"])
# eda.selectExpr("embeddings").show(10, truncate=False)

# eda.select('sentence').show(10, truncate=False)
eda.show()

# COMMAND ----------

# DBTITLE 1,Feature Selection--UnivariateFeatureSelector
# By default, the selection mode is numTopFeatures, with the default selectionThreshold sets to 50.

# TODO: change the selected number of features based on the final data set

from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.linalg import Vectors

selector = UnivariateFeatureSelector(featuresCol="features", outputCol="selectedFeatures",
                                     labelCol="label", selectionMode="numTopFeatures", selectionThreshold = 30)
selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(1)


# COMMAND ----------

# DBTITLE 1,Split Data-Shuffle
# set seed for reproducibility
(trainingData, testData) = df.randomSplit([0.9, 0.1], seed = 47)

# COMMAND ----------

# DBTITLE 1,Transform Training Data
# Fit the pipeline to training documents.
pipelineFit = Pipeline(stages=pipeline_pre).fit(trainingData)
trainingDataTransformed = pipelineFit.transform(trainingData)
trainingDataTransformed.show(5)

# COMMAND ----------

# DBTITLE 1,Build Logistic Regression Model
from pyspark.ml.classification import LogisticRegression

# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")
lrModel = lr.fit(trainingDataTransformed)

# COMMAND ----------

# DBTITLE 1,Show Training Metrics
# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = lrModel.summary

print("Training Accuracy:  " + str(trainingSummary.accuracy))
print("Training Precision: " + str(trainingSummary.precisionByLabel))
print("Training Recall:    " + str(trainingSummary.recallByLabel))
print("Training FMeasure:  " + str(trainingSummary.fMeasureByLabel()))
print("Training AUC:       " + str(trainingSummary.areaUnderROC))

# COMMAND ----------

trainingSummary.roc.show()
# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
for objective in objectiveHistory:
    print(objective)

# COMMAND ----------

# DBTITLE 1,Transform Testing Data
testingDataTransform = pipelineFit.transform(testingData)
testingDataTransform.show(5)

# COMMAND ----------

# DBTITLE 1,Use Model to Predict Test Data; Evaluate
from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = lrModel.transform(testingDataTransform)
predictions.show(5)

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

# DBTITLE 1,Make Predictions on Kaggle Test Data
# Load in the tables
test_df = spark.sql("select * from default.reviews_test")
test_df.show(5)
print((test_df.count(), len(test_df.columns)))

# COMMAND ----------

# Check Steve's code for generating for Kaggle

text_file_name = "_".join([now.strftime("%Y_%m_%d_%H_%M_%S"), Notes])
with open(text_file_name, "w") as text_file:
  for file in files:
    text_file.write(file)

# COMMAND ----------

test_df_Transform = pipelineFit.transform(test_df)
test_df_Transform.show(5)
predictions = lrModel.transform(test_df_Transform)


# COMMAND ----------

predictions = lrModel.transform(test_df_Transform)

# COMMAND ----------

display(predictions.select('reviewID', 'prediction'))
