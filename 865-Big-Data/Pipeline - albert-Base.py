# Databricks notebook source
# TODO: parallel
# TODO: summary?
# TODO: Sentiment https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.SentimentDetectorModel.html
# TODO: save model 
# TODO: RF Ensemble
# TODO: html anchor

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, AlbertEmbeddings, LemmatizerModel, ContextSpellCheckerModel, NerDLApproach
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

# df = df.sample(False, 0.30, seed=0)

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

# DBTITLE 1,build pre-proccessing Pipeline
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

# convert document to array of tokens
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

# Define the pretrained Albert model. 
# CHECK: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/dl-ner/ner_albert.ipynb
albert = AlbertEmbeddings.pretrained()\
        .setInputCols(["document", "token"])\
       .setOutputCol("albert")

sentence_embeddings = SentenceEmbeddings() \
            .setInputCols(["document", "albert"]) \
            .setOutputCol("sentence_embeddings") \
            .setPoolingStrategy("AVERAGE")


# Define the Char CNN - BiLSTM - CRF model. We will feed it the Albert tokens 
nerTagger = NerDLApproach()\
  .setInputCols(["document", "token", "albert"])\
  .setLabelColumn("label")\
  .setOutputCol("ner")\
  .setMaxEpochs(1)\
  .setRandomSeed(0)\
  .setVerbose(0)

assembler = VectorAssembler(inputCols=["verified", "overall", "idfFeatures"], outputCol="features")

# Combine all features into one final "features" column
# assembler = VectorAssembler(inputCols=["verified", "overall", "albert"], outputCol="features")

# put everything into the pipe
pipeline = Pipeline(
    stages = [
      document_assembler,
      tokenizer,
      albert
  ])

# pick and choose what pipeline you want.
# pipeline_pre = [document_assembler, tokenizer, spellChecker, lemmatizer, stopwords_cleaner, normalizer, finisher, tf, idf, assembler]
# pipeline_test = [document_assembler, tokenizer, normalizer, spellChecker, lemmatizer, stopwords_cleaner, finisher] # move normalizer to the back

# COMMAND ----------

# DBTITLE 1,Split Data-Shuffle
# set seed for reproducibility
(trainingData, testData) = df.randomSplit([0.9, 0.1], seed = 47)

# COMMAND ----------

# DBTITLE 1,Train and Predict (only for feature selection and that sort)
pipelineFit = pipeline.fit(trainingData)
# redictions = pipelineFit.transform(testData)

# COMMAND ----------

# Fit the pipeline to training documents.

trainingDataTransformed = pipelineFit.transform(trainingData)

from pyspark.ml.classification import LogisticRegression

# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingDataTransformed)

trainingSummary = lrModel.summary

print("Training Accuracy:  " + str(trainingSummary.accuracy))
print("Training Precision: " + str(trainingSummary.precisionByLabel))
print("Training Recall:    " + str(trainingSummary.recallByLabel))
print("Training FMeasure:  " + str(trainingSummary.fMeasureByLabel()))
print("Training AUC:       " + str(trainingSummary.areaUnderROC))


# COMMAND ----------

# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = pipelineFit.summary

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
