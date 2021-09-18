# Databricks notebook source
# TODO: save model 
# TODO: how to LOG https://stackoverflow.com/questions/61782919/pyspark-logging-printing-information-at-the-wrong-log-level
# TODO: Log Time, and download
# TODO: Train shuffle, strafied (https://spark.apache.org/docs/latest/ml-tuning.html)
# TODO: RF Ensemble

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, RegexTokenizer, StopWordsRemover, VectorSizeHint
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes

from pyspark.sql.functions import lit, col
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import datetime

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
# We'll tokenize the text using a simple RegexTokenizer
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")

# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

# pick and choose what pipeline you want.
pipeline_pre = [tokenizer, stopwordsRemover, counter]

# COMMAND ----------

# DBTITLE 1,Build Estimator list
# boosting + rf ensemble

# Machine Learning Algorithm
ml_lr  = LogisticRegression(maxIter=10)
paramGrid_lr = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()
tvs_lr = TrainValidationSplit(estimator=ml_lr,
                           estimatorParamMaps=paramGrid_lr,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)


ml_rf  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
## NEED TO FIND PRESET
paramGrid_rf = ParamGridBuilder()\
    .addGrid(ml_rf.regParam, [0.1, 0.01]) \
    .addGrid(ml_rf.fitIntercept, [False, True])\
    .addGrid(ml_rf.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()
tvs_rf = TrainValidationSplit(estimator=ml_rf,
                           estimatorParamMaps=paramGrid_rf,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)


ml_nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
# NEED TO FIND PRESET
paramGrid_nb = ParamGridBuilder()\
    .addGrid(ml_nb.regParam, [0.1, 0.01]) \
    .addGrid(ml_nb.fitIntercept, [False, True])\
    .addGrid(ml_nb.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()
tvs_nb = TrainValidationSplit(estimator=ml_nb,
                           estimatorParamMaps=paramGrid_nb,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)


estimators = [("lr", tvs_lr), ("rf", tvs_rf), ("nb", tvs_nb)]

# COMMAND ----------

# DBTITLE 1,Split Data-Shuffle
# set seed for reproducibility
(trainingData, testData) = df.randomSplit([0.9, 0.1], seed = 47)

# COMMAND ----------

# DBTITLE 1,Train and Predict (only for feature selection and that sort)
estimatorsfit_transform = [];
# TODO: This is for feature selection, not for tuning.
for name, ml in estimators:
  pipeline = Pipeline(stages=pipeline_pre + [ml])
  
  pipelineFit = pipeline.fit(trainingData)
  predictions = pipelineFit.transform(testData)
  fit_transform += [(name, pipelineFit, predictions)]

# COMMAND ----------

# DBTITLE 1,Metrics
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

# COMMAND ----------

 !pwd


# COMMAND ----------

for file in files:
  print(file)

# COMMAND ----------


