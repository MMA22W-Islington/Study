# Databricks notebook source
# DBTITLE 1,Load Data
# Load in one of the tables
df1 = spark.sql("select * from default.video_games_5")
df2 = spark.sql("select * from default.books_5_small")
df3 = spark.sql("select * from default.home_and_kitchen_5_small")
df = df1.union(df2).union(df3)
print((df.count(), len(df.columns)))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# DBTITLE 1,Describe Data
# Let's look at some quick summary statistics
df.describe().show()

# COMMAND ----------

# The count of each overall rating

from pyspark.sql.functions import col
df.groupBy("overall").count().orderBy(col("overall").asc()).show()

# COMMAND ----------

# The most common product IDs
df.groupBy("asin").count().orderBy(col("count").desc()).show(10)

# COMMAND ----------

# DBTITLE 1,Data Wrangling/Prep
# For our intitial modeling efforts, we are not going to use the following features
drop_list = ['overall', 'summary', 'asin', 'reviewID', 'reviewerID', 'summary', 'unixReviewTime','reviewTime', 'image', 'style', 'verified', 'reviewerName']
df = df.select([column for column in df.columns if column not in drop_list])
df.show(5)
print((df.count(), len(df.columns)))

# COMMAND ----------

df = df.na.drop(subset=["reviewText", "label"])
df.show(5)
print((df.count(), len(df.columns)))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.groupBy("label").count().show()

# COMMAND ----------

# DBTITLE 1,Split into testing/training
# set seed for reproducibility
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testingData.count()))

# COMMAND ----------

# DBTITLE 1,Create a Data Transformation/Preprocessing Pipeline
# In Spark's MLLib, it's considered good practices to combine all the preprocessing steps into a pipeline.
# That way, you can run the same steps on both the training data, and testing data and beyond (new data)
# without copying and pasting any code.

# It is possible to run all of these steps one-by-one, outside of a Pipeline, if desired. But that's
# not how I am going to do it here.

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# We'll tokenize the text using a simple RegexTokenizer
regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")


# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")


# Vectorize the sentences using simple BOW method. Other methods are possible:
# https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)


pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])


# COMMAND ----------

# DBTITLE 1,Transform Training Data
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(trainingData)
trainingDataTransformed = pipelineFit.transform(trainingData)
trainingDataTransformed.show(5)

# COMMAND ----------

# DBTITLE 1,Build Logistic Regression Model
from pyspark.ml.classification import LogisticRegression

# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
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

# COMMAND ----------

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

test_df_Transform = pipelineFit.transform(test_df)
test_df_Transform.show(5)

# COMMAND ----------

predictions = lrModel.transform(test_df_Transform)

# COMMAND ----------

predictions.show()
# display(predictions.select('reviewID', 'prediction'))
