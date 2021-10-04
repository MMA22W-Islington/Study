# Databricks notebook source
# DBTITLE 1,Data preprocessing
# MAGIC %run ./base2

# COMMAND ----------

# DBTITLE 1,[Duplicate this] Auto tune
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit,CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import datetime

lr = LogisticRegression(maxIter=20,regParam=0.01, elasticNetParam=0, fitIntercept = True)

# pipelineFit = gbcv.fit(trainingData)
pipelineFit = lr.fit(trainingData)
now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
pipelineFit.save(f"file:///dbfs/joe/{now}_model")
comment = [f"Pipeline name: {now}"]

comment += ["Train auc: ", pipelineFit.summary.areaUnderROC]

# prediction
predictions = pipelineFit.transform(testingData)

auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
comment += ["Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions))]

# Load in the tables
test_df = spark.sql("select * from default.reviews_holdout")
test_df, _ = featureEngineering(test_df)
submit_tranformed = preprocess.transform(test_df)
submit_predictions = pipelineFit.transform(submit_tranformed)
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

lastElement=udf(lambda v:float(v[1]),FloatType())
submission = submit_predictions.select('reviewID', lastElement('probability').alias("label"))
submission.write.csv(f"file:///dbfs/joe/{now}", header=True)

# print answer
for line in comment:
  print(line)
