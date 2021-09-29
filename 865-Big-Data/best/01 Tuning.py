# Databricks notebook source
# DBTITLE 1,Data preprocessing
# MAGIC %run ./base

# COMMAND ----------

# DBTITLE 1,[Duplicate this] Auto tune
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit,CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import datetime

gbt = GBTClassifier(maxIter=20)
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 5, 10])
             .addGrid(gbt.maxBins, [10, 20, 40])
             .addGrid(gbt.maxIter, [5, 10, 20])
             .build())
gbcv = CrossValidator(estimator = gbt,
                      estimatorParamMaps = paramGrid,
                      evaluator = BinaryClassificationEvaluator(),
                      numFolds = 5,
                      seed = 530,
                      parallelism = 10)

pipelineFit = gbcv.fit(trainingData)
now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
pipelineFit.save(f"file:///dbfs/joe/{now}_model")
comment = [f"Pipeline name: {now}"]

#TODO: NEED TO GET CV
# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
# trainingSummary = pipelineFit.summary

# comment += ["Training Accuracy:  " + str(trainingSummary.accuracy)]
# comment += ["Training Precision: " + str(trainingSummary.precisionByLabel)]
# comment += ["Training Recall:    " + str(trainingSummary.recallByLabel)]
# comment += ["Training FMeasure:  " + str(trainingSummary.fMeasureByLabel())]
# comment += ["Training AUC:       " + str(trainingSummary.areaUnderROC)]
# comment += ["\n"]

# move to next cell to check

# trainingSummary.roc.show()

# Obtain the objective per iteration
# objectiveHistory = trainingSummary.objectiveHistory
# for objective in objectiveHistory:
#     print(objective)

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
submission.write.csv(f"file:///dbfs/joe/{now}_sub.csv", header=True)

# print answer
for line in comment:
  print(line)
