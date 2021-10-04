# Databricks notebook source
"""
TODO:
1. proper missing value handle
2. imbalance data
3. topic
4. sentiment analytis
5. [RIGHT-NOW] note down all the auc after adding new features
6. change the count vector values
7. [RIGHT-NOW] count vector value add idf and summaryintion of features
8. [Done]"verified", "overall",


For Imbalance
+-----+-------+
|label|  count|
+-----+-------+
|    1| 626166|
|    0|2861165|
+-----+-------+
"""


# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Load Data
# Load in one of the tables
df1 = spark.sql("select * from default.video_games_5")
df2 = spark.sql("select * from default.books_5_small")
df3 = spark.sql("select * from default.home_and_kitchen_5_small")
df = df1.union(df2).union(df3)
print((df.count(), len(df.columns)))

# COMMAND ----------

import random
import numpy as np
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml.feature import VectorAssembler

def vectorizerFunction(dataInput, TargetFieldName):
    if(dataInput.select(TargetFieldName).distinct().count() != 2):
        raise ValueError("Target field must have only 2 distinct classes")
    columnNames = list(dataInput.columns)
    columnNames.remove(TargetFieldName)
    dataInput = dataInput.select((','.join(columnNames)+','+TargetFieldName).split(','))
    assembler=VectorAssembler(inputCols = columnNames, outputCol = 'features')
    pos_vectorized = assembler.transform(dataInput)
    vectorized = pos_vectorized.select('features',TargetFieldName).withColumn('label',pos_vectorized[TargetFieldName]).drop(TargetFieldName)
    return vectorized

def SmoteSampling(vectorized, k = 5, minorityClass = 1, majorityClass = 0, percentageOver = 200, percentageUnder = 100):
    if(percentageUnder > 100|percentageUnder < 10):
        raise ValueError("Percentage Under must be in range 10 - 100");
    if(percentageOver < 100):
        raise ValueError("Percentage Over must be in at least 100");
    dataInput_min = vectorized[vectorized['label'] == minorityClass]
    dataInput_maj = vectorized[vectorized['label'] == majorityClass]
    feature = dataInput_min.select('features')
    feature = feature.rdd
    feature = feature.map(lambda x: x[0])
    feature = feature.collect()
    feature = np.asarray(feature)
    nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto').fit(feature)
    neighbours =  nbrs.kneighbors(feature)
    gap = neighbours[0]
    neighbours = neighbours[1]
    min_rdd = dataInput_min.drop('label').rdd
    pos_rddArray = min_rdd.map(lambda x : list(x))
    pos_ListArray = pos_rddArray.collect()
    min_Array = list(pos_ListArray)
    newRows = []
    nt = len(min_Array)
    nexs = percentageOver/100
    for i in range(nt):
        for j in range(nexs):
            neigh = random.randint(1,k)
            difs = min_Array[neigh][0] - min_Array[i][0]
            newRec = (min_Array[i][0]+random.random()*difs)
            newRows.insert(0,(newRec))
    newData_rdd = sc.parallelize(newRows)
    newData_rdd_new = newData_rdd.map(lambda x: Row(features = x, label = 1))
    new_data = newData_rdd_new.toDF()
    new_data_minor = dataInput_min.unionAll(new_data)
    new_data_major = dataInput_maj.sample(False, (float(percentageUnder)/float(100)))
    return new_data_major.unionAll(new_data_minor)

SmoteSampling(vectorizerFunction(df, 'label'), k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 90, percentageUnder = 5)

# COMMAND ----------

# DBTITLE 1,Data Wrangling/Prep
from pyspark.sql.functions import to_date, year, month, dayofmonth, dayofyear, weekofyear
from pyspark.sql.functions import col, lit, when, split, size

def featureEngineering(df): 
  res = df.withColumn(
    "reviewTime",
    to_date(col("reviewTime"), "M d, y")
  )

  # Dates
#   res = res.withColumn('reviewTime_year', year(col('reviewTime')))
#   res = res.withColumn('reviewTime_month', month(col('reviewTime')))
#   res = res.withColumn('reviewTime_day', dayofmonth(col('reviewTime')))
#   res = res.withColumn('reviewTime_dayofy', dayofyear(col('reviewTime')))
#   res = res.withColumn('reviewTime_week_no', weekofyear(col('reviewTime')))

#   #Reviewer Name
  res = res.withColumn('reviewerName_Shorthand', when(col('reviewerName').rlike('\\. '),True).otherwise(False))
  res = res.withColumn('reviewerName_isAmazon', when(col('reviewerName').rlike('Amazon'),True).otherwise(False))
  res = res.withColumn('reviewerName_capsName', when(col('reviewerName').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))

#   # check if review contains all caps words
  res = res.withColumn('reviewTextHasCapsWord', when(col('reviewText').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
  res = res.withColumn('summaryHasCapsWord', when(col('summary').rlike('\\b[A-Z]{2,}\\b'),True).otherwise(False))
#   # check if review contians swear
  res = res.withColumn('reviewTextHasSwearWord', when(col('reviewText').rlike('\\*{2,}'),True).otherwise(False))
  res = res.withColumn('summaryHasSwearWord', when(col('summary').rlike('\\*{2,}'),True).otherwise(False))
  ## Number of Exclaimation
  res = res.withColumn('reviewTextNumberExclamation', size(split(col('reviewText'), r"!")) - 1)
  res = res.withColumn('summaryNumberExclamation', size(split(col('summary'), r"!")) - 1)
  ## Number of Exclaimation
  res = res.withColumn('reviewTextNumberComma', size(split(col('reviewText'), r",")) - 1)
  res = res.withColumn('summaryNumberComma', size(split(col('summary'), r",")) - 1)
  ## Number of Exclaimation
  res = res.withColumn('reviewTextNumberPeriod', size(split(col('reviewText'), r"\.")) - 1)
  res = res.withColumn('summaryNumberPeriod', size(split(col('summary'), r"\.")) - 1)
  
  return (res, [
#     "reviewTime_year",
#     "reviewTime_month",
#     "reviewTime_day",
#     "reviewTime_dayofy",
#     "reviewTime_week_no",
#     "reviewerName_Shorthand",
#     "reviewerName_isAmazon",
#     "reviewerName_capsName",
    "reviewTextHasCapsWord",
#     "summaryHasCapsWord",
#     "reviewTextHasSwearWord",
#     "summaryHasSwearWord",
#     "reviewTextNumberExclamation",
#     "summaryNumberExclamation",
#     "reviewTextNumberComma",
#     "summaryNumberComma",
#     "reviewTextNumberPeriod",
#     "summaryNumberPeriod",
    
    "overall", 
    "verified"
  ])

df, featureList = featureEngineering(df)
# For our intitial modeling efforts, we are not going to use the following features
drop_list = ['asin', 'reviewID', 'unixReviewTime','reviewTime', 'image', 'style', 'reviewerName']
df = df.select([column for column in df.columns if column not in drop_list])
df = df.na.drop(subset=["reviewText", "label", "summary"])
print((df.count(), len(df.columns)))

# class Weight - garbage
import pandas as pd
counts = df.groupBy('label').count().toPandas()
# Counts
count_fraud = counts[counts['label']==1]['count'].values[0]
count_total = counts['count'].sum()
# Weights
c = 2
weight_fraud = count_total / (c * count_fraud)
weight_no_fraud = count_total / (c * (count_total - count_fraud))
df = df.withColumn("classWeightCol", when(col("label") ==1, weight_fraud).otherwise(weight_no_fraud))



# Split
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)

# COMMAND ----------

# DBTITLE 1,Create a Data Transformation/Preprocessing Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler, SQLTransformer
from sparknlp.annotator import ContextSpellCheckerModel, LemmatizerModel
from pyspark.ml.classification import LogisticRegression
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel, DocumentNormalizer
from pyspark.ml.clustering import LDA
# , SymmetricDeleteModel, ContextSpellCheckerApproach, NormalizerModel, ContextSpellCheckerModel, NorvigSweetingModel, AlbertEmbeddings, DocumentNormalizer

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

#   spellChecker = ContextSpellCheckerModel.pretrained() \
#       .setInputCols(f"{fieldname}_token") \
#       .setOutputCol(f"{fieldname}_corrected")


#       .setInputCols([f"{fieldname}_corrected"]) \
  lemmatizer = LemmatizerModel.pretrained() \
      .setInputCols([f"{fieldname}_token"]) \
      .setOutputCol(f"{fieldname}_lemma")

  # remove stopwords
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


  # Vectorize the sentences using simple BOW method. Other methods are possible:
  # https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
  countVectors = CountVectorizer(inputCol=f"{fieldname}_token_features", outputCol=f"{fieldname}_rawFeatures", vocabSize=10000, minDF=5)


  # IDF
#   idf = IDF(inputCol=f"rawFeatures", outputCol=f"idfFeatures", minDocFreq=5)


#   cleaned_token_size = SQLTransformer(
#       statement = f"SELECT * , size({fieldname}_filtered) AS {fieldname}_tokenSize FROM __THIS__"
#   )
  cleaned_token_size = SQLTransformer(
      statement = f"SELECT * , size({fieldname}_token_features) AS {fieldname}_tokenSize FROM __THIS__"
  )
  
  return [
    document_assembler, 
    documentNormalizer, 
    tokenizer, 
#     spellChecker,
    lemmatizer,
    stopwords_cleaner,
    normalizer,
    finisher,
    countVectors,
    cleaned_token_size], [f"{fieldname}_rawFeatures", f"{fieldname}_tokenSize"]

# Build up the pipeline
cols = featureList
pipelineObj = []

for field in [ "reviewText", "summary"]: #, "summary"
  nlpPipe, output = NLPPipe(field)
  cols += output
  pipelineObj += nlpPipe
  
assembler = VectorAssembler(inputCols=cols, outputCol="features")

pipelineStages = pipelineObj + [assembler]

# COMMAND ----------

# DBTITLE 1,Transform Training Data
# Fit the pipeline to training documents.
pipeline = Pipeline(stages=pipelineStages)
pipelineFit = pipeline.fit(trainingData)
PipelineTransformed = pipelineFit.transform(trainingData)

# Topic modeling
lda = LDA(k=10, maxIter=10)
ldaModel = lda.fit(PipelineTransformed)

transformed = ldaModel.transform(PipelineTransformed).select("topicDistribution")
transformed.show(truncate=False)  

ll = ldaModel.logLikelihood(PipelineTransformed)  
lp = ldaModel.logPerplexity(PipelineTransformed) 

topicIndices = ldaModel.describeTopics(maxTermsPerTopic = wordNumbers)  
vocab_broadcast = sc.broadcast(vocabArray)  
udf_to_word = udf(to_word, ArrayType(StringType()))  
  
topics = topicIndices.withColumn("words", udf_to_word(topicIndices.termIndices))  
topics.show(truncate=False)  
exit()
