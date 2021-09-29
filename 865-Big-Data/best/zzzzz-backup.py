# Databricks notebook source
# for interpertation check the bottom cell named TOPIC Lib

# Topic modeling
# lda = LDA(k=10, maxIter=10)

# COMMAND ----------

# # Fit the pipeline to training documents.
# pipeline = Pipeline(stages=pipelineStages)
# pipelineFit = pipeline.fit(trainingData)
# PipelineTransformed = pipelineFit.transform(trainingData)

# # Topic modeling
# lda = LDA(k=10, maxIter=10)
# ldaModel = lda.fit(PipelineTransformed)

# transformed = ldaModel.transform(PipelineTransformed).select("topicDistribution")
# transformed.show(truncate=False)  

# ll = ldaModel.logLikelihood(PipelineTransformed)  
# lp = ldaModel.logPerplexity(PipelineTransformed) 

# topicIndices = ldaModel.describeTopics(maxTermsPerTopic = wordNumbers)  
# vocab_broadcast = sc.broadcast(vocabArray)  
# udf_to_word = udf(to_word, ArrayType(StringType()))  
  
# topics = topicIndices.withColumn("words", udf_to_word(topicIndices.termIndices))  
# topics.show(truncate=False)  
# exit()
