# Databricks notebook source
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
import pyspark.sql.functions as F

# COMMAND ----------

# Load in one of the tables
df1 = spark.sql("select * from default.video_games_5")
df1 = df1.withColumn('category', lit("video_games"))

df2 = spark.sql("select * from default.home_and_kitchen_5_small")
df2 = df2.withColumn('category', lit("home_and_kitchen"))

df3 = spark.sql("select * from default.books_5_small")
df3 = df3.withColumn('category', lit("books"))

df = df1.union(df2).union(df3)

print((df.count(), len(df.columns)))

# Take a sample (useful for code development purposes)
df_sample = df.sample(False, 0.15, seed=0)
# df_test = df.sample(False, 0.0001, seed=0)

df = df.cache()
# df_test = df_test.cache()

# COMMAND ----------

drop_list = [
  "reviewID",
  "reviewerID",
  "unixReviewTime", # reviewTime is the same as unixReviewTime
  "category"  # TODO: "category" is not part of Test
]

df = df.select([column for column in df.columns if column not in drop_list])

df.show(2)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
# from pyspark.sql import functions as f

# We'll tokenize the text using a simple RegexTokenizer
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")
df_token = tokenizer.transform(df)
df_token.show(2)

from sparknlp.annotator import LemmatizerModel
lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['words']) \
     .setOutputCol('lemmatized')
df_lemma = lemmatizer.transform(df_token)
df_lemma.show(2)

# # Remove standard Stopwords
# stopwordsRemover = StopWordsRemover(inputCol="lemmatized", outputCol="filtered")
# df_clean = stopwordsRemover.transform(df_token)

# COMMAND ----------

df_clean.select('filtered').dtypes
# df_token.select('words').printSchema

# COMMAND ----------

# from pyspark.ml import Pipeline
# pipeline = Pipeline() \
#      .setStages([documentAssembler,
#                  tokenizer,
#                  normalizer,
#                  lemmatizer,
#                  stopwords_cleaner,
#                  pos_tagger,
#                  chunker, finisher])

# COMMAND ----------

pipeline = Pipeline(stages=[tokenizer, normalizer, stopwordsRemover])

# pipelineFit = pipeline.fit(df)
df_t1 = pipeline.fit(df).transform(df)

df_t1.show(2)

# COMMAND ----------

counts = df_t1.select(F.explode('filtered').alias('col')).groupBy('col').count().sort(F.desc('count')).collect()
# counts.display()

# COMMAND ----------

display(counts)

# COMMAND ----------

from sparknlp.base import *
from sparknlp.annotator import *

# documentAssembler = DocumentAssembler() \      
#      .setInputCol("reviewText") \      
#      .setOutputCol('document')
documentAssembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

# from sparknlp.annotator import Tokenizer
tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('tokenized')

# from sparknlp.annotator import Normalizer
normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

# from sparknlp.annotator import LemmatizerModel
lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemmatized')

# from sparknlp.annotator import StopWordsCleaner
# stopwords_cleaner = StopWordsCleaner() \
#      .setInputCols(['lemmatized']) \
#      .setOutputCol('no_stop_lemmatized') \
#      .setStopWords(eng_stopwords)
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("lemmatized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

finisher = Finisher() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)# To generate Term Frequency

# COMMAND ----------

pipeline = Pipeline() \
     .setStages([documentAssembler,
                 tokenizer,
                 normalizer,
                 lemmatizer,
                 stopwords_cleaner, 
                finisher])

# COMMAND ----------

df_nlp = pipeline.fit(df).transform(df)

df_nlp.show(2)

# COMMAND ----------

counts = df_nlp.select(F.explode('token_features').alias('col')).groupBy('col').count().sort(F.desc('count')).collect()
# counts.display()

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC from wordcloud import WordCloud, STOPWORDS
# MAGIC import matplotlib.pyplot as plt
# MAGIC import seaborn as sns
# MAGIC from scipy import stats
# MAGIC 
# MAGIC %matplotlib inline
# MAGIC 
# MAGIC def cloud(data,backgroundcolor = 'white', width = 800, height = 600):
# MAGIC     wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor,
# MAGIC                          width = width, height = height).generate(data)
# MAGIC     plt.figure(figsize = (15, 10))
# MAGIC     plt.imshow(wordcloud)
# MAGIC     plt.axis("off")
# MAGIC     plt.show()

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC # from bs4 import BeautifulSoup
# MAGIC import re
# MAGIC import nltk
# MAGIC 
# MAGIC def prep(review):
# MAGIC     
# MAGIC     # Remove HTML tags.
# MAGIC     review = BeautifulSoup(review,'html.parser').get_text()
# MAGIC     
# MAGIC     # Remove non-letters
# MAGIC     review = re.sub("[^a-zA-Z]", " ", review)
# MAGIC     
# MAGIC     # Lower case
# MAGIC     review = review.lower()
# MAGIC     
# MAGIC     # Tokenize to each word.
# MAGIC     token = nltk.word_tokenize(review)
# MAGIC     
# MAGIC     # Stemming
# MAGIC     review = [nltk.stem.SnowballStemmer('english').stem(w) for w in token]
# MAGIC     
# MAGIC     # Join the words back into one string separated by space, and return the result.
# MAGIC     return " ".join(review)
# MAGIC     

# COMMAND ----------

# MAGIC %python
# MAGIC X_train['clean'] = df['reviewText'].apply(prep)

# COMMAND ----------


