# Databricks notebook source
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
import pyspark.sql.functions as F

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

print((df.count(), len(df.columns)))

# Take a sample (useful for code development purposes)
df_sample = df.sample(False, 0.15, seed=0)
df_test = df.sample(False, 0.0001, seed=0)

df = df.cache()
df_test = df_test.cache()

print((df_sample.count(), len(df_sample.columns)))
print((df_test.count(), len(df_sample.columns)))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# DBTITLE 1,Describe Data
# Let's look at some quick summary statistics
df.describe().show()

# COMMAND ----------

# Let's look at df1 - video_games
df1.describe().show()

# COMMAND ----------

# Let's look at df2 - home_and_kitchen
df2.describe().show()

# COMMAND ----------

# Let's look at df3 - books
df3.describe().show()

# COMMAND ----------

display(df.groupBy("category").count().orderBy("category"))

# COMMAND ----------

# from pyspark.sql.functions import col
display(df.groupBy("overall").count().orderBy("overall"))

# COMMAND ----------

display(df.groupBy("overall", "label").count().orderBy("overall"))

# COMMAND ----------

# The most common product IDs
display(df.groupBy("asin").count().orderBy(col("count").desc()).head(50))

# COMMAND ----------

display(df.groupBy("label").count().orderBy("label"))

# COMMAND ----------

# DBTITLE 1,Remove Unwanted Column
drop_list = [
  "reviewID",
  "reviewerID",
  "unixReviewTime", # reviewTime is the same as unixReviewTime
  "category"  # TODO: "category" is not part of Test
]

df = df.select([column for column in df.columns if column not in drop_list])

df.show(2)

# COMMAND ----------

# DBTITLE 1,Data Transformation Pipeline - Simplified Version
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
# from pyspark.sql import functions as f

# We'll tokenize the text using a simple RegexTokenizer
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")

# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

# Normalizer


#### TO-DO: ADD lemitizer

pipeline = Pipeline(stages=[tokenizer, stopwordsRemover])

# pipelineFit = pipeline.fit(df)
df_t1 = pipeline.fit(df).transform(df)

# COMMAND ----------

df_t1.show(2)

# COMMAND ----------

type([df_t1.filtered])

# COMMAND ----------

df.describe().show()

# COMMAND ----------

pipelineFit = nlp_pipeline.fit(df)
df_transform = pipelineFit.transform(df)

# COMMAND ----------

df_transform.head(1)

# COMMAND ----------

# Test ratio between label 0 and 1 per id
counts = df.select(F.explode('filtered').alias('col')).groupBy('col').count().sort(F.desc('count')).collect()
display(counts)

# COMMAND ----------

# DBTITLE 1,Data Transformation Pipeline
from sparknlp.base import DocumentAssembler
# documentAssembler = DocumentAssembler() \      
#      .setInputCol("reviewText") \      
#      .setOutputCol('document')
documentAssembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

from sparknlp.annotator import Tokenizer
tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('tokenized')

from sparknlp.annotator import Normalizer
normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') \
     .setLowercase(True)
  
from sparknlp.annotator import LemmatizerModel
lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemmatized')

# from nltk.corpus import stopwords
# eng_stopwords = stopwords.words('english')

from sparknlp.annotator import StopWordsCleaner
# stopwords_cleaner = StopWordsCleaner() \
#      .setInputCols(['lemmatized']) \
#      .setOutputCol('no_stop_lemmatized') \
#      .setStopWords(eng_stopwords)
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("lemmatized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
# # Remove standard Stopwords
# # stopwordsRemover = StopWordsRemover(inputCol="lemmatized", outputCol="filtered")
# stopwords_cleaner = StopWordsRemover(inputCol="lemmatized", outputCol="no_stop_lemmatized")

from sparknlp.annotator import PerceptronModel
pos_tagger = PerceptronModel.pretrained('pos_anc') \
     .setInputCols(['document', 'lemmatized']) \
     .setOutputCol('pos')

from sparknlp.annotator import Chunker
allowed_tags = ['<JJ>+<NN>', '<NN>+<NN>']
chunker = Chunker() \
     .setInputCols(['document', 'pos']) \
     .setOutputCol('ngrams') \
     .setRegexParsers(allowed_tags)

from sparknlp.base import Finisher
finisher = Finisher() \
     .setInputCols(['cleanTokens', 'ngrams'])
#      .setInputCols(['unigrams', 'ngrams'])


from pyspark.ml import Pipeline
pipeline = Pipeline() \
     .setStages([documentAssembler,
                 tokenizer,
                 normalizer,
                 lemmatizer,
                 stopwords_cleaner,
                 pos_tagger,
                 chunker, finisher])

# COMMAND ----------

# pipelineFit = pipeline.fit(df)

df_transform = pipeline.fit(df).transform(df)

df_transform.show(2)

# from pyspark.sql.functions import concat
# processed_review = df_sample_trans.withColumn('final',
#      concat(F.col('finished_unigrams'), 
#             F.col('finished_ngrams')))

# COMMAND ----------

# DBTITLE 1,Get Term Frequencies
counts = df_transform.select(F.explode('finished_cleanTokens').alias('col')).groupBy('col').count().sort(F.desc('count')).collect()
# display(counts)

# COMMAND ----------

# DBTITLE 1,Split dataset into label 1vs0 to see distribution if different - Term Frequencies
df_target = df_transform.filter(df_transform.label == 1)
df_target.show(5)
counts = df_target.select(F.explode('finished_cleanTokens').alias('col')).groupBy('col').count().sort(F.desc('count')).collect()
display(counts)

# COMMAND ----------

df_nonTarget = df_transform.filter(df_transform.label == 0)
df_nonTarget.show(5)
counts = df_nonTarget.select(F.explode('finished_cleanTokens').alias('col')).groupBy('col').count().sort(F.desc('count')).collect()
display(counts)

# COMMAND ----------

# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline# Start Spark Session with Spark NLP
#spark = sparknlp.start()spark = SparkSession.builder \
#     .appName("BBC Text Categorization")\
#     .config("spark.driver.memory","8G")\ change accordingly
#     .config("spark.memory.offHeap.enabled",True)\
#     .config("spark.memory.offHeap.size","8G") \
#     .config("spark.driver.maxResultSize", "2G") \
#     .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5")\
#     .config("spark.kryoserializer.buffer.max", "1000M")\
#     .config("spark.network.timeout","3600s")\
#     .getOrCreate()
    
from pyspark.ml.feature import HashingTF, IDF, StringIndexer, SQLTransformer,IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
 
# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)# stems tokens to bring it to root form
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)# To generate Term Frequency

hashingTF = HashingTF(inputCol="token_features", outputCol="rawFeatures", numFeatures=1000)# To generate Inverse Document Frequency
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)# convert labels (string) to integers. Easy to process compared to string.
label_stringIdx = StringIndexer(inputCol = "category", outputCol = "label")# define a simple Multinomial logistic regression model. Try different combination of hyperparameters and see what suits your data. You can also try different algorithms and compare the scores.
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)# To convert index(integer) to corresponding class labels
label_to_stringIdx = IndexToString(inputCol="label", outputCol="article_class")# define the nlp pipeline

pipeline_pre = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            stemmer, 
            finisher])

nlp_pipeline = Pipeline(
    stages=[pipeline_pre,
            hashingTF,
            idf,
            label_stringIdx,
            lr,
            label_to_stringIdx])

# COMMAND ----------

df_clean = pipeline_pre.fit(df).transform(df)
df_clean.show(2)

# COMMAND ----------

counts = df_clean.select(F.explode('token_features').alias('col')).groupBy('col').count().sort(F.desc('count')).collect()

# COMMAND ----------

# DBTITLE 1,Get Review Text Length
import pyspark.sql.functions as F

### this line works
# df_test.withColumn('text_length', F.length(df_test.reviewText)).collect()

### this line works
# df_test.select(df_test.reviewText, F.length(df_test.reviewText).alias('text_length')).collect()

# lens = df_test.withColumn('text_length', F.length(df_test.reviewText)).alias('col').groupBy('col').count().sort(F.desc('count')).collect()

# code test on df_test 
# lens = df_test.select(F.length(df_test.reviewText).alias('col')).groupBy('col').count().sort(F.desc('count')).collect()                 
# display(lens)

lens = df.select(F.length(df.reviewText).alias('col')).groupBy('col').count().sort(F.desc('count')).collect()                 
display(lens)

# COMMAND ----------

# df_target = df.filter(df.label == 1)
# df_target.show(5)
lens = df_target.select(F.length(df_target.reviewText).alias('lengh')).groupBy('lengh').count().sort(F.desc('count')).collect()                 
display(lens)

# COMMAND ----------

df = df.na.drop(subset=["reviewText", "label"])
df.show(5)
print((df.count(), len(df.columns)))

# COMMAND ----------

# DBTITLE 1,Sentiment Analysis (To to later)
import nltk
nltk.download('vader_lexicon')

# COMMAND ----------

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()

# COMMAND ----------

from pyspark.sql.functions import udf
@udf
sent.polarity_scores(df_transform['reviewText'])

df = ddf_transform.withColumn('reviewText', polarity_scores('reviewText'))

# COMMAND ----------

# DBTITLE 1,Topic Analysis


# COMMAND ----------



# COMMAND ----------

print((df_transform.count(), len(df_transform.columns)))

# COMMAND ----------

type([df_transform.tokenized])

# COMMAND ----------



# COMMAND ----------

## Do Term Frequency
counts = df_transform.select(F.explode('finished_ngrams').alias('col')).groupBy('col').count().sort(F.desc('count')).collect()
display(counts)

# COMMAND ----------

# tf-idf vectorization 
from pyspark.ml.feature import CountVectorizer
tfizer = CountVectorizer(inputCol='finished_no_stop_lemmatized',
                         outputCol='tf_features')
tf_model = tfizer.fit(df_transform)
tf_result = tf_model.transform(df_transform)

# COMMAND ----------



# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Building Pipeline Stage
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, RegexTokenizer, StopWordsRemover, VectorSizeHint
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes

# We'll tokenize the text using a simple RegexTokenizer
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")

# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

# TODO: insert other clearning steps here (and put into the pipeline, of course!)
# E.g., n-grams? document length?
# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")


# # convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
 
# # clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

# # remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# # stems tokens to bring it to root form
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

# # Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)


# Vectorize the sentences using simple BOW method. Other methods are possible:
# https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
tf = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=2000, minTF=1, maxDF=0.40)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=100)

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "idfFeatures"], outputCol="features")

# Machine Learning Algorithm
ml_lr  = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)
ml_rf  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
ml_nb = NaiveBayes(smoothing=1.0, modelType="multinomial")


# COMMAND ----------

# DBTITLE 1,Create Pipeline Obj
counter = CountVectorizer(inputCol="filtered", outputCol="wordCount")


# pick and choose what pipeline you want.
pipeline_pre = [tokenizer, stopwordsRemover, counter]

#tf, idf, assembler
# paramGrid = ParamGridBuilder() \
#     .addGrid(ml_alg.regParam, [0.3, 0.5, 0.7]) \
#     .addGrid(ml_alg.elasticNetParam, [0.0]) \
#     .addGrid(tf.minTF, [1, 100, 1000]) \
#     .addGrid(tf.vocabSize, [500, 1000, 2500, 5000]) \
#     .build()

eda = Pipeline(stages=pipeline_pre).fit(df).transform(df)

# COMMAND ----------

# DBTITLE 1,Flag words for Noun, Verb and Adjectives
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

from IPython.display import display
lemmatizer = nltk.WordNetLemmatizer()

#word tokenizeing and part-of-speech tagger
document = 'The little brown dog barked at the black cat'
tokens = [nltk.word_tokenize(sent) for sent in [document]]
postag = [nltk.pos_tag(sent) for sent in tokens][0]

# Rule for NP chunk and VB Chunk
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        {<RB.?>*<VB.?>*<JJ>*<VB.?>+<VB>?} # Verbs and Verb Phrases
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        
"""
#Chunking
cp = nltk.RegexpParser(grammar)

# the result is a tree
tree = cp.parse(postag)

# COMMAND ----------


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label() =='NP'):
        yield subtree.leaves()
        
def get_word_postag(word):
    if pos_tag([word])[0][1].startswith('J'):
        return wordnet.ADJ
    if pos_tag([word])[0][1].startswith('V'):
        return wordnet.VERB
    if pos_tag([word])[0][1].startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.NOUN
    
def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    postag = get_word_postag(word)
    word = lemmatizer.lemmatize(word,postag)
    return word

def get_terms(tree):    
    for leaf in leaves(tree):
        terms = [normalise(w) for w,t in leaf]
        yield terms

terms = get_terms(tree)

features = []
for term in terms:
    _term = ''
    for word in term:
        _term += ' ' + word
    features.append(_term.strip())
features

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Longer Pipeline from Tutorials (takes forever to run)
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier


# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")


# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
 
# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# stems tokens to bring it to root form
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Generate Term Frequency
tf = CountVectorizer(inputCol="token_features", outputCol="rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.40)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=5)

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "idfFeatures"], outputCol="features")

# Machine Learning Algorithm
#ml_alg  = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)
ml_alg  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)

# nlp_pipeline = Pipeline(
#     stages=[document_assembler, 
#             tokenizer,
#             normalizer,
#             stopwords_cleaner, 
#             stemmer, 
#             finisher,
#             tf,
#             idf,
#             assembler,
#             ml_alg])

# nlp_pipeline = Pipeline(
#     stages=[document_assembler, 
#             tokenizer,
#             normalizer,
#             stopwords_cleaner, 
#             stemmer, 
#             finisher,
#             tf,
#             idf,
#             assembler])

nlp_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            stemmer, 
            finisher])

# COMMAND ----------

drop_list = ['overall', 'summary', 'asin', 'reviewID', 'reviewerID', 'summary', 'unixReviewTime','reviewTime', 'image', 'style', 'verified', 'reviewerName']
df = df.select([column for column in df.columns if column not in drop_list])
df.show(5)
print((df.count(), len(df.columns)))

# COMMAND ----------

# set seed for reproducibility
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testingData.count()))

# COMMAND ----------

# Fit the pipeline to training documents.
pipelineFit = nlp_pipeline.fit(trainingData)
# trainingDataTransformed = pipelineFit.transform(trainingData)
# trainingDataTransformed.show(5)

# COMMAND ----------

# use sample data to do the analysis
# set seed for reproducibility
(trainingData, testingData) = df_sample.randomSplit([0.8, 0.2], seed = 47)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testingData.count()))

# COMMAND ----------

# Fit the pipeline to training documents.
pipelineFit = nlp_pipeline.fit(trainingData)

# COMMAND ----------


