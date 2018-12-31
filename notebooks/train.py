# Databricks notebook source
print "Mounting Blob Storage" #We Should Only Attempt Mounting if the Required mount Folders do not already exist - It seems to remain even when cluster terminated.

# Collect the Parameters Indicating the Model + data location
dbutils.widgets.text("data", "historical-data","")
dbutils.widgets.text("model", "models","")
dbutils.widgets.get("data")
dbutils.widgets.get("model")
dataSource = getArgument("data")
modelSource = getArgument("model")

print modelSource, dataSource

# Capture mounting execeptions and log the relevant detail
def mountExceptionHandler(e):
   # The error message has a long stack trace.  This code tries to print just the relevent line indicating what failed.  
  import re  
  result = re.findall(r"^\s*Caused by:\s*\S+:\s*(.*)$", e.message, flags=re.MULTILINE)  
  if result:  
    print result[-1] # Print only the relevant error message  
  else:  
    print e # Otherwise print the whole stack trace. 
    
# Mount the Historical Data Source
storageName = dataSource 
blobName = "spucket"
accessKey = "RoKgP5hyfipvPM6GQjZH0wnPCeXXQDhz+53s0Bvr94ZXhOFFEUd2mJZXPSaAp64uTDOleBYNpQkThYTOAFwpTA==" 

try:  
  dbutils.fs.mount(  
    source = "wasbs://" + storageName + "@" + blobName + ".blob.core.windows.net/", mount_point = "/mnt/adfdata-historical",  
	extra_configs = {"fs.azure.account.key."+blobName+".blob.core.windows.net": accessKey})  
except Exception as e:  
  mountExceptionHandler(e)
  
# Mount the Model Blob Storage Sink
storageName = modelSource  
	  
try:  
  dbutils.fs.mount(  
    source = "wasbs://" + storageName + "@" + blobName + ".blob.core.windows.net/", mount_point = "/mnt/adfdata-models",  
	extra_configs = {"fs.azure.account.key."+blobName+".blob.core.windows.net": accessKey})  
except Exception as e:  
   mountExceptionHandler(e) 

print "Blob Storage successfully mounted"

# COMMAND ----------

# IMPORTS - Data Analysis & Modelling
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
import os

# COMMAND ----------

#LOAD DATA - Loop and Merge the data in the historical data blob

# Create Empty Dataframe - for source appendage
field = [StructField('class',StringType(), True),StructField('text', StringType(), True)]
schema = StructType(field)
df = sqlContext.createDataFrame(sc.emptyRDD(), schema)

# Loop & Merge all historical data using Python API - this approach has a 2gb file size limit for a local IO operation
for filename in os.listdir("/dbfs/mnt/adfdata-historical/"):
  print("Found File: " + filename)
  df = df.union(spark.read.format("csv").option("header", "true").load("/mnt/adfdata-historical/" + filename))
  
display(df)

# COMMAND ----------

#BASIC TEXT CLASSIFIER - TRAIN AND RETRAIN ARE THE SAME ROUTINE - TRAINING SET JUST GETS BIGGER

def trainModel():
  # regular expression tokenizer - turns the string into seperate words as a list.
  regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")

  # stop words - remove words which dont contribute to the classification
  add_stopwords = ["http","https","amp","rt","t","c","the"] 
  stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)

  # Transform the data into vectors - restrict classes to words that have a document frequency of 5
  countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

  # Transform the string label into a number representation
  label_stringIdx = StringIndexer(inputCol = "class", outputCol = "label")

  # set seed for reproducibility - Split the dataset
  (trainingData, testData) = df.randomSplit([0.7, 0.3], seed = 100)
  print("Training Dataset Count: " + str(trainingData.count()))
  print("Test Dataset Count: " + str(testData.count()))

  # Build the pipeline to take a raw input and model - using a basic LR
  pipelineModel = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx, LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)])

  # Train the Model
  return pipelineModel.fit(trainingData)

model = trainModel()

# Perform test set classification
predictions = model.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("text","class","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

# Evaluate the results - ACC
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)

# COMMAND ----------

# Save the Model is accuracy is good enough!

if accuracy > 0.85:
  lrModel.write().overwrite().save("/mnt/adfdata-models/latest-model")
  print "Model Promoted to production"
else:
  print "Model not accepted for production"

# COMMAND ----------

# USEFUL REFERENCE COMMANDS
# %sh ls
# %sql
# dbutils.fs.ls("/mnt/adfdata-models")
# dbutils.fs.ls("/mnt/")