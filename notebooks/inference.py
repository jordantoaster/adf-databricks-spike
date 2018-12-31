# Databricks notebook source
print "Mounting Data Sources & Sinks" # Do not need to mount models, as training will have done this on the cluster already.

# Collect the Parameters Indicating the Model + data location
dbutils.widgets.text("data", "staging-data","")
dbutils.widgets.get("data")
dataSource = getArgument("data")

print dataSource

def mountExceptionHandler(e):
   # The error message has a long stack trace.  This code tries to print just the relevent line indicating what failed.  
  import re  
  result = re.findall(r"^\s*Caused by:\s*\S+:\s*(.*)$", e.message, flags=re.MULTILINE)  
  if result:  
    print result[-1] # Print only the relevant error message  
  else:  
    print e # Otherwise print the whole stack trace. 
    
# Mount the Model Blob Storage Sink
storageName = dataSource
blobName = "spucket"
accessKey = "RoKgP5hyfipvPM6GQjZH0wnPCeXXQDhz+53s0Bvr94ZXhOFFEUd2mJZXPSaAp64uTDOleBYNpQkThYTOAFwpTA==" 

try:  
  dbutils.fs.mount(  
    source = "wasbs://" + storageName + "@" + blobName + ".blob.core.windows.net/", mount_point = "/mnt/adfdata-staging",  
	extra_configs = {"fs.azure.account.key."+blobName+".blob.core.windows.net": accessKey})  
except Exception as e:  
  mountExceptionHandler(e)

# Mount the sink folder for the reports
storageName = "reports"
try:  
  dbutils.fs.mount(  
    source = "wasbs://" + storageName + "@" + blobName + ".blob.core.windows.net/", mount_point = "/mnt/adfdata-report",  
	extra_configs = {"fs.azure.account.key."+blobName+".blob.core.windows.net": accessKey})  
except Exception as e:  
  mountExceptionHandler(e)

print "Blob Storage successfully mounted"

# COMMAND ----------

# IMPORTS - Data Analysis & Modelling
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
import os

# COMMAND ----------

#LOAD DATA - Loop and Merge the data in the staging data blob

# Create Empty Dataframe - for source appendage
field = [StructField('class',StringType(), True),StructField('text', StringType(), True)]
schema = StructType(field)
df = sqlContext.createDataFrame(sc.emptyRDD(), schema)

# Loop & Merge all historical data using Python API - this approach has a 2gb file size limit for a local IO operation
for filename in os.listdir("/dbfs/mnt/adfdata-staging/"):
  print("Found File: " + filename)
  df = df.union(spark.read.format("csv").option("header", "true").load("/mnt/adfdata-staging/" + filename))
  
display(df)

# COMMAND ----------

#LOAD THE MODEL
lrModel = PipelineModel.load("/mnt/adfdata-models/latest-model")

# COMMAND ----------

#RUN INFERENCE
predictions = lrModel.transform(df)

#Get Accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print accuracy

# COMMAND ----------

import time
import datetime
from pyspark.sql.functions import lit,unix_timestamp

# Append the time stamp
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
predictionsTimestamp = predictions.withColumn('timestamp',lit(timestamp))  

# Prepare the probability vector column for writing to a CSV - DROP, even though in real world this will be return.
# The data is expected to have some sense of an ID as well and other info
# Create Report DataFrame with Results & Store
predictionsFiltered = predictionsTimestamp.select("timestamp","prediction", "text")

# Load the old report CSV - this recompiles the partitions
filename = "daily-report.csv"
# resultsDf = spark.read.format("csv").option("header", "true").load("/mnt/adfdata-report/" + filename)
# mergedDF = predictionsFiltered.union(resultsDF) # The union stops the save from working 'job aborted' - WHY?

# Output as a CSV to the reports container - this writes the file in partitions - need a way to merge, or just load in script when I want a csv
# Is there a conversion to pandas I can do first?
predictionsFiltered.write.mode("overwrite").format("csv").option("header","true").csv('/mnt/adfdata-report/' + filename)

# COMMAND ----------

# Container delete
# az storage blob delete-batch --account-name spucket --source reports