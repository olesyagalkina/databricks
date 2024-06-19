# Databricks notebook source
# MAGIC %md
# MAGIC ### Ingest data

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r /dbfs/spark_lab
# MAGIC mkdir /dbfs/spark_lab
# MAGIC url='https://drive.google.com/uc?export=download&id=10ykFazw1WGoW6ZLkBLinDPW_WvcE-A19'
# MAGIC wget -O /dbfs/spark_lab/waze.csv $url

# COMMAND ----------

df = spark.read.load('spark_lab/*.csv', format='csv')
display(df.limit(10))

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
waze_schema = StructType([
    StructField("ID", IntegerType()),
    StructField("label", StringType()),
    StructField("sessions", IntegerType()),
    StructField("drives", IntegerType()),
    StructField("total_sessions", FloatType()),
    StructField("n_days_after_onboarding", IntegerType()),
    StructField("total_navigations_fav1", IntegerType()),
    StructField("total_navigations_fav2", IntegerType()),
    StructField("driven_km_drives", FloatType()),
    StructField("duration_minutes_drives", FloatType()),
    StructField("activity_days", IntegerType()),
    StructField("driving_days", IntegerType()),
    StructField("device", StringType())
])
df = spark.read.load('/spark_lab/*.csv', format='csv', schema=waze_schema, header=True)
display(df.limit(10))


# COMMAND ----------

from pyspark.sql.functions import col, round, count, lit

total_count = df.count()
label_counts = df.groupBy("label").count()

# Calcular a porcentagem de cada valor
label_percentages = label_counts.withColumn("percentage", round((col("count") / total_count) * 100, 3))

label_percentages.show()

# COMMAND ----------

null_counts = df.select([count(when(isnull(c), c)).alias(c) for c in df.columns])
null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Coluna "label" tem 700 valores ausentes.

# COMMAND ----------

# MAGIC %md
# MAGIC Criamos uma coluna nova que representa o número médio de quilômetros percorridos em cada dia no último mês para cada usuário, chamamos de `km_per_driving_day`.

# COMMAND ----------

# Criar a coluna `km_per_driving_day`
df = df.withColumn("km_per_driving_day", col("driven_km_drives") / col("driving_days"))

# Verificar a estatística descritiva
df.select("km_per_driving_day").describe().show()

# COMMAND ----------

# Contar valores nulos na `km_per_driving_day`
count_nulls = df.filter(df["km_per_driving_day"].isNull()).count()

# Mostrar o resultado
print(f"Quantidade de valores nulos {count_nulls}")

# COMMAND ----------

import math
coluna = "km_per_driving_day"
count_positive_infinite = df.filter(col(coluna) == float('inf')).count()

# Mostrar o resultado
print(f"Quantidade de valores infinitos: {count_positive_infinite}")


# COMMAND ----------

# MAGIC %md
# MAGIC Agora criamos uma coluna nova `percent_sessions_in_last_month` para mostrar a porcentagem do total de sessões de cada usuário que foram registradas no último mês de uso.

# COMMAND ----------

# Criar a coluna 'percent_sessions_in_last_month'
df = df.withColumn("percent_sessions_in_last_month", col("sessions") / col("total_sessions"))

# Verificar a estatística descritiva
df.select("percent_sessions_in_last_month").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Criamos a `professional_driver` com valores binários, i.é, 1 para usuários que tiveram 60 ou mais viagens **e** dirigiram mais de 15 dias no último mês.

# COMMAND ----------

# Criar a coluna 'professional_driver'
df = df.withColumn("professional_driver", when((col("drives") >= 60) & (col("driving_days") >= 15), 1).otherwise(0))

# COMMAND ----------

# MAGIC %md
# MAGIC Agora, criamos uma nova coluna `total_sessions_per_day` que represente o número médio de sessões por dia desde que o usuário se cadastrou no aplicativo.

# COMMAND ----------

# Criar a coluna 'total_sessions_per_day'
df = df.withColumn("total_sessions_per_day", col("total_sessions") / col("n_days_after_onboarding"))

# Verificar a estatística descritiva
df.select("total_sessions_per_day").describe().show()


# COMMAND ----------

# MAGIC %md
# MAGIC Criamos uma coluna `km_per_hour` representando a média de quilômetros por hora percorridos no último mês.

# COMMAND ----------

# Criar a coluna 'km_per_hour'
df = df.withColumn("km_per_hour", col("driven_km_drives") / (col("duration_minutes_drives") / 60))

# Verificar a estatística descritiva
df.select("km_per_hour").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Esses números são claramente problemáticos, e seria válido buscar esclarecimentos sobre como esses dados são coletados, para entender melhor por que velocidades irreais estão sendo observadas.

# COMMAND ----------

# MAGIC %md
# MAGIC Criamos uma coluna `km_per_drive` representando o número médio de quilômetros percorridos por viagem no último mês para cada usuário.

# COMMAND ----------

df = df.withColumn("km_per_drive", col("driven_km_drives") / col("drives"))
df.select("km_per_drive").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Por fim, criamos uma nova coluna `percent_of_sessions_to_favorite` que representa a porcentagem do total de sessões que foram utilizadas para navegar até um dos locais favoritos dos usuários. Como o `total drives since onboarding` não está neste conjunto de dados, o `total_sessions` deve servir como uma aproximação razoável. As pessoas cujas viagens para locais menos favoráveis representam uma porcentagem mais elevada do total de viagens podem ter menos probabilidade de abandonar o aplicativo, visto que estão realizando mais viagens para locais menos familiares.

# COMMAND ----------

# Criar a coluna 'percent_of_drives_to_favorite'
df = df.withColumn("percent_of_drives_to_favorite", 
  (col("total_navigations_fav1") + col("total_navigations_fav2")) / col("total_sessions"))

# Verificar a estatística descritiva
df.select("percent_of_drives_to_favorite").describe().show()

# COMMAND ----------

null_counts = df.select([count(when(isnull(c), c)).alias(c) for c in df.columns])

null_counts.show()

# COMMAND ----------

# Eliminar linhas com valores nulos nas colunas especificadas
df = df.na.drop(subset=['label', 'km_per_driving_day', 'km_per_drive'])


# COMMAND ----------

null_counts = df.select([count(when(isnull(c), c)).alias(c) for c in df.columns])

null_counts.show()

# COMMAND ----------

df.drop('ID')

# COMMAND ----------

total_count = df.count()
label_counts = df.groupBy("label").count()

# Calcular a porcentagem de cada valor
label_percentages = label_counts.withColumn("percentage", round((col("count") / total_count) * 100, 3))

label_percentages.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the data

# COMMAND ----------

splits = df.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
print ("Training Rows:", train.count(), " Testing Rows:", test.count())


# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature engineering

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
#train = labelIndexer.fit(train).transform(train).drop("label")
train = labelIndexer.fit(train).transform(train) # não pode apagar "label", vai usar em pipeline

indexer = StringIndexer(inputCol="device", outputCol="device2")
#indexedData = indexer.fit(train).transform(train).drop("device")
indexedData = indexer.fit(train).transform(train)

display(indexedData)

# COMMAND ----------

# MAGIC %md
# MAGIC We need to scale multiple column values at the same time, so the technique we use is to create a single column containing a vector (essentially an array) of all the numeric features, and then apply a scaler to produce a new vector column with the equivalent normalized values. Use the following code to normalize the numeric features and see a comparison of the pre-normalized and normalized vector columns.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, MinMaxScaler

# Create a vector column containing all numeric features
numericFeatures = ["sessions","drives","total_sessions","n_days_after_onboarding","total_navigations_fav1","total_navigations_fav2","driven_km_drives","duration_minutes_drives","activity_days","driving_days","km_per_driving_day","percent_sessions_in_last_month","professional_driver","total_sessions_per_day","km_per_hour","km_per_drive","percent_of_drives_to_favorite","device2"]
numericColVector = VectorAssembler(inputCols=numericFeatures, outputCol="numericFeatures")
vectorizedData = numericColVector.transform(indexedData)
   
# Use a MinMax scaler to normalize the numeric values in the vector
minMax = MinMaxScaler(inputCol = numericColVector.getOutputCol(), outputCol="normalizedFeatures")
scaledData = minMax.fit(vectorizedData).transform(vectorizedData)
   
# Display the data with numeric feature vectors (before and after scaling)
compareNumerics = scaledData.select("numericFeatures", "normalizedFeatures")
display(compareNumerics)


# COMMAND ----------

# MAGIC %md
# MAGIC The numericFeatures column in the results contains a vector for each row. The vector includes four unscaled numeric values.
# MAGIC The normalizedFeatures column also contains a vector for each penguin observation, but this time the values in the vector are normalized to a relative scale based on the minimum and maximum values for each measurement.

# COMMAND ----------

# MAGIC %md
# MAGIC create a single column containing all of the features (the encoded categorical features), and another column containing the class label we want to train a model to predict (label).

# COMMAND ----------

featVect = VectorAssembler(inputCols=["normalizedFeatures"], outputCol="featuresVector")
preppedData = featVect.transform(scaledData)[col("featuresVector").alias("features"), col("indexedLabel").alias("label")]
display(preppedData)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Train a machine learning model

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Cria o classificador RandomForestClassifier
rf = RandomForestClassifier(labelCol='label', featuresCol='features', seed=42)

# Define CrossValidator
paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [5, 10]) \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .build()

# Define a métrica de avaliação
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
crossval = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=4)

# Ajustar o modelo usando CrossValidator
cv_model = crossval.fit(preppedData)

print ("Model trained!")


# COMMAND ----------

# Prepare the test data
labelIdTesData = labelIndexer.fit(test).transform(test).drop("label")
indexedTestData = indexer.fit(labelIdTesData).transform(labelIdTesData).drop("device")
vectorizedTestData = numericColVector.transform(indexedTestData)
scaledTestData = minMax.fit(vectorizedTestData).transform(vectorizedTestData)
preppedTestData = featVect.transform(scaledTestData)[col("featuresVector").alias("features"), col("indexedLabel").alias("label")]
   

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Avaliar o modelo no conjunto de teste
predictions = cv_model.transform(preppedTestData)
accuracy = evaluator.evaluate(predictions)
precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
f1 = evaluator.setMetricName("f1").evaluate(predictions)

# Exibir as métricas
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# COMMAND ----------

# Avaliando o modelo usando MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Calculando a matriz de confusão
conf_matrix = predictions.groupBy("label", "prediction").count().orderBy("label", "prediction")

# Exibindo a matriz de confusão
conf_matrix.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Verdadeiros Negativos (TN)**: O modelo corretamente previu 3290 casos onde o rótulo verdadeiro era 0.0 (negativo).
# MAGIC
# MAGIC **Falsos Positivos (FP)**: O modelo incorretamente previu 4 casos como positivos (1.0), quando na verdade eram negativos (0.0).
# MAGIC
# MAGIC **Falsos Negativos (FN)**: O modelo incorretamente previu 629 casos como negativos (0.0), quando na verdade eram positivos (1.0).
# MAGIC
# MAGIC **Verdadeiros Positivos (TP)**: O modelo corretamente previu 11 casos onde o rótulo verdadeiro era 1.0 (positivo).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier

alvo = "label"
catFeature = "device"
numFeatures = ["sessions","drives","total_sessions","n_days_after_onboarding","total_navigations_fav1","total_navigations_fav2","driven_km_drives","duration_minutes_drives","activity_days","driving_days","km_per_driving_day","percent_sessions_in_last_month","professional_driver","total_sessions_per_day","km_per_hour","km_per_drive","percent_of_drives_to_favorite"]
   
# Define the feature engineering and model training algorithm steps
alvoIndexer = StringIndexer(inputCol=alvo, outputCol=alvo + "id")
catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "2")
numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
#featVect = VectorAssembler(inputCols=["normalizedFeatures"], outputCol="Features")
featVect = VectorAssembler(inputCols=["normalizedFeatures", catFeature + "2"], outputCol="Features")
rf = RandomForestClassifier(labelCol="labelid", featuresCol="Features", seed=42)

# Chain the steps as stages in a pipeline
pipeline = Pipeline(stages=[alvoIndexer, catIndexer, numVector, numScaler, featVect, rf])

paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [5, 10]) \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .build()

evaluator = MulticlassClassificationEvaluator(labelCol='labelid', predictionCol='prediction', metricName='accuracy')
crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=4)

# Ajustar o modelo usando pipeline
model = crossval.fit(train)

print("Model trained!")

# COMMAND ----------

# MAGIC %md
# MAGIC Use the following code to apply the pipeline to the test data.

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("label").alias("trueLabel"))
display(predicted)
