from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Row

# 1. Iniciar sesión Spark
spark = SparkSession.builder \
    .appName("Predict Single Football Match Goals") \
    .getOrCreate()

# 2. Cargar modelos desde HDFS
path_home_model = "hdfs:///user/hadoop/models/als_home_goals"
path_away_model = "hdfs:///user/hadoop/models/als_away_goals"

model_home = ALSModel.load(path_home_model)
model_away = ALSModel.load(path_away_model)

# 3. Crear DataFrame con el partido a predecir
home_team_id = 8634
away_team_id = 8633

match_df_home = spark.createDataFrame([Row(home_team_api_id=home_team_id, away_team_api_id=away_team_id)])
match_df_away = spark.createDataFrame([Row(away_team_api_id=away_team_id, home_team_api_id=home_team_id)])

# 4. Predecir goles esperados del local y visitante
prediction_home = model_home.transform(match_df_home).select("prediction").collect()[0][0]
prediction_away = model_away.transform(match_df_away).select("prediction").collect()[0][0]

# 5. Preparar resultado como DataFrame
result_data = [(home_team_id, away_team_id, float(prediction_home), float(prediction_away))]
result_df = spark.createDataFrame(result_data, ["home_team_api_id", "away_team_api_id", "pred_home_goals", "pred_away_goals"])

# 6. Guardar resultado en HDFS (sobrescribe si existe)
output_path = "hdfs:///user/hadoop/models/predicted_match_result"

# Borrar si ya existe
fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
path = spark._jvm.org.apache.hadoop.fs.Path(output_path)
if fs.exists(path):
    fs.delete(path, True)

result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

print(f"Predicción guardada en: {output_path}")

# 7. Cerrar sesión
spark.stop()
