from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row

# 1. Iniciar sesión Spark
spark = SparkSession.builder \
    .appName("Predict Football Match Goals with ALS") \
    .getOrCreate()

# 2. Leer Match.csv desde HDFS
file_path = "hdfs:///user/hadoop/Match/Match.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# 3. Seleccionar columnas necesarias y limpiar
df_clean = df.select("home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal") \
             .na.drop()

# 4. Entrenar ALS para goles del local
als_home = ALS(
    userCol="home_team_api_id",
    itemCol="away_team_api_id",
    ratingCol="home_team_goal",
    coldStartStrategy="drop",
    nonnegative=True,
    maxIter=20,
    regParam=0.1,
    rank=10
)
model_home = als_home.fit(df_clean)
predictions_home = model_home.transform(df_clean)

# Evaluar modelo local
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="home_team_goal",
    predictionCol="prediction"
)
rmse_home = evaluator.evaluate(predictions_home)
print(f"RMSE goles del equipo local: {rmse_home:.3f}")

# 5. Entrenar ALS para goles del visitante
als_away = ALS(
    userCol="away_team_api_id",
    itemCol="home_team_api_id",
    ratingCol="away_team_goal",
    coldStartStrategy="drop",
    nonnegative=True,
    maxIter=20,
    regParam=0.1,
    rank=10
)
model_away = als_away.fit(df_clean)
predictions_away = model_away.transform(df_clean)

evaluator.setLabelCol("away_team_goal")
rmse_away = evaluator.evaluate(predictions_away)
print(f"RMSE goles del equipo visitante: {rmse_away:.3f}")

# 6. Guardar modelos en HDFS
output_home = "hdfs:///user/hadoop/models/als_home_goals"
output_away = "hdfs:///user/hadoop/models/als_away_goals"
model_home.save(output_home)
model_away.save(output_away)
print("Modelos guardados en HDFS.")

# 7. Guardar RMSEs en archivo de log en HDFS
log_lines = [
    f"RMSE goles del equipo local: {rmse_home:.3f}",
    f"RMSE goles del equipo visitante: {rmse_away:.3f}"
]

log_rdd = spark.sparkContext.parallelize(log_lines)

# Ruta donde guardar el archivo
log_path = "hdfs:///user/hadoop/models/als_rmse_log"

# Eliminar log previo si existe
fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
path = spark._jvm.org.apache.hadoop.fs.Path(log_path)
if fs.exists(path):
    fs.delete(path, True)

# Guardar log
log_rdd.saveAsTextFile(log_path)
print(f"Log de RMSE guardado en: {log_path}")
