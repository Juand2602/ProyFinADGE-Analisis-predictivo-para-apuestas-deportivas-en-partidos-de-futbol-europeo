#   ProyFinADGE-Análisis predictivo para apuestas deportivas en partidos de fútbol europeo (UEFA)

Este repositorio contiene el código y la documentación para el proyecto final, titulado "Análisis predictivo para apuestas deportivas en partidos de fútbol europeo (UEFA)".

##   Descripción del Proyecto

El proyecto aborda el problema de predecir los resultados de partidos de fútbol profesional y estimar las cuotas (odds) de apuestas deportivas basadas en datos históricos. El objetivo es desarrollar un sistema automatizado que replique el proceso de las casas de apuestas utilizando técnicas de análisis de datos a gran escala.

El sistema busca predecir:

* El resultado del partido (victoria local, empate o victoria visitante).
* La cantidad de goles que anotará cada equipo.
* Las cuotas estimadas para apuestas en función de las probabilidades.

##   Objetivos

* **Objetivo General:** Diseñar un sistema inteligente que prediga el resultado y los goles de un partido de fútbol profesional, y que a partir de estas predicciones, calcule las cuotas u "odds" de apuestas, utilizando técnicas de análisis de datos a gran escala.
* **Objetivos Específicos:**
    * Recolectar y procesar un conjunto de datos históricos de partidos de fútbol, integrando estadísticas de los equipos y cuotas reales de apuestas.
    * Identificar y utilizar bases de datos públicas relevantes que contengan estadísticas detalladas de partidos, como el "European Soccer Database" y datos de Understat.
    * Entrenar modelos de machine learning escalables para clasificación (resultado del partido) y regresión (cantidad de goles) utilizando PySpark.
    * Desarrollar un pipeline de datos que incluya la ingesta, limpieza, transformación de datos, entrenamiento de modelos y generación de predicciones.
    * Evaluar el rendimiento de los modelos utilizando métricas apropiadas (e.g., accuracy, F1-score) y comparar las cuotas generadas con las cuotas reales del mercado.
    * Calcular el valor esperado de las apuestas basándose en las probabilidades ajustadas por el modelo.

##   Conceptos Teóricos Clave

* Machine Learning (ML): Clasificación (e.g., RandomForestClassifier), Regresión (e.g., RandomForestRegressor)
* Cuotas (Odds) en Apuestas Deportivas
* Valor Esperado (Expected Value - EV)
* Métricas de Evaluación: Accuracy, F1-Score
* Análisis de Datos a Gran Escala (Big Data)
* Web Scraping

##   Dataset

El dataset utilizado es una combinación de:

* Datos Base: European Soccer Database (Kaggle)
* Datos Recopilados vía Web Scraping: Understat.com (temporadas 2016/17 - 2024/25)

El dataset contiene información de partidos, equipos, jugadores, y cuotas de apuestas.

##   Metodología

La metodología incluye:

* Uso de PySpark para procesamiento distribuido.
* Desarrollo de un pipeline de datos para el procesamiento, modelado y predicción.
* Implementación de modelos de Machine Learning para clasificación (RandomForestClassifier) y regresión (RandomForestRegressor).
* Cálculo del valor esperado para evaluar la rentabilidad de las apuestas.

![Diagrama del Pipeline](https://github.com/Juand2602/ProyFinADGE-Analisis-predictivo-para-apuestas-deportivas-en-partidos-de-futbol-europeo/blob/main/pipeline.png) 

##   Herramientas y Tecnologías

* Apache Spark (PySpark)
* Google Colab + Drive
* Scrap (requests, cloudscraper, BeautifulSoup, re, json, pandas)
* Scikit-learn
* Matplotlib / Seaborn
* Pandas / NumPy
* HDFS

##   Resultados

El modelo de clasificación alcanzó un accuracy del 52.22% y un F1-score de 43.49%.

Se exploraron visualizaciones de la relación entre características (e.g., agresividad, velocidad) y resultados del partido.

##   Conclusiones

* La calidad de los datos es fundamental.
* La predicción deportiva es un desafío complejo.
* PySpark es una herramienta poderosa para el análisis de datos a gran escala.
* La ingeniería de características es crucial para mejorar el rendimiento del modelo.

##   Instrucciones de Uso

1.  Clonar el repositorio
2.  Configurar el entorno con las librerías necesarias (ver "Herramientas y Tecnologías").
3.  Ejecutar los scripts.
4.  Consultar la documentación adicional para detalles sobre la configuración de Spark y la ejecución en el clúster.

###   Detalles de Ejecución

* Asegúrese de tener Python 3.10 instalado.
* Instale las dependencias.
* Para ejecutar los scripts de PySpark, Spark debe estar instalado y configurado correctamente en su entorno. Esto puede implicar configurar las variables de entorno `SPARK_HOME` y `PYSPARK_PYTHON`.
* Los scripts pueden tener argumentos de línea de comandos para especificar archivos de entrada/salida, parámetros del modelo, etc. Consulte el encabezado de cada script o la documentación interna para obtener más detalles.
* Si se utiliza Google Colab, asegúrese de montar Google Drive correctamente para acceder a los datos y guardar los resultados.
