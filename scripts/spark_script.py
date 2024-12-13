# spark_script.py

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StructField, StructType, LongType

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.mllib.evaluation import RankingMetrics

import logging
from pathlib import Path

PROJECT_LOCATION = Path(__file__).parent.parent
CHECKPOINT_DIR = Path(f"{PROJECT_LOCATION}/scripts/checkpoints")
DATASET_PATH = Path(f"{PROJECT_LOCATION}/data/u.data")
MODEL_PATH = Path(f'{PROJECT_LOCATION}/scripts/best_model/best_model.model')

K = 10

logger = logging.getLogger('spark_ml_script')
logging.basicConfig(level=logging.INFO)



def get_spark_session(
    master: str = "local[*]", app_name: str = "ALSModel", log_level: str = "INFO"
) -> SparkSession:

    spark: SparkSession = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.checkpoint.dir", CHECKPOINT_DIR)
        .getOrCreate()
    )

    spark.sparkContext.setCheckpointDir(CHECKPOINT_DIR)
    # spark.sparkContext.setLogLevel(log_level)
    spark.conf.set("spark.sql.shuffle.partitions", "4")

    return spark


def get_data(spark: SparkSession) -> tuple[DataFrame, DataFrame]:

    schema = StructType(
        [
            StructField("user_id", LongType(), True),
            StructField("item_id", LongType(), True),
            StructField("rating", LongType(), True),
            StructField("timestamp", LongType(), True),
        ]
    )

    ratings = spark.read.csv(DATASET_PATH, sep="\t", header=True, schema=schema)

    train_data, test_data = ratings.randomSplit([0.8, 0.2], seed=42)

    return train_data, test_data


def get_best_model(train_data: DataFrame, evaluator: RegressionEvaluator) -> ALS:

    if MODEL_PATH.exists():

        logger.info("Best Model found")

        return ALSModel.load(MODEL_PATH)

    als = ALS(
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        checkpointInterval=5,
    )

    paramGrid = (
        ParamGridBuilder()
        .addGrid(als.rank, [10, 20, 30])
        .addGrid(als.regParam, [0.01, 0.1, 1.0])
        .addGrid(als.maxIter, [10, 20, 30]).build()
    )

    cv = CrossValidator(
        estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3
    )

    cv_model = cv.fit(train_data)
    model = cv_model.bestModel
    model.save(MODEL_PATH)

    return model


def get_predicitions_and_labels(
    model: ALS, test_data: DataFrame, k: int = 10
) -> tuple[DataFrame, DataFrame]:

    predictions = model.transform(test_data)
    user_recs: DataFrame = model.recommendForAllUsers(k)

    actual_items = (
        test_data.groupBy("user_id")
        .agg(F.collect_list("item_id").alias("actual_items"))
        .filter(F.size("actual_items") >= k // 2)
    )

    exploded_recs = user_recs.withColumn(
        "recommendation", F.explode("recommendations")
    ).select("user_id", F.col("recommendation").getField("item_id").alias("item_id"))

    pred_items = exploded_recs.groupBy("user_id").agg(
        F.collect_list("item_id").alias("pred_items")
    )

    prediction_and_labels = actual_items.join(
        pred_items, on="user_id", how="left"
    ).select("pred_items", "actual_items")

    return prediction_and_labels, predictions


def get_metrics(
    prediction_and_labels: DataFrame,
    predictions: DataFrame,
    evaluator: RegressionEvaluator,
    k: int = 10,
) -> None:

    metrics = RankingMetrics(prediction_and_labels.rdd)

    rmse = evaluator.evaluate(predictions)

    logger.info(f"Root Mean Squared Error: {rmse}")
    logger.info(f"Precision@{k}: {metrics.precisionAt(k)}")
    logger.info(f"Recall@{k}: {metrics.recallAt(k)}")
    logger.info(f"NDCG@{k}: {metrics.ndcgAt(k)}")
    logger.info(f"Mean Average Precision: {metrics.meanAveragePrecision}")


if __name__ == "__main__":


    spark = get_spark_session()

    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )

    train_data, test_data = get_data(spark)
    model = get_best_model(train_data, evaluator)
    prediction_and_labels, predictions = get_predicitions_and_labels(
        model, test_data, k=K
    )
    get_metrics(prediction_and_labels, predictions, evaluator, k=K)

    spark.stop()
