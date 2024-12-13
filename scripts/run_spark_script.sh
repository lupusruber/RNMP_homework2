#!/bin/sh
# run_spark_script.sh

docker exec -it spark bash -c "spark-submit /app/scripts/spark_script.py"
