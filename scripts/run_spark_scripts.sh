#!/bin/sh
# run_spark_scripts.sh

docker exec -it spark bash -c "spark-submit /app/scripts/spark_script.py"
