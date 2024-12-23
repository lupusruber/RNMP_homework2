#!/bin/sh
# main_script.sh

echo "Getting the data needed for this project"
source data/get_data_script.sh
echo "Data Loaded Successfully"

cd scripts
mkdir best_model
mkdir checkpoints
cd ..
echo "Created dirs inside the scripts folder"

chmod -R 777 ./data ./scripts ./checkpoints
echo "Creating Spark Cluster inside Docker Compose"
docker-compose up -d
echo "Created Spark Cluster inside Docker Compose"

docker exec -it spark bash -c "pip install numpy"


echo "Sleeping for 5 seconds before executing scripts"
sleep 5

echo "Executing spark scripts inside Docker"
source scripts/run_spark_script.sh

echo "Script finished successfully"