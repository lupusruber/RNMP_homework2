
# Рударење на масивни податоци: Домашна работа 2

This project runs a set of scripts to train and eval a recommendation system (ALS, Alternating Least Squares model) in the Spark MLlib with the MovieLens dataset. Below are the project details, setup instructions, and other highlights.

---

## Project Structure

```
.
├── data
│   └── get_data_script.sh
├── docker-compose.yml
├── main_script.sh
├── poetry.lock
├── pyproject.toml
├── README.md
└── scripts
    ├── best_model
    ├── checkpoints
    ├── run_spark_scripts.sh
    └── spark_script.py
```

---

## Setup and Execution

### Prerequisites

- Docker and Docker Compose installed on your machine.
- Python 3.10 or above with `poetry` for dependency management.
- Spark installed in the Docker container.

### Build Project Guide

1. Pull project from Github
```bash
git clone https://github.com/lupusruber/RNMP_homework2.git
cd RNMP_homework2
```

2. Run the main script
```bash
source main_script.sh
```

This script performs the following:
- Downloads and preprocesses the data.
- Sets up a Spark cluster in Docker.
- Trains an ALS model on the MovieLens dataset.
- Evaluates the model's performance using the metrics: RMSE, Precision@K, Recall@K, and NDCG.

---

## Scripts Overview

### `main_script.sh`
Main script to orchestrate data loading, Spark cluster setup, and model training.

### `get_data_script.sh`
Script to download and preprocess the MovieLens dataset.

### `run_spark_scripts.sh`
Script to submit the Spark job (`spark_script.py`) to the Spark cluster.

### `spark_script.py`
Implements the following:
- Loads the MovieLens dataset.
- Splits the data into training and testing sets.
- Trains the ALS model using a cross-validator to determine the best hyperparameters.
- Evaluates the model using Spark's and MLlib's evaluation tools.

---

## Key Features

- **Data Preprocessing**
  - Adds headers to the MovieLens dataset files for easier readability.
  - Converts data encoding to UTF-8.

- **ALS Model**
  - Trained with user-item interaction data.
  - Hyperparameter tuning using cross-validation.

- **Evaluation**
  - Metrics: RMSE, Precision@K, Recall@K, and NDCG are logged for model performance assessment.

---

## Docker Integration

The project uses `docker-compose` for running a Spark cluster. Why?
- Simplified environment setup.
- Consistency across development and production.

---

## Dependencies for dev

Dependencies are managed using `poetry`. Install them with:
```bash
poetry install
```

---

## Outputs

- **Best Model**
  Saved under `scripts/best_model/best_model.model`.

- **Metrics**
  Logged to the console.

---