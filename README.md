# MLOps Introduction: Final Project
FInal work description in  the [final_project_description.md](final_project_description.md) file.

Student info:

- Full name: Irving Pérez
- e-mail: iperezh@uni.pe
- Grupo: 2

## Project Name: Credit Card Fraud Detection - End-to-End MLOps Project

This project implements and end-to-end machine learning workflow for credit card fraud detection, it covers:

- Data extraction and preparation
- Exploratory Data Analysis
- Model baseline and model selection with MLFlow tracking
- Model serving via API
- Inference testing
- Automated reporting 

Our objective is not only to build a strong model well fitted to the task at hand, but also to develop a production-style ML pipeline.

## Problem

Credit card fraud detection is a highly imbalanced classification problem, where fraudulent transactions represent only a small fraction of all operations.

Key challenges:

- Extreme class imbalance
- High cost of false negatives
- Need for probabilistic predictions
- Reproducible experimentation
  

# Preparing the data
Run:

```
python src/data_preparation.py
```
1. It downloads the Kaggle dataset for Credit card fraud detection
2. Performs data transformation on the columns Amount (log scaling) and Time (standard scaling)
3. Saves the prepared dataset locally
   
## Key findings
- Most columns come from a PCA process according to the dataset authors, so they're already standardized.
- The amount column is heavily skewed, so a log-scaling transformation should be performed.
- The time and log-amount columns are standardized to improve later training.

<img width="736" height="552" alt="imagen" src="https://github.com/user-attachments/assets/8c1ed95d-e155-4122-a480-0a549649fa92" />
<img width="716" height="548" alt="image" src="https://github.com/user-attachments/assets/005784d3-8066-44bd-be92-2674fa9f98b2" />
<img width="730" height="543" alt="imagen" src="https://github.com/user-attachments/assets/d3283a5f-cf1e-416e-8f8b-2b2254e36960" />

# Selecting the Model
1. Define AUPRC as our key performance metric, due to the imbalanced nature of the dataset. 
2. Define logistic regression as *baseline model* with 0.63 AUPRC
3. Train 4 different models:
- Random Forest
- Extra Trees
- HistGradientBoosting
- XGBoosting
4. Log the results with MLFlow

<img width="1113" height="506" alt="imagen" src="https://github.com/user-attachments/assets/d67e0a98-242e-4e56-979b-e6f95f97c81c" />

The results of model trials are stored locally on mlruns/, they can be accessed with:
``` 
mlflow ui
```
 **The XGBoost model was selected due to its high AUPRC (0.83) and recall (0.76)**

# Training the Model
Run:

```bash
python src/train.py
```

## The training pipeline:

1. Loads prepared dataset
2. Performs stratified train/test split (85% train / 15% test)
3. Trains an XGBoost classifier
4. Logs parameters, metrics, and model artifacts with MLflow
5. Saves a deployable model locally

<img width="1109" height="509" alt="imagen" src="https://github.com/user-attachments/assets/047a9765-0fe8-421a-8fb5-23ebdd1dfce5" />

# Serving the model
Run 

```bash
python src/serving.py
```

1. Set up a flask api with the serialized model
2. Make inference with scripts in test/
3. Inference results are reporte to reports/
   
Server runs at:

```bash
http://127.0.0.1:5000
```

The API can be tested with sample.py to test individual json strings, and inference.py for batch inference.
``` 
tests/sample.py
tests/inference.py
```

Reports from inference.py are saved to reports/

<img width="634" height="510" alt="imagen" src="https://github.com/user-attachments/assets/180a76b6-6db9-4136-9be9-0039e0a0717c" />

# Conclusion
We were able to train Machine Learning model, obtain and prepare data for credit card fraud, and serve the model in a comfortable fashion, and make good inference with it (0.85 AUPRC)

# Improvements and lessons
- The model can still be improved with hyperparameter tuning
- We can choose different thresholds to improve our AUPRC or recall
- The model serving can be improved with a more user friendly interface via flask
- Datasets were commited accidentaly, causing several issues in later commits, we should always add *.csv to .gitignore
