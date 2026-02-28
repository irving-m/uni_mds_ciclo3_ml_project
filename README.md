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

Our objective is not only to build a strong model well fitted to the  task at hand, but also to develop a production-style ML pipeline.

## Problem

Credit card fraud detection is a highly imbalanced classification problem, where fraudulent transactions represent only a small fraction of all operations.

Key challenges:

- Extreme class imbalance
- High cost of false negatives
- Need for probabilistic predictions
- Reproducible experimentation
  
<img width="730" height="572" alt="imagen" src="https://github.com/user-attachments/assets/6e455e79-eb9e-412d-a32d-391e74190d0b" />

  
Put here the description, implementation doc, info, results, etc about your work.
You can also use links/reference to other documents/files form this repository or outside resources.

## Preparing the data
Run:

```
python src/data_preparation.py
```
1. It downloads the Kaggle dataset for Credit card fraud detection
2. Performs data transformation on the columns Amount (log scaling) and Time (standard scaling)
3. Saves the prepared dataset locally

Other columns don't need transformations since they come from a PCA process according to the author of the dataset.
<img width="736" height="552" alt="imagen" src="https://github.com/user-attachments/assets/8c1ed95d-e155-4122-a480-0a549649fa92" />
<img width="716" height="546" alt="imagen" src="https://github.com/user-attachments/assets/5aabfc45-f916-404d-91b6-be75dce18598" />
<img width="730" height="543" alt="imagen" src="https://github.com/user-attachments/assets/d3283a5f-cf1e-416e-8f8b-2b2254e36960" />

## Selecting the Model
1. Define AUPRC as our target metric, imbalanced nature of the dataset. 
2. Define logistic regression as baseline model with 0.63 AUPRC
3. Train 4 different models:
- Random Forest
- Extra Trees
- HistGradientBoosting
- XGBoosting
4. Log the results with MLFlow, stored locally on mlruns/; inspect the results with
  ```
  mlflow ui
  ```

<img width="1113" height="506" alt="imagen" src="https://github.com/user-attachments/assets/d67e0a98-242e-4e56-979b-e6f95f97c81c" />

This XGBoost model was selected due to its high AUPRC and recall.


## Training the Model
Run:

```bash
python src/train.py
```

The training pipeline:

1. Loads prepared dataset
2. Performs stratified train/test split (85% train / 15% test)
3. Trains an XGBoost classifier
4. Logs parameters, metrics, and model artifacts with MLflow
5. Saves a deployable model locally

<img width="1109" height="509" alt="imagen" src="https://github.com/user-attachments/assets/047a9765-0fe8-421a-8fb5-23ebdd1dfce5" />

## Serving the model
Run 

```bash
python src/serving.py
```

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

Reproducibility

- Fixed random seeds
- Stratified splits
- MLflow experiment tracking
- Saved model artifacts
- Large datasets are excluded from the repository via .gitignore.
