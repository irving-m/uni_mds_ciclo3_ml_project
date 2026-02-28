import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score
)

from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost


def main():
    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    DATA_PATH = Path("data/training/creditcard_prepared.csv")
    MODEL_DIR = Path("models")
    MLRUNS_DIR = Path("mlruns")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    print("Loading prepared data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]


    # --------------------------------------------------
    # FINAL SPLIT
    # (test set says untouched until now)
    # --------------------------------------------------
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )


    # --------------------------------------------------
    # Champion Model (XGBoost)
    # --------------------------------------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


    # --------------------------------------------------
    # MLflow setup
    # --------------------------------------------------
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.resolve()}")
    mlflow.set_experiment("creditcard_fraud_detection")


    with mlflow.start_run(run_name="xgboost_final_training"):

        print("Training model...")
        model.fit(X_train_full, y_train_full)

        # ----------------------------------------------
        # Predictions
        # ----------------------------------------------
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # ----------------------------------------------
        # Metrics (TRUE performance)
        # ----------------------------------------------
        auprc = average_precision_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"AUPRC (test): {auprc:.4f}")
        print(f"Precision (test): {precision:.4f}")
        print(f"Recall (test): {recall:.4f}")

        # ----------------------------------------------
        # Log params
        # ----------------------------------------------
        mlflow.log_param("model", "XGBoost")
        mlflow.log_params(model.get_params())

        # ----------------------------------------------
        # Log metrics
        # ----------------------------------------------
        mlflow.log_metric("AUPRC_test", auprc)
        mlflow.log_metric("precision_test", precision)
        mlflow.log_metric("recall_test", recall)

        # ----------------------------------------------
        # Log model artifact
        # ----------------------------------------------
        mlflow.xgboost.log_model(model, name="model")

        # ----------------------------------------------
        # Save local copy (deployment artifact)
        # ----------------------------------------------
        model_path = MODEL_DIR / "xgboost_fraud_model.joblib"
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)

        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            name="CreditCard_XGB"
        )
        
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
