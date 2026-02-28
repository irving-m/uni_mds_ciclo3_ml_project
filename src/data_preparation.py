import kagglehub
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def get_data():
    # project data folder
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    if (target_dir / "creditcard.csv").exists():
        print("Dataset already present.")
        return
    
    # download to kaggle cache
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    # copy dataset into project structure
    for file in Path(path).glob("*"):
        shutil.copy(file, target_dir / file.name)

    print("Dataset stored in:", target_dir)


def transform_data():
    raw_data_path = Path("data/raw/creditcard.csv")
    prepared_data_path = Path("data/training/creditcard_prepared.csv")
    prepared_data_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_data_path)

    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()

    # Amount preprocessing
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Amount_scaled"] = amount_scaler.fit_transform(df[["Amount_log"]])

    # Time preprocessing
    df["Time_scaled"] = time_scaler.fit_transform(df[["Time"]])

    df = df.drop(columns=["Amount", "Amount_log", "Time"])

    df.to_csv(prepared_data_path, index=False)


if __name__ == "__main__":
    get_data()
    transform_data()