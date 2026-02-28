import requests
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------
# Load data
# -------------------------------
DATA_PATH = Path("data/training/creditcard_prepared.csv")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])
y = df["Class"]

# -------------------------------
# Reproduce train/test split
# -------------------------------
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# Combine features and labels for sampling
test_df = X_test.copy()
test_df["Class"] = y_test

# -------------------------------
# Sample 500 from each class
# -------------------------------

sample_size = 50
class_0_sample = test_df[test_df["Class"] == 0].sample(sample_size, random_state=35)
class_1_sample = test_df[test_df["Class"] == 1].sample(sample_size, random_state=35)

# Combine into a single batch
test_sample = pd.concat([class_0_sample, class_1_sample])

# Separate features and labels
X_test_sample = test_sample.drop(columns=["Class"])
y_test_sample = test_sample["Class"].tolist()  # <-- use this for metrics!

# -------------------------------
# Convert features to JSON for API
# -------------------------------
data = X_test_sample.to_dict(orient="records")
response = requests.post("http://127.0.0.1:5000/predict", json=data)
preds = response.json()
y_pred = preds["predictions"]
y_proba = preds["probabilities"]

# -------------------------------
# Compute metrics on sampled batch
# -------------------------------
precision = precision_score(y_test_sample, y_pred)
recall = recall_score(y_test_sample, y_pred)
auprc = average_precision_score(y_test_sample, y_proba)
cm = confusion_matrix(y_test_sample, y_pred)

# -------------------------------
# Save metrics to report
# -------------------------------
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

metrics_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "AUPRC"],
    "Value": [precision, recall, auprc]
})
metrics_df.to_csv(REPORTS_DIR / "api_inference_metrics.csv", index=False)

# Optional: save confusion matrix plot
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")
plt.savefig(REPORTS_DIR / "api_confusion_matrix.png")
plt.close()

print("API inference metrics and plots saved to 'reports/' folder.")