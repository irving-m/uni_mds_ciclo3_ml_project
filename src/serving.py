from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Load serialized model
# --------------------------------------------------
MODEL_PATH = Path("models/xgboost_fraud_model.joblib")
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Initialize Flask app
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Receive JSON data
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # 2. Convert to DataFrame
        df = pd.DataFrame(input_data)

        # 3. Make predictions
        y_pred_class = model.predict(df).tolist()
        y_pred_proba = model.predict_proba(df)[:, 1].tolist()  # probability of class 1

        # 4. Return results
        return jsonify({
            "predictions": y_pred_class,
            "probabilities": y_pred_proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# Run API
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)