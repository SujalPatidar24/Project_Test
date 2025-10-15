from flask import Flask, render_template, request
import pandas as pd
import joblib
import time
import os
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Initialize Flask App
# ----------------------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------------------
# Load Model and Frequency Maps
# ----------------------------------------------------------------
MODEL_PATH = "models/lr_model.pkl"
FREQ_MAP_PATH = "models/freq_maps.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")

model = joblib.load(MODEL_PATH)
freq_maps = joblib.load(FREQ_MAP_PATH) if os.path.exists(FREQ_MAP_PATH) else {}

# ----------------------------------------------------------------
# Prometheus Metrics
# ----------------------------------------------------------------
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ----------------------------------------------------------------
# Helper: Prepare input for model
# ----------------------------------------------------------------
def prepare_input(form_data):
    """Convert HTML form input to dataframe compatible with model"""
    try:
        data = {k: [float(v) if v.replace('.', '', 1).isdigit() else v] for k, v in form_data.items()}

        df = pd.DataFrame(data)

        # frequency encode brand and product_subcategory if present
        for col in ["brand", "product_subcategory"]:
            if col in df.columns and col in freq_maps:
                df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0.0)

        return df
    except Exception as e:
        raise ValueError(f"Error preparing input: {e}")

# ----------------------------------------------------------------
# Routes
# ----------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    resp = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return resp


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    try:
        input_df = prepare_input(request.form)
        probability = model.predict_proba(input_df)[:, 1][0]
        prediction = int(probability >= 0.5)

        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        result_text = f"Return Probability: {probability:.2%} → {'Returned' if prediction==1 else 'Not Returned'}"
        return render_template("index.html", result=result_text)

    except Exception as e:
        print("❌ Prediction Error:", e)
        return render_template("index.html", result=f"Error: {str(e)}")


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


# ----------------------------------------------------------------
# Run App
# ----------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
