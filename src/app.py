# src/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

def standardize_and_prepare(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
    # joined text
    text_candidates = [c for c in df.columns if any(k in c for k in ['message','remark','comment','text','body','customer_remarks'])]
    if len(text_candidates) == 0 and 'customer_remarks' in df.columns:
        text_candidates = ['customer_remarks']
    existing = [c for c in text_candidates if c in df.columns]
    if len(existing) == 0:
        df['_joined_text_'] = ""
    else:
        df['_joined_text_'] = df[existing].fillna("").astype(str).agg(' '.join, axis=1)
    df['_joined_text__len'] = df['_joined_text_'].str.len().fillna(0).astype(int)
    # date features (dayfirst)
    date_cols = [c for c in df.columns if any(k in c for k in ['date','time','reported','responded'])]
    for dc in date_cols:
        try:
            tmp = pd.to_datetime(df[dc], errors='coerce', dayfirst=True)
            df[f"{dc}_hour"] = tmp.dt.hour.fillna(-1).astype(int)
            df[f"{dc}_dow"] = tmp.dt.dayofweek.fillna(-1).astype(int)
        except Exception:
            df[f"{dc}_hour"] = -1
            df[f"{dc}_dow"] = -1
    return df

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    # expect a dict where keys are column names -> values single record
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        return jsonify({"error":"invalid payload"}), 400

    df_prepared = standardize_and_prepare(df)
    preds = model.predict(df_prepared)
    probs = model.predict_proba(df_prepared)[:, 1] if hasattr(model, "predict_proba") else None

    results = []
    for i in range(len(df_prepared)):
        r = {"pred": int(preds[i])}
        if probs is not None:
            r["prob"] = float(probs[i])
        results.append(r)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
