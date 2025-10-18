# src/predict_demo.py
"""
Safe prediction demo that recreates the minimal preprocessing expected by the saved pipeline.
"""

import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"  # change if your best model filename differs
DATA_PATH = PROJECT_ROOT / "data" / "eCommerce_Customer_support_data.csv"

# --- helpers that mirror train.py preprocessing ---
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def create_joined_text_and_lengths(df: pd.DataFrame, text_candidates):
    # Ensure the candidate columns exist; fall back gracefully
    existing = [c for c in text_candidates if c in df.columns]
    if len(existing) == 0:
        df['_joined_text_'] = ""
    else:
        df['_joined_text_'] = df[existing].fillna("").astype(str).agg(' '.join, axis=1)
    df['_joined_text__len'] = df['_joined_text_'].str.len().fillna(0).astype(int)
    return df

def create_date_features(df: pd.DataFrame):
    # date-like columns same heuristic as train.py
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

# --- Load model ---
model = joblib.load(MODEL_PATH)
print("Loaded model:", MODEL_PATH.name)

# --- Load data and pick a few samples to demo ---
df = pd.read_csv(DATA_PATH)
df = standardize_columns(df)

# Choose some rows to demo: either sample or specific indexes
sample = df.sample(n=3, random_state=42).reset_index(drop=True)
print("Original sample (first 3 rows preview):")
print(sample.head(3).T)

# --- Recreate minimal preprocessing used in training ---
# Text candidates heuristic from train.py
text_candidates = [c for c in df.columns if any(k in c for k in ['message','remark','comment','text','body','customer_remarks'])]
if len(text_candidates) == 0 and 'customer_remarks' in df.columns:
    text_candidates = ['customer_remarks']

sample = create_joined_text_and_lengths(sample, text_candidates)
sample = create_date_features(sample)

# Note: train.py dropped some high-cardinality id columns (unique_id, order_id, agent_name, supervisor, manager)
# It's okay to keep them in the DataFrame; ColumnTransformer will pick only the columns it needs.
# But if you want to mimic exactly, you can drop them:
for c in ['unique_id', 'order_id', 'agent_name', 'supervisor', 'manager']:
    if c in sample.columns:
        # keep the column but it's safe to leave it; uncomment next line to actually drop
        # sample = sample.drop(columns=[c])
        pass

# --- Predict ---
X_demo = sample  # pipeline expects a DataFrame with the same column names used at training
preds = model.predict(X_demo)
probs = model.predict_proba(X_demo)[:, 1] if hasattr(model, "predict_proba") else None

print("\nPredictions for the sample rows:")
for i in range(len(X_demo)):
    print("----")
    print("index:", X_demo.index[i])
    print("pred:", int(preds[i]))
    if probs is not None:
        print("prob:", float(probs[i]))
    # show short customer remark context if exists
    if 'customer_remarks' in X_demo.columns:
        print("customer_remarks:", str(X_demo.loc[X_demo.index[i], 'customer_remarks'])[:200])
    else:
        print("customer_remarks: (no column)")
