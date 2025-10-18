# src/tune_threshold.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "eCommerce_Customer_support_data.csv"

model = joblib.load(MODEL_PATH)
print("Loaded model:", MODEL_PATH.name)

# Load data and preprocess minimal (same as predict_demo)
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')

# create joined text & date features (same logic as train/predict demo)
text_candidates = [c for c in df.columns if any(k in c for k in ['message','remark','comment','text','body','customer_remarks'])]
if len(text_candidates) == 0 and 'customer_remarks' in df.columns:
    text_candidates = ['customer_remarks']

# create joined text
df['_joined_text_'] = df[text_candidates].fillna("").astype(str).agg(' '.join, axis=1)
df['_joined_text__len'] = df['_joined_text_'].str.len().fillna(0).astype(int)

# date features
date_cols = [c for c in df.columns if any(k in c for k in ['date','time','reported','responded'])]
for dc in date_cols:
    tmp = pd.to_datetime(df[dc], errors='coerce', dayfirst=True)
    df[f"{dc}_hour"] = tmp.dt.hour.fillna(-1).astype(int)
    df[f"{dc}_dow"] = tmp.dt.dayofweek.fillna(-1).astype(int)

# target
target_col = [c for c in df.columns if any(k in c for k in ['csat','satisfaction','rating','score'])][0]
y_raw = df[target_col]
y = (pd.to_numeric(y_raw, errors='coerce') >= pd.to_numeric(y_raw, errors='coerce').median()).astype(int)

# Use a sample to speed up
X_sample = df.sample(n=8000, random_state=42)
y_sample = y.loc[X_sample.index]

probs = model.predict_proba(X_sample)[:, 1]
thresholds = np.linspace(0.1, 0.9, 17)

print("threshold | precision_pos | recall_pos | f1_pos | precision_neg | recall_neg | f1_neg")
for t in thresholds:
    preds = (probs >= t).astype(int)
    p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(y_sample, preds, pos_label=1, average='binary')
    p_neg, r_neg, f1_neg, _ = precision_recall_fscore_support(y_sample, preds, pos_label=0, average='binary')
    print(f"{t:.2f}      | {p_pos:.3f}         | {r_pos:.3f}    | {f1_pos:.3f}  | {p_neg:.3f}        | {r_neg:.3f}    | {f1_neg:.3f}")
