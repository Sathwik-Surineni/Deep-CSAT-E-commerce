# src/plot_metrics.py
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
model = joblib.load(MODEL_DIR / "logistic_regression.joblib")  # change if needed

# load test split quickly by reusing training script's preproc (simple way: re-run split logic)
DATA_PATH = MODEL_DIR.parents[0] / "data" / "eCommerce_Customer_support_data.csv"
df = pd.read_csv(DATA_PATH)
# minimal target detection (mirror train.py)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
target = [c for c in df.columns if any(k in c for k in ['csat','satisfaction','rating','score'])][0]
y_raw = df[target]
y = (pd.to_numeric(y_raw, errors='coerce') >= pd.to_numeric(y_raw, errors='coerce').median()).astype(int)

# Sample a test set (we can't easily re-run the exact preprocessing here)
# But to plot using saved model, we need to reproduce the exact split/preprocessing.
# Easiest: reload training script outputs or rerun train.py that saved model and used last split.
# For a quick demo, just show confusion matrix from predictions on a small subset:
X_demo = df.sample(n=2000, random_state=42)
y_demo = (pd.to_numeric(X_demo[target], errors='coerce') >= pd.to_numeric(df[target], errors='coerce').median()).astype(int)
pred = model.predict(X_demo)
probs = model.predict_proba(X_demo)[:, 1] if hasattr(model, "predict_proba") else None

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_demo, pred, normalize=None)
plt.title("Confusion Matrix (sample)")
plt.show()

# ROC curve (if probs available)
if probs is not None:
    fpr, tpr, _ = roc_curve(y_demo, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve (sample)")
    plt.legend()
    plt.show()
