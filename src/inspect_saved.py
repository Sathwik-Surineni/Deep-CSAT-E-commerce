# src/inspect_saved.py
import joblib
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

eval_summary = joblib.load(MODEL_DIR / "eval_summary.joblib")
print("Eval summary:", eval_summary)

# list models
for p in MODEL_DIR.glob("*.joblib"):
    print("Model file:", p.name)
