# src/train.py
"""
Updated end-to-end training script for DeepCSAT (memory-friendly).

Key improvements over earlier version:
- Drops high-cardinality identifier columns (unique_id, order_id, agent_name, supervisor, manager)
- Converts date/time-like columns into hour and day-of-week
- Treats customer_remarks (and similar) as TF-IDF text (max_features=1000)
- Uses sparse OneHotEncoder (sparse_output=True) to avoid allocating huge dense arrays
- Converts sparse -> dense only for RandomForest pipeline
- Saves best model and eval summary to models/
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# -----------------------
# Config / paths
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "eCommerce_Customer_support_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# -----------------------
# Utility functions
# -----------------------
def load_data(path: Path) -> pd.DataFrame:
    print("Loading data from:", path)
    df = pd.read_csv(path)
    print("Initial shape:", df.shape)
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def detect_target(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if any(k in c for k in ['csat','satisfaction','rating','score'] )]
    if len(candidates) == 0:
        raise ValueError("No candidate target column found. Please rename your CSAT column to include 'csat' or 'satisfaction' or 'rating'. Columns: " + ", ".join(df.columns))
    print("Candidate target columns:", candidates)
    return candidates[0]

# -----------------------
# Main
# -----------------------
def main():
    df = load_data(DATA_PATH)
    df = standardize_columns(df)

    print("Columns:", df.columns.tolist())
    print("Sample rows:")
    print(df.head(5))

    # Detect target column
    target_col = detect_target(df)
    print("Using target column:", target_col)

    # Basic target processing: if numeric, binarize at median; if categorical, attempt mapping
    y_raw = df[target_col]
    if y_raw.dtype == object:
        # attempt to map common text categories -> 0/1
        uniques = list(y_raw.dropna().unique())[:10]
        print("Target sample uniques:", uniques)
        lowered = [str(x).strip().lower() for x in uniques]
        # simple heuristics
        if all(x in ('satisfied','yes','y','true','t','1','5') for x in lowered):
            y = (y_raw.astype(str).str.strip().str.lower().isin(['satisfied','yes','y','true','t','1','5'])).astype(int)
        elif all(x in ('unsatisfied','no','n','false','f','0') for x in lowered):
            y = (y_raw.astype(str).str.strip().str.lower().isin(['satisfied','yes','y','true','t','1','5'])).astype(int)
        else:
            # fallback: if two categories, map first->0, second->1
            cats = list(y_raw.dropna().unique())
            if len(cats) == 2:
                mapping = {cats[0]:0, cats[1]:1}
                y = y_raw.map(mapping).astype(int)
            else:
                # try numeric conversion fallback
                numeric = pd.to_numeric(y_raw, errors='coerce')
                if numeric.notna().sum() / len(numeric) > 0.5:
                    med = numeric.median()
                    y = (numeric >= med).astype(int)
                    print(f"Converted textual target to numeric and thresholded at median {med}")
                else:
                    raise ValueError("Cannot reliably map target to binary. Please preprocess target manually.")
    else:
        numeric = pd.to_numeric(y_raw, errors='coerce')
        med = numeric.median()
        y = (numeric >= med).astype(int)
        print(f"Numeric target binarized at median {med:.3f}")

    print("Target distribution:\n", y.value_counts())

    # -----------------------
    # Feature selection (memory-friendly)
    # -----------------------
    # Columns to exclude from one-hot encoding (identifiers / very high-cardinality)
    high_card_cols = ['unique_id', 'order_id', 'agent_name', 'supervisor', 'manager']

    # Date/time-like columns to parse (heuristic)
    date_cols = [c for c in df.columns if any(k in c for k in ['date','time','reported','responded'])]
    # Candidate text columns
    text_candidates = [c for c in df.columns if any(k in c for k in ['message','remark','comment','text','body','customer_remarks'])]
    if len(text_candidates) == 0 and 'customer_remarks' in df.columns:
        text_candidates = ['customer_remarks']

    # Candidate categorical columns (object dtype), excluding text and target
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in text_candidates and c != target_col]

    # Exclude high-card columns and date cols from categorical candidates
    cat_cols = [c for c in cat_cols if c not in high_card_cols and c not in date_cols]

    # Numeric columns
    num_cols = [c for c in df.columns if df[c].dtype in ['int64','float64'] and c != target_col]

    print("Before datetime parsing: text:", text_candidates, "categorical candidates:", cat_cols, "numeric:", num_cols, "date cols:", date_cols)

    # ---------- Date parsing -> numeric features ----------
    for dc in date_cols:
        try:
            tmp = pd.to_datetime(df[dc], errors='coerce')
            df[f"{dc}_hour"] = tmp.dt.hour.fillna(-1).astype(int)
            df[f"{dc}_dow"] = tmp.dt.dayofweek.fillna(-1).astype(int)
            num_cols.extend([f"{dc}_hour", f"{dc}_dow"])
        except Exception:
            pass

    # Remove duplicates in num_cols
    num_cols = list(dict.fromkeys(num_cols))

    # ---------- Drop/ignore identifier columns ----------
    drop_cols = [c for c in high_card_cols if c in df.columns]
    if drop_cols:
        print("Will drop high-cardinality identifier columns (not used for one-hot):", drop_cols)
    else:
        print("No explicit high-cardinality id columns found to drop.")

    # ---------- Text feature ----------
    if len(text_candidates) > 0:
        df['_joined_text_'] = df[text_candidates].fillna("").astype(str).agg(' '.join, axis=1)
        text_feature = '_joined_text_'
        tfidf_max_features = 1000
        # add a text length numeric feature
        df[f"{text_feature}_len"] = df[text_feature].str.len().fillna(0).astype(int)
        num_cols.append(f"{text_feature}_len")
    else:
        text_feature = None
        tfidf_max_features = 0

    # ---------- Heuristic: keep only categorical cols with limited unique values ----------
    safe_cat_cols = []
    dropped_cat_cols = []
    for c in cat_cols:
        try:
            nunique = df[c].nunique(dropna=True)
            if nunique <= 100:
                safe_cat_cols.append(c)
            else:
                dropped_cat_cols.append((c, nunique))
        except Exception:
            dropped_cat_cols.append((c, 'error'))

    if dropped_cat_cols:
        print("Dropped categorical columns due to high cardinality (name, nunique):", dropped_cat_cols)

    cat_cols = safe_cat_cols

    print("Final selected columns -> text:", text_feature, "categorical (one-hot):", cat_cols, "numeric:", num_cols)

    # Build transformers
    transformers = []

    if text_feature is not None:
        # TfidfVectorizer acts on the single text column
        transformers.append((
            "tfidf",
            TfidfVectorizer(max_features=tfidf_max_features, stop_words='english'),
            text_feature
        ))

    if len(cat_cols) > 0:
        # Use sparse_output=True to keep one-hot as sparse
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    if len(num_cols) > 0:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))

    # ColumnTransformer -- set sparse_threshold so result can stay sparse when possible
    ct = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0.3)

    # -----------------------
    # Prepare training data (drop target and non-feature drop_cols)
    # -----------------------
    features_to_drop = [target_col] + drop_cols
    X = df.drop(columns=[c for c in features_to_drop if c in df.columns])
    # Ensure target index aligns
    y = y.loc[X.index]

    # Train/test split (stratify)
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print("Train size:", X_train_df.shape, "Test size:", X_test_df.shape)

    # -----------------------
    # Pipelines: LogisticRegression (sparse-friendly) and RandomForest (dense)
    # -----------------------
    # Logistic Regression pipeline (works with sparse matrices)
    pipe_logreg = Pipeline([
        ('pre', ct),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))
    ])

    # RandomForest pipeline: convert to dense before the RF classifier (only if memory allows)
    # We convert sparse -> dense using FunctionTransformer; if it's already dense, this is a no-op.
    def to_dense_if_sparse(X_in):
        if hasattr(X_in, "toarray"):
            return X_in.toarray()
        return X_in

    pipe_rf = Pipeline([
        ('pre', ct),
        ('todense', FunctionTransformer(to_dense_if_sparse, accept_sparse=True)),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced'))
    ])

    # -----------------------
    # Fit and evaluate Logistic Regression
    # -----------------------
    print("\nTraining Logistic Regression baseline...")
    pipe_logreg.fit(X_train_df, y_train)
    pred_lr = pipe_logreg.predict(X_test_df)
    prob_lr = pipe_logreg.predict_proba(X_test_df)[:, 1]
    print("Logistic Regression classification report:")
    print(classification_report(y_test, pred_lr))
    try:
        print("ROC-AUC LR:", roc_auc_score(y_test, prob_lr))
    except Exception:
        pass

    # -----------------------
    # Fit and evaluate Random Forest
    # -----------------------
    print("\nTraining Random Forest (may take longer)...")
    pipe_rf.fit(X_train_df, y_train)
    pred_rf = pipe_rf.predict(X_test_df)
    # convert to probability if possible
    try:
        prob_rf = pipe_rf.predict_proba(X_test_df)[:, 1]
    except Exception:
        prob_rf = None
    print("Random Forest classification report:")
    print(classification_report(y_test, pred_rf))
    if prob_rf is not None:
        try:
            print("ROC-AUC RF:", roc_auc_score(y_test, prob_rf))
        except Exception:
            pass

    # -----------------------
    # Choose best by macro F1 and save
    # -----------------------
    f1_lr = f1_score(y_test, pred_lr, average='macro')
    f1_rf = f1_score(y_test, pred_rf, average='macro')
    print(f"\nF1 scores -> LR: {f1_lr:.4f}, RF: {f1_rf:.4f}")

    best_pipe = pipe_rf if f1_rf >= f1_lr else pipe_logreg
    best_name = "random_forest" if best_pipe is pipe_rf else "logistic_regression"
    save_path = MODEL_DIR / f"{best_name}.joblib"
    joblib.dump(best_pipe, save_path)
    print("Saved best model to:", save_path)

    eval_summary = {
        "model": best_name,
        "f1_lr": float(f1_lr),
        "f1_rf": float(f1_rf),
        "test_size": int(len(y_test))
    }
    joblib.dump(eval_summary, MODEL_DIR / "eval_summary.joblib")
    print("Saved evaluation summary to models/eval_summary.joblib")

    # Print confusion matrix of best model
    best_pred = pred_rf if best_pipe is pipe_rf else pred_lr
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, best_pred))

if __name__ == "__main__":
    main()
