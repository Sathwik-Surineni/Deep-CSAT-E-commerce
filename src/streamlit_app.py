# src/streamlit_app.py
"""
Robust Streamlit demo for DeepCSAT (updated to avoid errors).
- Only registers joblib files that look like models (have callable .predict)
- Fills missing columns with safe defaults before calling model.predict
- Avoids parsing numeric columns as dates
- Handles both sklearn Pipeline models and plain estimators
- Graceful explainability with SHAP (optional) and coefficient / feature_importance fallbacks
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "eCommerce_Customer_support_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"

# ------------------------------
# Utility: safe model discovery
# ------------------------------
@st.cache_resource
def available_models():
    """
    Return dict model_name -> path for joblib files that are actually models (have callable predict).
    Skip evaluation summary files (eval_summary.joblib etc).
    """
    models = {}
    if not MODEL_DIR.exists():
        return models
    for p in MODEL_DIR.glob("*.joblib"):
        name = p.name
        # skip known non-model names
        if name.lower().startswith("eval") or "summary" in name.lower():
            continue
        try:
            obj = joblib.load(p)
            if hasattr(obj, "predict") and callable(getattr(obj, "predict")):
                models[name] = p
        except Exception:
            # ignore unreadable/non-model objects
            continue
    return models

@st.cache_resource
def load_model_by_path(p: Path):
    obj = joblib.load(p)
    if not (hasattr(obj, "predict") and callable(getattr(obj, "predict"))):
        raise ValueError(f"Loaded object from {p.name} is not a model (no predict).")
    return obj

# ------------------------------
# Preprocessing helpers (safe)
# ------------------------------
def ensure_columns_and_defaults(df: pd.DataFrame):
    """
    Ensure all columns that the pipeline used in training exist in df.
    Fill safe defaults when missing.
    These defaults match the expectations used in train.py earlier.
    """
    df = df.copy()
    # standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Expected columns used in training (heuristic based on train.py)
    expected_text_cols = ['customer_remarks']
    expected_cat_cols = ['channel_name', 'category', 'sub-category', 'product_category', 'tenure_bucket', 'agent_shift']
    expected_num_cols = ['item_price', 'connected_handling_time']
    date_like_cols = ['order_date_time', 'issue_reported_at', 'issue_responded', 'survey_response_date']

    # Fill missing categorical columns with sensible defaults
    for c in expected_cat_cols:
        if c not in df.columns:
            if c == 'channel_name':
                df[c] = 'Inbound'
            elif c == 'tenure_bucket':
                df[c] = 'On Job Training'
            elif c == 'agent_shift':
                df[c] = 'Morning'
            else:
                df[c] = 'missing'

    # Fill missing numeric columns with NaN (imputer in pipeline will handle)
    for c in expected_num_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Fill missing text columns
    for c in expected_text_cols:
        if c not in df.columns:
            df[c] = ""

    # Fill missing date-like columns with NaN strings (will be parsed to NaT)
    for c in date_like_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df

def create_joined_text_and_lengths(df: pd.DataFrame):
    # join available text columns (only customer_remarks currently)
    text_candidates = [c for c in df.columns if any(k in c for k in ['message','remark','comment','text','body','customer_remarks'])]
    if len(text_candidates) == 0:
        df['_joined_text_'] = ""
    else:
        df['_joined_text_'] = df[text_candidates].fillna("").astype(str).agg(' '.join, axis=1)
    df['_joined_text__len'] = df['_joined_text_'].str.len().fillna(0).astype(int)
    return df

def create_date_features_safe(df: pd.DataFrame):
    """
    Create <col>_hour and <col>_dow features for date-like columns.
    We avoid parsing numeric columns as dates (check dtype).
    Also ensure fallback derived columns exist so pipeline won't crash.
    """
    df = df.copy()
    # pick date-like columns by name but skip if dtype is numeric
    date_cols = [c for c in df.columns if any(k in c for k in ['date','reported','responded','_at','order_date_time','survey_response'])]
    derived_expected = []
    for dc in date_cols:
        derived_expected.extend([f"{dc}_hour", f"{dc}_dow"])
        try:
            if pd.api.types.is_numeric_dtype(df[dc]):
                df[f"{dc}_hour"] = -1
                df[f"{dc}_dow"] = -1
                continue
        except Exception:
            # if column doesn't exist or error, set defaults later
            df[f"{dc}_hour"] = -1
            df[f"{dc}_dow"] = -1
            continue
        # parse as datetime with dayfirst True (your data is dd/mm/yyyy)
        try:
            tmp = pd.to_datetime(df[dc], errors='coerce', dayfirst=True)
            df[f"{dc}_hour"] = tmp.dt.hour.fillna(-1).astype(int)
            df[f"{dc}_dow"] = tmp.dt.dayofweek.fillna(-1).astype(int)
        except Exception:
            df[f"{dc}_hour"] = -1
            df[f"{dc}_dow"] = -1

    # ensure common derived columns exist
    fallback_names = [
        "order_date_time_hour", "order_date_time_dow",
        "issue_reported_at_hour", "issue_reported_at_dow",
        "issue_responded_hour", "issue_responded_dow",
        "survey_response_date_hour", "survey_response_date_dow",
        "connected_handling_time_hour", "connected_handling_time_dow"
    ]
    for name in fallback_names:
        if name not in df.columns:
            df[name] = -1
    return df

def minimal_prepare(df: pd.DataFrame):
    """
    Perform minimal preprocessing to match the transformer's expectations:
    - ensure expected columns exist (with defaults)
    - create _joined_text_ and _joined_text__len
    - create date-derived features safely
    """
    df = ensure_columns_and_defaults(df)
    df = create_joined_text_and_lengths(df)
    df = create_date_features_safe(df)
    return df

# ------------------------------
# Explainability helpers
# ------------------------------
def get_feature_names_from_preprocessor(preprocessor):
    """
    Heuristic: try to call get_feature_names_out; if not available, attempt to build names.
    This may fail for complex cases — we catch exceptions and return None.
    """
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            return list(preprocessor.get_feature_names_out())
    except Exception:
        pass
    try:
        # attempt to extract transformer info (best-effort)
        names = []
        if hasattr(preprocessor, "transformers_"):
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'remainder':
                    continue
                # numeric pipeline
                if transformer is None:
                    if isinstance(cols, (list, tuple)):
                        names.extend(list(cols))
                    else:
                        names.append(cols)
                    continue
                # if onehot inside pipeline
                if hasattr(transformer, "named_steps") and 'onehot' in transformer.named_steps:
                    ohe = transformer.named_steps['onehot']
                    cats = ohe.categories_
                    if isinstance(cols, (list, tuple)):
                        for col, cat in zip(cols, cats):
                            for c in cat:
                                names.append(f"{col}__{c}")
                    else:
                        for c in cats[0]:
                            names.append(f"{cols}__{c}")
                elif hasattr(transformer, "get_feature_names_out"):
                    try:
                        out = transformer.get_feature_names_out(cols)
                        names.extend(list(out))
                    except Exception:
                        if isinstance(cols, (list, tuple)):
                            names.extend(list(cols))
                        else:
                            names.append(cols)
                else:
                    if isinstance(cols, (list, tuple)):
                        names.extend(list(cols))
                    else:
                        names.append(cols)
        return names if names else None
    except Exception:
        return None

def explain_model_basic(model, preprocessor, X_sample, top_k=20):
    """
    Return a dict with either:
    - 'tree': feature_importances_ and feature names
    - 'linear': coef array and feature names
    - or 'none' if unavailable
    """
    try:
        clf = model.named_steps['clf'] if isinstance(model, Pipeline) and 'clf' in model.named_steps else model
        pre = model.named_steps['pre'] if isinstance(model, Pipeline) and 'pre' in model.named_steps else preprocessor
    except Exception:
        clf = model
        pre = preprocessor

    feat_names = None
    try:
        feat_names = get_feature_names_from_preprocessor(pre) if pre is not None else None
    except Exception:
        feat_names = None

    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
        return {"type": "tree", "importances": fi, "feature_names": feat_names}
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_.ravel()
        return {"type": "linear", "coefs": coefs, "feature_names": feat_names}
    else:
        return {"type": "none"}

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="DeepCSAT — Demo", layout="wide")
st.title("DeepCSAT — E-commerce CSAT demo (robust)")

# Sidebar: models
st.sidebar.header("Controls")
models = available_models()
if len(models) == 0:
    st.sidebar.warning("No model pipelines found in models/. Run training to save a model (e.g., logistic_regression.joblib).")
selected_model = st.sidebar.selectbox("Select saved model", options=list(models.keys()) if models else ["(none)"])

model_obj = None
if selected_model and selected_model != "(none)":
    try:
        model_obj = load_model_by_path(models[selected_model])
        st.sidebar.success(f"Loaded {selected_model}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model {selected_model}: {e}")
else:
    st.sidebar.info("Select a model to enable predictions and explainability.")

# Load a small sample for EDA/explain (cached)
@st.cache_data
def load_dataset(nrows=None):
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, nrows=nrows)
    # standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

# Tabs for organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "EDA", "Predict", "Batch", "Explain"])

# ---------- Overview ----------
with tab1:
    st.header("Project overview")
    st.markdown("""
    This demo uses a saved sklearn pipeline to predict Customer Satisfaction (CSAT).
    The saved pipeline expects preprocessed columns: text (`customer_remarks`), some categorical fields
    (channel_name, category, sub-category, product_category, tenure_bucket, agent_shift), numeric fields,
    and some date/time derived features. The app will create safe defaults for missing columns.
    """)
    df_preview = load_dataset(nrows=5)
    if df_preview.empty:
        st.info("No dataset found in data/. Place your CSV there as eCommerce_Customer_support_data.csv")
    else:
        st.subheader("Sample data (first 5 rows)")
        st.dataframe(df_preview.head())

# ---------- EDA ----------
with tab2:
    st.header("EDA")
    df_full = load_dataset()
    if df_full.empty:
        st.info("Dataset not available.")
    else:
        st.write("Rows:", df_full.shape[0], "Columns:", df_full.shape[1])
        st.dataframe(df_full.head())
        # attempt to detect target and show class distribution
        target_candidates = [c for c in df_full.columns if any(k in c for k in ['csat','satisfaction','rating','score'])]
        if target_candidates:
            tc = target_candidates[0]
            st.markdown(f"Detected target: `{tc}`")
            y = pd.to_numeric(df_full[tc], errors='coerce')
            y_bin = (y >= y.median()).astype(int)
            st.subheader("Binarized CSAT distribution (median split)")
            st.bar_chart(y_bin.value_counts())
        # numeric hist for item_price if exists
        if 'item_price' in df_full.columns:
            fig, ax = plt.subplots()
            sns.histplot(df_full['item_price'].dropna(), bins=30, ax=ax)
            ax.set_title("Item price distribution")
            st.pyplot(fig)

# ---------- Single Predict ----------
with tab3:
    st.header("Single prediction")
    st.write("Fill in a ticket and press Predict. App will add defaults for any missing columns.")
    col1, col2 = st.columns([2,1])
    with col1:
        customer_remarks = st.text_area("Customer remarks", value="", height=150)
    with col2:
        channel_name = st.selectbox("Channel", ["Inbound","Outcall","Outbound","Chat","Email"], index=0)
        category = st.text_input("Category", value="Returns")
        sub_category = st.text_input("Sub-category", value="Reverse Pickup Enquiry")
        product_category = st.text_input("Product category", value="")
        item_price = st.number_input("Item price", min_value=0.0, value=0.0)
        tenure_bucket = st.selectbox("Agent tenure", ["On Job Training","0-30","31-60","61-90",">90"], index=0)
        agent_shift = st.selectbox("Agent shift", ["Morning","Evening","Night"], index=0)
    order_date_time = st.text_input("Order date/time (e.g. 09/08/2023 11:15)", value="")
    issue_reported_at = st.text_input("Issue reported at", value="")
    issue_responded = st.text_input("Issue responded at", value="")
    survey_response_date = st.text_input("Survey response date", value="")

    threshold = st.sidebar.slider("Decision threshold for positive (satisfied)", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict"):
        if model_obj is None:
            st.error("No model loaded. Select a model in the sidebar.")
        else:
            # Build dataframe with keys named exactly as training
            row = {
                "customer_remarks": customer_remarks,
                "channel_name": channel_name,
                "category": category,
                "sub-category": sub_category,
                "product_category": product_category,
                "order_date_time": order_date_time,
                "issue_reported_at": issue_reported_at,
                "issue_responded": issue_responded,
                "survey_response_date": survey_response_date,
                "item_price": item_price,
                "tenure_bucket": tenure_bucket,
                "agent_shift": agent_shift,
                # include connected_handling_time if your pipeline expects it (safe default)
                "connected_handling_time": np.nan
            }
            X_input = pd.DataFrame([row])
            Xp = minimal_prepare(X_input)
            try:
                preds = model_obj.predict(Xp)
                probs = model_obj.predict_proba(Xp)[:,1] if hasattr(model_obj, "predict_proba") else None
                pred_default = int(preds[0])
                pred_thresh = int((probs[0] >= threshold) if probs is not None else pred_default)
                st.metric("Predicted class (default threshold 0.5)", pred_default)
                if probs is not None:
                    st.metric("Predicted probability (positive/satisfied)", f"{probs[0]:.3f}")
                    st.write(f"Predicted class at threshold {threshold:.2f}: **{pred_thresh}**")
                st.write("Preprocessed features (first row):")
                st.dataframe(Xp.head(1).T)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------- Batch Predict ----------
with tab4:
    st.header("Batch prediction (CSV upload)")
    uploaded = st.file_uploader("Upload CSV with original columns", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            df_in.columns = df_in.columns.str.strip().str.lower().str.replace(' ','_')
            st.write("Preview of uploaded data:")
            st.dataframe(df_in.head())
            if st.button("Run batch predictions"):
                Xp = minimal_prepare(df_in)
                preds = model_obj.predict(Xp) if model_obj is not None else None
                if preds is None:
                    st.error("No model loaded")
                else:
                    df_out = df_in.copy()
                    df_out['pred_default'] = preds
                    if hasattr(model_obj, "predict_proba"):
                        df_out['pred_prob'] = model_obj.predict_proba(Xp)[:,1]
                        df_out['pred_thresholded'] = (df_out['pred_prob'] >= st.sidebar.slider("batch threshold", 0.0, 1.0, 0.5, 0.01)).astype(int)
                    st.success("Batch prediction finished")
                    st.dataframe(df_out.head(20))
                    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

# ---------- Explainability ----------
with tab5:
    st.header("Explainability & feature importance")
    if model_obj is None:
        st.info("Load a model in the sidebar to enable explainability.")
    else:
        st.write("Model selected:", selected_model)
        # if pipeline, try to access preprocessor
        preproc = None
        clf = model_obj
        if isinstance(model_obj, Pipeline):
            if 'pre' in model_obj.named_steps:
                preproc = model_obj.named_steps['pre']
            if 'clf' in model_obj.named_steps:
                clf = model_obj.named_steps['clf']

        # Quick global feature importance / coefficients
        st.subheader("Global feature importances / coefficients (top 30)")
        try:
            res = explain_model_basic(model_obj, preproc, None)
            if res['type'] == 'tree':
                imps = res['importances']
                feat_names = res.get('feature_names', None)
                top_idx = np.argsort(imps)[-30:][::-1]
                fig, ax = plt.subplots(figsize=(6,8))
                if feat_names is not None and len(feat_names) == len(imps):
                    imp_df = pd.DataFrame({"feature": np.array(feat_names)[top_idx], "importance": imps[top_idx]})
                    sns.barplot(x="importance", y="feature", data=imp_df, ax=ax)
                else:
                    sns.barplot(x=imps[top_idx], y=[str(i) for i in top_idx], ax=ax)
                st.pyplot(fig)
            elif res['type'] == 'linear':
                coefs = res['coefs']
                feat_names = res.get('feature_names', None)
                if feat_names is not None and len(feat_names) == len(coefs):
                    coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
                    coef_df['abscoef'] = coef_df['coef'].abs()
                    coef_df = coef_df.sort_values('abscoef', ascending=False).head(30)
                    st.dataframe(coef_df[['feature','coef']].set_index('feature'))
                else:
                    st.write("Linear model coefficients available but feature names mapping not found.")
            else:
                st.info("Model exposes no importances or coefficients.")
        except Exception as e:
            st.error(f"Could not compute global importances: {e}")

        # SHAP per-row explanation (if pipeline + tree model)
        st.subheader("Row-level explanation (SHAP if available)")
        df_small = load_dataset(nrows=2000)
        if df_small.empty:
            st.info("No dataset to sample from for background. Skip SHAP.")
        else:
            # prepare background and sample
            background = minimal_prepare(df_small.sample(n=min(200, len(df_small)), random_state=42))
            idx = st.number_input("Row index (0-based) to explain from background sample", min_value=0, max_value=max(0,len(background)-1), value=0)
            row = background.iloc[[idx]]
            st.write("Row preview (preprocessed):")
            st.dataframe(row.head(1).T)
            if st.button("Compute SHAP/Explanation for row"):
                with st.spinner("Computing explanation..."):
                    try:
                        # only attempt SHAP if clf is tree or model is ensemble with feature_importances_
                        if hasattr(clf, "feature_importances_"):
                            try:
                                import shap
                                explainer = shap.TreeExplainer(clf)
                                # transform inputs via preprocessor if pipeline exists
                                Xb = preproc.transform(background) if preproc is not None else background.values
                                Xr = preproc.transform(row) if preproc is not None else row.values
                                # convert sparse to dense if necessary
                                if hasattr(Xr, "toarray"):
                                    Xr = Xr.toarray()
                                if hasattr(Xb, "toarray"):
                                    Xb = Xb.toarray()
                                shap_values = explainer.shap_values(Xr)
                                st.success("SHAP computed. Showing force/widget if available.")
                                # try to show simple summary bar (shap has nice plots)
                                try:
                                    shap.initjs()
                                    fig = shap.plots.bar(shap_values, max_display=20, show=False)
                                    st.pyplot(bbox_inches='tight')
                                except Exception:
                                    st.write("SHAP values computed but plotting failed.")
                            except Exception as e:
                                st.error(f"SHAP computation failed: {e}")
                        elif hasattr(clf, "coef_"):
                            # linear explanation: show coefficients * feature values
                            feat_names = get_feature_names_from_preprocessor(preproc) if preproc is not None else None
                            coefs = clf.coef_.ravel()
                            if feat_names is not None and len(feat_names) == len(coefs):
                                Xt = preproc.transform(row)
                                if hasattr(Xt, "toarray"):
                                    Xt = Xt.toarray()
                                # compute contribution = coef * x
                                contrib = coefs * Xt.ravel()
                                contr_df = pd.DataFrame({"feature": feat_names, "contrib": contrib})
                                contr_df = contr_df.reindex(contr_df['contrib'].abs().sort_values(ascending=False).index).head(30)
                                st.dataframe(contr_df.set_index('feature'))
                            else:
                                st.info("Cannot map coefficients to feature names for linear model.")
                        else:
                            st.info("Model not supported for SHAP/row explanation in this demo.")
                    except Exception as e:
                        st.error(f"Explanation failed: {e}")

st.markdown("---")
st.caption("This app is robust to missing columns and non-model joblib files. If you retrain your model, save it to models/ and refresh this app.")
