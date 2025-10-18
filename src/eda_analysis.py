# src/eda_analysis.py
"""
Exploratory Data Analysis (EDA) for DeepCSAT – E-commerce Customer Support Data
Author: Sathwik Surineni
Purpose: To understand dataset structure, data quality, and key feature relationships before model training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------------
# Basic setup
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "eCommerce_Customer_support_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset with shape: {df.shape}")
print("\nColumns:", df.columns.tolist())

# -------------------------
# Basic info
# -------------------------
print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Basic Statistics (Numeric Columns):")
print(df.describe(include=[np.number]).T)

print("\n🔹 Sample Rows:")
print(df.head(5))

# -------------------------
# Missing Values Analysis
# -------------------------
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\n🔹 Missing Value Count:")
print(missing)

plt.figure(figsize=(10,5))
missing.plot(kind='bar', color='orange')
plt.title("Missing Values by Column")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "missing_values.png")
plt.close()

# -------------------------
# Target Column Analysis: CSAT Score
# -------------------------
target_col = 'csat_score'
if target_col in df.columns:
    print(f"\n🔹 Target Column ({target_col}) value counts:")
    print(df[target_col].value_counts(dropna=False))

    # Convert to numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df, palette='Blues')
    plt.title("CSAT Score Distribution")
    plt.xlabel("Customer Satisfaction Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "csat_distribution.png")
    plt.close()

    # Binary classification (median split)
    median_csat = df[target_col].median()
    df['csat_binary'] = (df[target_col] >= median_csat).astype(int)
    print(f"\n🔹 Median CSAT: {median_csat}")
    print(df['csat_binary'].value_counts())

    plt.figure(figsize=(5,4))
    df['csat_binary'].value_counts().plot.pie(
        autopct='%1.1f%%',
        labels=['Unsatisfied (0)', 'Satisfied (1)'],
        colors=['#f08080','#90ee90']
    )
    plt.title("Binary CSAT Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "csat_binary_pie.png")
    plt.close()

# -------------------------
# Categorical Analysis
# -------------------------
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
important_cats = ['channel_name', 'category', 'sub-category', 'product_category', 'agent_shift', 'tenure_bucket']

for col in important_cats:
    if col in df.columns:
        plt.figure(figsize=(10,4))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='viridis')
        plt.title(f"{col} Distribution")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{col}_distribution.png")
        plt.close()

        # Relationship with CSAT (if target available)
        if target_col in df.columns:
            plt.figure(figsize=(8,4))
            sns.barplot(x=target_col, y=col, data=df, estimator=lambda x: np.mean(x >= median_csat))
            plt.title(f"Avg Satisfaction by {col}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{col}_vs_csat.png")
            plt.close()

# -------------------------
# Numeric Analysis
# -------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 0:
    plt.figure(figsize=(10,6))
    sns.heatmap(df[num_cols].corr(), cmap='coolwarm', annot=True, fmt='.2f')
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()

    for nc in num_cols:
        if nc == 'csat_score': continue
        plt.figure(figsize=(6,4))
        sns.histplot(df[nc].dropna(), kde=True, color='teal')
        plt.title(f"Distribution of {nc}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{nc}_histogram.png")
        plt.close()

# -------------------------
# Text Feature Exploration
# -------------------------
if 'customer_remarks' in df.columns:
    df['text_length'] = df['customer_remarks'].fillna('').astype(str).apply(len)
    plt.figure(figsize=(8,4))
    sns.histplot(df['text_length'], bins=50, color='purple', kde=True)
    plt.title("Distribution of Customer Remarks Length")
    plt.xlabel("Text Length (characters)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_remarks_length.png")
    plt.close()

# -------------------------
# Channel vs Category vs CSAT
# -------------------------
if all(col in df.columns for col in ['channel_name','category','csat_binary']):
    pivot = df.pivot_table(index='channel_name', columns='category', values='csat_binary', aggfunc='mean')
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt=".2f")
    plt.title("Avg Satisfaction by Channel and Category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "channel_category_csat.png")
    plt.close()

# -------------------------
# Save cleaned summary
# -------------------------
summary = {
    "rows": df.shape[0],
    "columns": df.shape[1],
    "missing_values": int(missing.sum()),
    "median_csat": float(df['csat_score'].median()) if 'csat_score' in df.columns else None,
    "binary_satisfied_percent": float(df['csat_binary'].mean() * 100) if 'csat_binary' in df.columns else None
}
summary_path = OUTPUT_DIR / "eda_summary.txt"
with open(summary_path, "w") as f:
    for k,v in summary.items():
        f.write(f"{k}: {v}\n")

print("\n✅ EDA completed successfully!")
print(f"All plots saved to: {OUTPUT_DIR}")
print(f"Summary saved to: {summary_path}")
