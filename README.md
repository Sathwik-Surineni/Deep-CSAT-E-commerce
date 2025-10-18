<div align="center">

# 🧠 Deep CSAT – E-commerce Customer Support Analytics

🚀 **Predicting Customer Satisfaction (CSAT) Using Machine Learning & NLP**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/WebApp-Streamlit-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📘 Overview

**Deep CSAT (Customer Satisfaction Analysis Tool)** is an end-to-end **machine learning project** designed to predict customer satisfaction scores from **e-commerce support data**.

The project analyzes how factors like **query category**, **response time**, **agent behavior**, and **customer feedback text** influence customer happiness (CSAT).  
It also includes an **interactive Streamlit dashboard** to explore and predict satisfaction levels in real-time.

---

## 🎯 Objective

- 📊 Perform deep **Exploratory Data Analysis (EDA)** on customer service data  
- 🤖 Build **predictive models** for customer satisfaction (binary CSAT classification)  
- 🔍 Identify **key drivers of satisfaction** (categories, shifts, product types, etc.)  
- 🌐 Deploy a **Streamlit web app** for interactive predictions  

---

## ⚙️ Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| **Language** | Python 3.11 |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Web App** | Streamlit |
| **Model Saving** | Joblib |
| **IDE** | Visual Studio Code |

---

## 🧩 Project Structure

~~~
Deep-CSAT-E-commerce/
│
├── data/
│ └── eCommerce_Customer_support_data.csv # Dataset (excluded from GitHub)
│
├── models/
│ ├── logistic_regression.joblib # Best performing ML model
│ └── eval_summary.joblib # Evaluation results
│
├── outputs/
│ ├── EDA plots and summary files
│
├── src/
│ ├── train.py # Data cleaning + model training
│ ├── predict_demo.py # Quick prediction testing
│ ├── streamlit_app.py # Interactive Streamlit app
│ └── eda_analysis.py # EDA script
│
├── requirements.txt # All dependencies
├── README.md # Project documentation
└── .gitignore # Ignored system & data files
~~~

---

## 🧠 ML Pipeline Overview

1. **Data Preprocessing**
   - Handle missing values, categorical encoding, and time parsing  
   - Drop high-cardinality identifiers (agent names, IDs)

2. **Feature Engineering**
   - Extracted date features (hour, day-of-week)  
   - Text length from customer remarks  
   - Derived binary CSAT (based on median satisfaction)

3. **Model Training**
   - Compared **Logistic Regression** and **Random Forest**  
   - Evaluated via **F1 Score**, **ROC-AUC**, and **Confusion Matrix**

4. **Model Selection**
   - Logistic Regression chosen for best generalization  
   - Saved using `joblib` for deployment

5. **Visualization & Insights**
   - EDA plots for channel, category, product, and sentiment behavior  
   - Correlation heatmaps and textual feedback distribution  

---

## 📊 Key Insights

- ✅ **Inbound calls** and **morning shifts** have higher CSAT scores  
- ⚡ **Faster response times** directly correlate with customer satisfaction  
- 💬 **Short, polite customer feedback** often indicates satisfaction  
- 📦 Categories like *“Product Queries”* and *“Installation/Demo”* drive high engagement  
- 🧍‍♂️ Agent performance and tenure have measurable CSAT impact  

---

## 🚀 Running the Project Locally

###  1. Clone the Repository
~~~
git clone https://github.com/Sathwik-Surineni/Deep-CSAT-E-commerce.git
cd Deep-CSAT-E-commerce
~~~
###  2. Create Virtual Environment
~~~
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # (for Windows PowerShell)

~~~
### 3. Install Dependencies
~~~
pip install -r requirements.txt
~~~
### 4. Train Model
~~~
python src/train.py
~~~
### 5. Run Streamlit App
~~~
streamlit run src/streamlit_app.py

~~~
Then open the local link shown (usually http://localhost:8501) in your browser.

## 🧩 Streamlit App Features
~~~
| Feature               | Description                              |
| --------------------- | ---------------------------------------- |
| 🧾 **Data Upload**    | Upload new support logs for analysis     |
| 🔮 **Predict CSAT**   | Get satisfaction predictions instantly   |
| 📈 **Model Insights** | See top contributing features            |
| 💬 **Text Analysis**  | View effect of remarks & feedback length |

~~~
## Model Performance Summary
~~~
|   Metric | Logistic Regression | Random Forest |
| -------: | ------------------: | ------------: |
| Accuracy |                 64% |           67% |
| F1-Score |                0.61 |          0.58 |
|  ROC-AUC |                0.69 |          0.63 |
~~~
✅ Logistic Regression chosen as best overall model (better generalization + stability).

## EDA Highlights

CSAT Distribution → Majority of customers gave scores ≥5

Category-wise Trends → “Returns” & “Cancellations” show lower CSAT

Text Length Analysis → Short, positive remarks correlate with high CSAT

Heatmaps → Channel × Category strongly impacts satisfaction

All visuals are auto-generated in /outputs/.

## Future Improvements

🔍 Add sentiment analysis using BERT or VADER

📊 Integrate agent performance dashboards

☁️ Deploy Streamlit app on Streamlit Cloud / Render / Hugging Face Spaces

🧠 Experiment with Deep Learning (RNNs) for text analysis







<div align="center">
⭐ If you found this project insightful, consider giving it a star!

💡 “Good customer experience starts with great data-driven understanding.”

</div> 
