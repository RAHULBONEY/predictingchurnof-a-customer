
An end-to-end machine learning project that predicts customer churn for a telecom company. This repository contains the full workflow, from data analysis and model tuning to a live, interactive Streamlit dashboard that provides predictions, financial impact analysis, and model explanations.

🚀 View the Live Deployed App Here
https://predictingchurnof-a-customer-sp4kxnzy7cxjycrrdjxhnq.streamlit.app/

📋 Project Overview
The goal of this project is to build a tool that not only predicts if a customer will churn but also why. By identifying high-risk customers, a business can proactively offer targeted incentives, reducing revenue loss.

This app goes beyond a simple prediction by answering key business questions:

Which customers are at high risk of leaving?

Why is the model flagging them as a risk? (Model Explainability)

How much money could this model save us? (Financial Impact)

Which model is best for this business problem? (Model Comparison)

✨ Key Features
This application is a 4-page Streamlit dashboard:

🚀 Churn Prediction:

Enter a customer's details (tenure, contract, etc.) to get a real-time churn prediction.

See the churn probability (e.g., "85% High Risk").

Model Explainability (SHAP): Includes a "Why?" section with a SHAP force plot, showing exactly which features (like "Contract: Month-to-month") pushed the model toward its decision.

💸 Financial Impact Simulation:

The most powerful feature of this project.

An interactive calculator that simulates the Return on Investment (ROI) of using this model.

Allows a manager to set business assumptions (e.g., "Cost of retention offer," "Average revenue per customer") and see the net profit/loss from a retention campaign.

Proves the business value of the high-recall tuned model.

📊 Model Performance:

Compares the baseline Logistic Regression model against the final, tuned XGBoost model.

Uses Classification Reports, Confusion Matrices, and ROC/AUC Curves to justify why the tuned model was chosen.

Key Insight: This page demonstrates the classic trade-off: we intentionally sacrificed overall accuracy (80% vs 73%) to dramatically increase Recall (57% vs 80%), making the model far better at its real job—finding at-risk customers.

📈 Data Exploration:

An interactive dashboard (using tabs) that shows the key insights from the original data analysis.

Includes visualizations for Contract, Tenure, and InternetService to show their strong correlation with churn.

🛠️ Tech Stack
Python 3.10+

Streamlit: For the interactive web dashboard.

Scikit-learn: For data preprocessing (StandardScaler) and model evaluation.

XGBoost: For the final, tuned classification model.

SHAP: For model explainability ("Why?" plots).

Pandas: For data manipulation and analysis.

Seaborn & Matplotlib: For data visualization.

Joblib: For saving and loading the trained model assets.

📁 Project Structure
├── .streamlit/
│   └── config.toml        # Streamlit theme file
├── data/
│   └── ...-Churn.csv      # Raw dataset
├── notebooks/
│   └── churn analysis.ipynb # Full data analysis and model tuning
├── app.py                 # The main Streamlit application
├── requirements.txt       # Python libraries for deployment
├── churn_model.pkl        # 1. Tuned XGBoost model
├── log_model.pkl          # 2. Baseline Logistic Regression model
├── scaler.pkl             # 3. Fitted StandardScaler
├── shap_explainer.pkl     # 4. SHAP explainer
├── trained_columns.json   # 5. List of model features
└── roc_curve_comparison.png # Saved plot for the dashboard
Running This Project Locally
Clone the repository:
git clone https://github.com/RAHULBONEY/predictingchurnof-a-customer.git
cd predictingchurnof-a-customer
Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
Install the required libraries:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py
