import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap 
import numpy as np


st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide"
)


@st.cache_resource
def load_assets():
    """Loads all models, scalers, and metadata."""
    try:
        model_xgb = joblib.load('churn_model.pkl')
        model_log = joblib.load('log_model.pkl')
        scaler = joblib.load('scaler.pkl')
        explainer = joblib.load('shap_explainer.pkl')
        with open('trained_columns.json', 'r') as f:
            trained_columns = json.load(f)
    except FileNotFoundError:
        st.error("Error: Asset files not found. Please run your notebook to save all 5 assets first.")
        st.stop()
    return model_xgb, model_log, scaler, explainer, trained_columns


model_xgb, model_log, scaler, explainer, TRAINED_COLUMNS = load_assets()


@st.cache_data
def load_raw_data():
    """Loads the raw CSV data for exploration."""
    try:
        df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: Raw data file 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv' not found.")
        st.stop()

df_raw = load_raw_data()
COLS_TO_SCALE = ['tenure', 'MonthlyCharges', 'TotalCharges']



st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "🚀 Churn Prediction", 
    "💸 Financial Impact",
    "📊 Model Performance",
    "📈 Data Exploration"
])


if page == "🚀 Churn Prediction":
    st.title("Customer Churn Prediction Tool")
    st.write("""
    Enter your customer's details to predict their churn likelihood and 
    understand **why** the model made its decision.
    """)
    
    st.header("Enter Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=1)
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.01)
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    
    with col3:
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=monthly_charges * tenure, step=0.01, help="Auto-calculated.")
        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])

    if st.button("Predict Churn", type="primary", use_container_width=True):
        
        data = {
            'Contract': contract, 'tenure': tenure, 'InternetService': internet_service,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges, 'TechSupport': tech_support,
            'OnlineSecurity': online_security, 'PaymentMethod': payment_method, 'PaperlessBilling': paperless_billing,
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No'
        }
        
        try:
            input_df = pd.DataFrame([data])
            input_df_processed = pd.get_dummies(input_df)
            input_df_aligned = input_df_processed.reindex(columns=TRAINED_COLUMNS, fill_value=0)
            input_df_aligned[COLS_TO_SCALE] = scaler.transform(input_df_aligned[COLS_TO_SCALE])
            
            prediction = model_xgb.predict(input_df_aligned)[0]
            probability = model_xgb.predict_proba(input_df_aligned)[0][1]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("High Risk of Churn")
            else:
                st.success("Low Risk of Churn")

            st.metric("Churn Probability", f"{probability*100:.2f}%")
            st.progress(float(probability))

            with st.expander("See *why* the model made this prediction"):
                st.write("""
                This chart shows which customer features pushed the model's
                prediction towards 'High Risk' (red arrows) or 'Low Risk' (blue arrows).
                """)
                shap_values = explainer.shap_values(input_df_aligned)
                fig, ax = plt.subplots()
                shap.force_plot(
                    explainer.expected_value, 
                    shap_values[0], 
                    input_df_aligned.iloc[0], 
                    matplotlib=True,
                    show=False,
                    text_rotation=15
                )
                st.pyplot(fig, bbox_inches='tight', clear_figure=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")


elif page == "💸 Financial Impact":
    st.title("Financial Impact Simulation")
    st.write("""
    This page shows how much money your model could save. It uses the
    model's actual performance on the 1,407 test customers.
    """)

  
    TN = 729
    FP = 304
    FN = 75
    TP = 299
    
    actual_churners = TP + FN 
    actual_loyal = TN + FP   

    st.header("Business Assumptions (Adjust the sliders)")
    
    col1, col2 = st.columns(2)
    with col1:
        avg_revenue = st.slider(
            "Average Monthly Revenue per Customer ($)", 
            min_value=50.0, max_value=200.0, value=100.0, step=1.0
        )
        cost_of_offer = st.slider(
            "Cost of Retention Offer ($)", 
            min_value=5.0, max_value=50.0, value=25.0, step=0.50
        )
    with col2:
        offer_success_rate = st.slider(
            "Offer Success Rate (%)", 
            min_value=10.0, max_value=50.0, value=30.0, step=1.0
        ) / 100.0
    
    st.write("---")
    st.header("Simulation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scenario 1: Do Nothing (No Model)")
        revenue_lost = actual_churners * avg_revenue
        st.metric("Total Customers who Churned", actual_churners)
        st.metric("Total Revenue Lost", f"-${revenue_lost:,.0f}")
        st.metric("Net Financial Impact", f"-${revenue_lost:,.0f}", delta_color="inverse")
    
    with col2:
        st.subheader("Scenario 2: Use Your Tuned Model")
        
        total_offers_sent = TP + FP
        total_cost_of_offers = total_offers_sent * cost_of_offer
        
        churners_saved = TP * offer_success_rate
        revenue_saved = churners_saved * avg_revenue
        
        churners_lost_with_model = FN + (TP * (1 - offer_success_rate))
        revenue_lost_with_model = churners_lost_with_model * avg_revenue
        
        net_financial_impact = revenue_saved - total_cost_of_offers - revenue_lost_with_model
        
        st.metric("Total Offers Sent (to TP and FP)", total_offers_sent)
        st.metric("Total Cost of Offers", f"-${total_cost_of_offers:,.0f}")
        st.metric("Customers Saved", f"{churners_saved:,.0f}")
        st.metric("Revenue Saved", f"+${revenue_saved:,.0f}")
        st.metric("Remaining Revenue Lost", f"-${revenue_lost_with_model:,.0f}")
        
        st.metric("Net Financial Impact", f"${net_financial_impact:,.0f}", 
                  delta=f"${net_financial_impact - (-revenue_lost):,.0f} vs. Doing Nothing")

    st.subheader("Return on Investment (ROI)")
   
    if total_cost_of_offers > 0:
        roi = (net_financial_impact - (-revenue_lost)) / total_cost_of_offers
        value_created = net_financial_impact - (-revenue_lost)
        
        st.metric("Total Value Created by Model", f"${value_created:,.0f}")
        st.metric("Return on Investment (ROI) on Offers Sent", f"{roi:.2%}")
    else:
        st.metric("Return on Investment (ROI) on Offers Sent", "N/A (Cost of Offer is $0)")



elif page == "📊 Model Performance":
    st.title("Model Performance Comparison")
    st.write("""
    The tuned XGBoost model was chosen because it is **80% effective at finding churners (High Recall)**,
    which is the most important business goal. This is a classic trade-off:
    we sacrificed overall accuracy to get a model that is much better at its job.
    """)

    tab1, tab2, tab3 = st.tabs(["📈 Classification Reports", "🔲 Confusion Matrices", "🎯 ROC/AUC Curves"])

    with tab1:
        st.header("Classification Reports")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Baseline: Logistic Regression")
           
            st.text("""
                  precision    recall  f1-score   support
               0       0.85      0.89      0.87      1033
               1       0.65      0.57      0.61       374
            """)
            st.metric("Overall Accuracy", "80%")
            st.metric("Churn F1-Score", "0.61")
            st.metric("Churn Recall", "0.57")
        with col2:
            st.subheader("Final Model: Tuned XGBoost")
           
            st.text("""
                  precision    recall  f1-score   support
               0       0.91      0.71      0.79      1033
               1       0.50      0.80      0.61       374
            """)
            st.metric("Overall Accuracy", "73%")
            st.metric("Churn F1-Score", "0.61")
            st.metric("Churn Recall", "0.80", delta="23% improvement")
            
    with tab2:
        st.header("Confusion Matrices")
        st.write("The Tuned XGBoost model finds 299 churners (TP) vs. the baseline's ~213.")
        
        
        cm_log = [[919, 114], [160, 214]] 
        cm_xgb = [[729, 304], [75, 299]] 
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
            ax.set_title('Logistic Regression')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig, clear_figure=True)
            
        with col2:
            fig, ax = plt.subplots()
            sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
            ax.set_title('Tuned XGBoost (Final Model)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig, clear_figure=True)
            
    with tab3:
        st.header("ROC/AUC Curves")
        st.write("""
        Both models have a similar AUC score (~0.82), meaning they are equally good
        at distinguishing between classes. Our tuning didn't change this, but it
        **changed the decision threshold** to focus on finding more churners.
        """)
        
        
        try:
            st.image('roc_curve_comparison.png')
        except FileNotFoundError:
            st.error("roc_curve_comparison.png not found. Please run the notebook cell to save the plot.")



elif page == "📈 Data Exploration":
    st.title("Data Exploration")
    st.write("Key insights from the notebook that inspired the model.")
    
    tab1, tab2, tab3 = st.tabs(["Contract", "Tenure", "Internet Service"])
    
    with tab1:
        st.subheader("Churn by Contract")
        fig, ax = plt.subplots()
        sns.countplot(x='Contract', hue='Churn', data=df_raw, ax=ax, palette="pastel")
        st.pyplot(fig, clear_figure=True)
        st.write("**Insight:** Month-to-Month contracts are the #1 driver of churn.")
        
    with tab2:
        st.subheader("Churn by Tenure")
        fig, ax = plt.subplots()
        
        sns.histplot(data=df_raw, x='tenure', hue='Churn', multiple='stack', kde=True, ax=ax)
        st.pyplot(fig, clear_figure=True)
        st.write("**Insight:** New customers (low tenure) are at the highest risk.")
        
    with tab3:
        st.subheader("Churn by Internet Service")
        fig, ax = plt.subplots()
        sns.countplot(x='InternetService', hue='Churn', data=df_raw, ax=ax, palette="viridis")
        st.pyplot(fig, clear_figure=True)
        st.write("**Insight:** Customers with Fiber Optic have a higher churn rate.")

    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.head())