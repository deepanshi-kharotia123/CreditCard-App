import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier

# Load the trained model 
RFmodel = joblib.load('fraud_detection_model.pkl')

# Dummy Model for testing 
# RFmodel = RandomForestClassifier()
# RFmodel.fit(np.random.randn(100, 30), np.random.randint(0, 2, 100))

# Load the dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('creditCard.csv')

# Calculate fraud and non-fraud statistics
total_transactions = df.shape[0]
fraud_transactions = df[df['Class'] == 1].shape[0]
non_fraud_transactions = df[df['Class'] == 0].shape[0]
fraud_rate = (fraud_transactions / total_transactions) * 100

# Streamlit App Layout
st.set_page_config(page_title="Credit Card Fraud Detection", layout='wide')
st.title("💳 Credit Card Fraud Detection")

# Dataset Overview
st.sidebar.header("Dataset Overview")
st.sidebar.write(f"**Total Transactions:** {total_transactions}")
st.sidebar.write(f"**Fraudulent Transactions:** {fraud_transactions}")
st.sidebar.write(f"**Non-Fraudulent Transactions:** {non_fraud_transactions}")
st.sidebar.write(f"**Fraud Rate:** {fraud_rate:.2f}%")

# Display metrics
st.markdown("### 📊 Dataset Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", total_transactions)
col2.metric("Fraud Transactions", fraud_transactions, delta=f"{fraud_rate:.2f}% Fraud Rate")
col3.metric("Non-Fraud Transactions", non_fraud_transactions)

# Visualization: Fraud vs Non-Fraud Transactions
st.markdown("### 📈 Transaction Analysis")
fraud_count = df['Class'].value_counts()

# Bar plot using Plotly for interactive visualization
fig = px.bar(
    x=['Non-Fraud', 'Fraud'],
    y=fraud_count.values,
    color=['Non-Fraud', 'Fraud'],
    color_discrete_map={'Non-Fraud': 'green', 'Fraud': 'red'},
    labels={'x': 'Transaction Type', 'y': 'Count'},
    title='Total Transactions (Fraud vs Non-Fraud)',
    text=fraud_count.values
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig)

# Transaction Input Section
st.markdown("### 📝 Enter the Transaction Details Below:")

# Input fields for transaction features
with st.form("transaction_form"):
    time = st.number_input('Time', format="%.3f")
    features = [st.number_input(f'V{i}', format="%.3f") for i in range(1, 29)]
    amount_log = st.number_input('Amount Log', format="%.3f")
    
    # Prepare input for prediction
    input_features = np.array([[time] + features + [amount_log]])

    # Prediction button
    predict_button = st.form_submit_button("Predict")

# Perform prediction
if predict_button:
    prediction = RFmodel.predict(input_features)
    prediction_prob = RFmodel.predict_proba(input_features)
    fraud_prob = round(prediction_prob[0][1] * 100, 3)

    # Display the prediction result with enhanced UI
    st.markdown("### 🔍 Prediction Result")
    if fraud_prob > 50:
        st.error(f"⚠️ The transaction is predicted to be FRAUDULENT")
    else:
        st.success(f"✅ The transaction is predicted to be NOT FRAUDULENT")

    # Show probability distribution
    st.write(f"Probability of Fraudulent: {fraud_prob:.3f}%")
    st.write(f"Probability of Not Fraudulent: {round(100 - fraud_prob, 3)}%")

    # Probability Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_prob,
        title={'text': "Fraud Probability"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red" if fraud_prob > 50 else "green"}}
    ))
    st.plotly_chart(fig_gauge)
