import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

import requests
import os



API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


st.set_page_config(page_title="Churn Prediction Dashboard", layout='wide')



st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
This dashboard allows business users to:
- Upload customer data
- Run churn predictions
- View SHAP-based explanations for each customer  
""")

# ----------- sidebar ---------------
st.sidebar.header("Input Options")

threshold = st.sidebar.slider(
    "Churn Classification Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type= ["csv"])

# ---------- Main UI ------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Predict
    st.subheader("Prediction Results")

    preds_response = requests.post(url= API_URL + "/predict_batch", json=df.to_dict(orient="records"))
    probas = np.array(preds_response.json()["churn_probabilities"])

    labels = (probas >= threshold).astype(int)

    results = df.copy()
    results['Churn_Probability'] = probas
    results['Churn_Label'] = labels

    st.dataframe(results)

    # --------- SHAP Explanation --------
    st.subheader("SHAP Explanation for Selected Record") 

    idx = st.number_input(
        "Select Row Index for SHAP Explanation",
        min_value=0,
        max_value=len(df)-1,
        value = 0,
        step = 1
    )

    shap_response = requests.post(url=API_URL+"/shap_values", json=df.to_dict(orient="records"))
    shap_values = shap_response.json()['shap_values']
    expected_value = shap_response.json()['expected_value']
    feature_names = shap_response.json()['feature_names']

    shap_values = np.array(shap_values)   # Convert back to numpy

    st.write(f"Showing SHAP force plot for row index: {idx}")

     #Detect binary classifier with class dimension
    if shap_values.ndim == 3:
        # shape = (samples, features, classes)
        shap_for_sample = shap_values[idx, :, 1]       # class 1 only
        base_value = expected_value[1]
    else:
        # Logistic Regression: shape = (samples, features)
        shap_for_sample = shap_values[idx]
        base_value = expected_value
    # Generate the force plot (do not use ax)
    shap.plots.force(
        base_value,
        shap_for_sample,
        feature_names=feature_names,
        matplotlib=True,
        show=False  # prevents plt.show() call
    )

    # Get the current figure created by SHAP
    fig = plt.gcf()  

    # Display in Streamlit
    st.pyplot(fig)

else:
    st.info("Upload a CSV to begin predictions.")
    
    

