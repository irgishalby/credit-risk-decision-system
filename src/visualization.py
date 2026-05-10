import shap
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plot_shap_explanation(explainer, scaled_data, feature_names):
    # 1. Calculate SHAP values
    shap_values = explainer.shap_values(scaled_data)
    
    # 2. Create the figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 3. Use the standard summary_plot with plot_type='bar' for single instances
    # We pass the scaled data and feature names to make it readable
    shap.summary_plot(
        shap_values, 
        scaled_data, 
        feature_names=feature_names, 
        plot_type="bar", 
        max_display=5, 
        show=False # Crucial: prevents plt.show() from blocking Streamlit
    )
    
    plt.title("Top 5 Impact Factors for this Decision")
    plt.tight_layout()
    
    # 4. Render in Streamlit
    st.pyplot(fig)