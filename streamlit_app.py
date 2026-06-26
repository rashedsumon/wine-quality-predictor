import streamlit as st
import pandas as pd
import pickle
import os
from model import train_and_save_model

# Set up page configurations
st.set_page_config(page_title="AI Wine Sommelier", page_icon="🍷", layout="centered")

MODEL_PATH = 'wine_pipeline.pkl'

# Caching the model loader so it doesn't reload from disk on every interaction
@st.cache_resource
def load_trained_pipeline():
    if not os.path.exists(MODEL_PATH):
        
        train_and_save_model()
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

# Load the AI pipeline
pipeline = load_trained_pipeline()

# App UI Header
st.title("🍷 Wine Quality Predictor")
st.markdown("Predict how wine critics will rate a bottle based on its profile details.")

st.divider()

# Input UI Fields
st.subheader("Enter Wine Characteristics")

col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("Country of Origin", ["US", "Italy", "France", "Spain", "Portugal", "Chile", "Argentina", "Australia", "Other"])
    price = st.number_input("Price per Bottle ($)", min_value=1.0, max_value=5000.0, value=25.0, step=1.0)

with col2:
    variety = st.selectbox("Wine Variety / Grape", ["Pinot Noir", "Chardonnay", "Cabernet Sauvignon", "Red Blend", "Bordeaux-style Red Blend", "Sauvignon Blanc", "Syrah", "Riesling", "Merlot"])

description = st.text_area(
    "Wine Sommelier Description / Tasting Notes", 
    placeholder="e.g., A rich, full-bodied red with notes of dark cherry, leather, robust tannins, and a hint of spice on the finish."
)

# Inference Action
if st.button("Predict Expert Rating", type="primary"):
    if not description.strip():
        
    else:
        # Construct exact DataFrame input expected by pipeline
        input_data = pd.DataFrame([{
            'description': description,
            'country': country,
            'variety': variety,
            'price': price
        }])
        
        # Make prediction
        predicted_score = pipeline.predict(input_data)[0]
        
        # Clamp bounds to match real-life Kaggle dataset parameters (80-100 points scale)
        final_score = min(max(predicted_score, 80.0), 100.0)
        
        # Display Result Visuals
        st.success(f"### Predicted Rating: **{final_score:.1f} / 100 Points**")
        
        # Contextual messaging
        if final_score >= 95:
            st.balloons()
            st.markdown("🏆 **An absolute masterpiece.** This score reflects an iconic, benchmark wine.")
        elif final_score >= 90:
            st.markdown("✨ **Excellent.** Highly recommended for its superior quality and distinction.")
        elif final_score >= 85:
            st.markdown("👍 **Good.** A very solid, well-made everyday wine.")
        else:
            st.markdown("😐 **Mediocre.** Acceptable quality, but lacks complexity.")