import os
import streamlit as st

from src.dataprep import DataPrep
from src.kmeans import KMeansClustering

st.title("Predict which medication to use!")

st.loader("Loading the Kmeans model...")

prerpocess = DataPrep()
df = prerpocess.main()

model = KMeansClustering(nb_clusters=15, df=df)
model.train()

st.success("Model trained successfully!")

cause_options, disease_options, symptom_options = prerpocess.load_options()

# Dropdowns for user input
selected_symptoms = st.multiselect("Select symptoms", symptom_options, default=['abdominal_pain','anxiety'])
selected_disease = st.selectbox("Select disease", disease_options, index=0)
selected_causes = st.multiselect("Select causes", cause_options, default=['allergies','chronic_fatigue_syndrome'])

if st.button("Predict Medication"):
    try:
        result = model.infer(selected_symptoms, [selected_disease], selected_causes)
        st.success("Result:")
        st.write(result)
    except Exception as e:
        st.error(f"Error: {str(e)}")