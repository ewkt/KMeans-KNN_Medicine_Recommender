import os
import streamlit as st

from src.dataprep import DataPrep
from src.kmeans import KMeansClustering
from src.knn import KnnClustering

st.title("Predict which medication to use!")

with st.spinner("Loading the models..."):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'medical_data.csv')

    preprocess = DataPrep(data_path)
    df = preprocess.main()

    kmeans = KMeansClustering(nb_clusters=15, df=df)
    kmeans.train()

    knn = KnnClustering(n_neighbors=3, df=df)
    knn.train()

st.success("Models ready to use!")

cause_options, disease_options, symptom_options = preprocess.load_options()

selected_model = st.radio("Select the model to use for prediction:", ["KMeans", "KNN"], horizontal=True)

# Dropdowns for user input
selected_symptoms = st.multiselect("Select symptoms", symptom_options, default=['abdominal_pain','anxiety'])
selected_disease = st.selectbox("Select disease", disease_options, index=0)
selected_causes = st.multiselect("Select causes", cause_options, default=['allergies','chronic_fatigue_syndrome'])

if st.button("Predict Medication"):
    try:
        if selected_model == "KMeans":
            result = kmeans.infer(selected_symptoms, [selected_disease], selected_causes)
        elif selected_model == "KNN":
            result = knn.infer(selected_symptoms, [selected_disease], selected_causes)

        st.success("Result:")
        st.write(result)
    except Exception as e:
        st.error(f"Error: {str(e)}")