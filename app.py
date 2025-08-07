import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load model dan statistik
model = joblib.load('stacking_model.pkl')
mean = np.load('mean.npy')
std = np.load('std.npy')

def load_data():
    df = pd.read_csv("Cancer_Data.csv")
    df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

df = load_data()

feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.title("🔬 Breast Cancer Prediction")
st.subheader("Menggunakan Stacking Ensemble Learning")

st.markdown("""
---

- **Base Models**:
    1. K-Nearest Neighbors (KNN)
    2. Logistic Regression
    3. Decision Tree

- **Meta Model**:
    - Logistic Regression

---
""")

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Breast Cancer Prediction", "Info Dataset"])

if menu == "Breast Cancer Prediction":
    st.subheader("🧪 Input Karakteristik Sel")
    st.markdown("Masukkan nilai karakteristik sel untuk memprediksi jenis kanker.")

    input_features = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", value=0.0, format="%.6f")
        input_features.append(val)

    if st.button("Predict"):
        input_array = np.array(input_features).reshape(1, -1)
        input_norm = (input_array - mean) / std

        prediction = model.predict(input_norm)
        result = "Malignant" if prediction[0] == 1 else "Benign"
        st.success(f"Prediction: {result}")

elif menu == "Info Dataset":
    st.subheader("📊 Ringkasan Dataset")
    st.write(df.head())

    st.markdown("Distribusi Diagnosis (0 = Benign, 1 = Malignant):")
    counts = df['diagnosis'].value_counts()
    st.bar_chart(counts)

    with st.expander("Lihat Statistik Deskriptif"):
        st.write(df.describe())