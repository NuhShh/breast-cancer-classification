Breast Cancer Classification with Stacking Ensemble and Streamlit UI
====================================================================

This project applies machine learning to classify breast cancer cases (benign vs malignant)
using a stacking ensemble method and provides an interactive web app via Streamlit.

### 🔗 Try it here: https://breast-cancer-classification-nuhshh.streamlit.app
### 📦 Source: https://github.com/NuhShh/breast-cancer-classification


📊 Dataset
----------
- Source: https://www.kaggle.com/datasets/erdemtaha/cancer-data
- Description: Breast Cancer Wisconsin (Diagnostic) dataset
- Target: `diagnosis` (M = malignant, B = benign)
- Features: Mean values of cell nuclei characteristics (radius, texture, perimeter, area, etc.)

🧠 Model Overview
-----------------
- Preprocessing:
  - Z-score normalization
  - Outlier handling
  - Class balancing with SMOTE

- Base classifiers:
  - Logistic Regression
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)

- Meta-classifier:
  - Logistic Regression (used in stacking ensemble)

- Evaluation:
  - Accuracy, Confusion Matrix, Classification Report

💻 App Features (Streamlit)
---------------------------
- Simple and responsive UI to input custom cell feature values
- Predicts diagnosis result using trained model
- Automatically applies normalization before prediction
- Instant result display with confidence

📁 Project Structure
--------------------
- Breast_Cancer_Classification.ipynb   : Notebook for preprocessing, training, evaluation
- app.py                               : Streamlit web app
- Cancer_Data.csv                      : Main dataset
- stacking_model.pkl                   : Saved trained model
- mean.npy, std.npy                    : Normalization stats for deployment

⚙️ Installation & Usage
------------------------
1. Clone this repository:
   git clone https://github.com/NuhShh/breast-cancer-classification.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit app locally:
   streamlit run app.py

🛠️ Tools & Libraries
---------------------
- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn
- joblib
- streamlit

📌 Note
-------
This project is intended for educational purposes. Model performance may vary
depending on feature engineering and parameter tuning.
