import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# Load the model and scaler
model = joblib.load('./model/logistic_regression_model.pkl')
scaler = joblib.load('./model/scaler.pkl')

# Load Iris dataset for reference
iris = load_iris()

# Streamlit app
st.title("Iris Flower Species Prediction")

st.image('./asset/iris_petal-sepal.png', caption='Sepal length and width')

st.write("This app predicts the species of Iris flower based on its characteristics.")

# Input features
sepal_length = st.slider(
    "Sepal Length", 
    float(iris.data[:, 0].min()), 
    float(iris.data[:, 0].max()), 
    float(iris.data[:, 0].mean())
    )

sepal_width = st.slider(
    "Sepal Width", 
    float(iris.data[:, 1].min()), 
    float(iris.data[:, 1].max()), 
    float(iris.data[:, 1].mean())
    )

petal_length = st.slider(
    "Petal Length", 
    float(iris.data[:, 2].min()), 
    float(iris.data[:, 2].max()), 
    float(iris.data[:, 2].mean())
    )

petal_width = st.slider(
    "Petal Width", 
    float(iris.data[:, 3].min()), 
    float(iris.data[:, 3].max()), 
    float(iris.data[:, 3].mean())
    )

# Predict
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features = scaler.transform(features)
    prediction = model.predict(features)
    st.write(f"The predicted species is: {iris.target_names[prediction][0]}")
