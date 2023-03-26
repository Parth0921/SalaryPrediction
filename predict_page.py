from copyreg import pickle
import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_prediction_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States of America",
        "United Kingdom of Great Britain and Northern Ireland",
        "India",
        "Canada",
        "France",
        "Brazil",
        "Germany",
        "Spain",
        "Netherlands",
        "Australia",
        "Italy",
        "Poland",
        "Sweden",
        "Russian Federation",
        "Switzerland",
    )
    
    Education = (
        "Master’s degree", 
        "Bachelor’s degree", 
        "Less than a Bachelors",
        "Post grad"
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", Education)
    experience = st.slider("Years of experience", 0, 50, 3)

    ok = st.button("Calculate Salary") #If button pressed ok is set to true otherwise false

    if ok:
        X = np.array([[country, education, experience ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)


        salary = regressor_loaded.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
