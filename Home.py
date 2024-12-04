# Home.py (main page)
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
)

st.title("Crop Yield Prediction System")
st.sidebar.success("Select a page above.")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'model' not in st.session_state:
    st.session_state.model = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None

st.write("### Upload Dataset")
    
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")
reload_data = st.button("Reload Data")

if uploaded_file or reload_data:
    # Load the dataset
    st.session_state.df = pd.read_csv(uploaded_file if uploaded_file else st.session_state.df)
    if "Unnamed: 0" in st.session_state.df.columns:
        st.session_state.df.drop("Unnamed: 0", axis=1, inplace=True)

    st.write("### Dataset Preview")
    st.write(st.session_state.df.head())

    # Encode categorical columns
    for col in ['Area', 'Item']:
        if col in st.session_state.df.columns:
            le = LabelEncoder()
            st.session_state.df[col] = le.fit_transform(st.session_state.df[col])
            st.session_state.label_encoders[col] = le

    st.write("### Dataset After Encoding")
    st.write(st.session_state.df.head())
    
    # Reset models_trained flag when new data is uploaded
    st.session_state.models_trained = False