# pages/2_Prediction.py
import streamlit as st
import pandas as pd

st.title("Crop Yield Prediction")

if st.session_state.model is None:
    st.warning("Please train models in the Model Training page first!")
else:
    # Get user inputs
    area = st.selectbox("Area", list(st.session_state.label_encoders['Area'].classes_))
    item = st.selectbox("Item", list(st.session_state.label_encoders['Item'].classes_))
    
    # Extended year selection up to 2050
    current_year = pd.Timestamp.now().year
    year = st.slider("Year", 
                    int(st.session_state.df['Year'].min()), 
                    2050,
                    step=1)
    
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
    pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, step=0.1)
    avg_temp = st.number_input("Average Temperature", min_value=0.0, step=0.1)

    # Prepare input data
    input_data = pd.DataFrame({
        'Area': [st.session_state.label_encoders['Area'].transform([area])[0]],
        'Item': [st.session_state.label_encoders['Item'].transform([item])[0]],
        'Year': [year],
        'average_rain_fall_mm_per_year': [rainfall],
        'pesticides_tonnes': [pesticides],
        'avg_temp': [avg_temp]
    })

    # Make prediction
    if st.button("Predict"):
        prediction = st.session_state.model.predict(input_data)[0]
        st.write(f"### Predicted Crop Yield: {prediction:.2f} hg/ha")