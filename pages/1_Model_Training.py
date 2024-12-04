# pages/1_Model_Training.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

st.title("Model Training and Evaluation")

# Define models function to keep them in scope
def get_models():
    return [
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor(random_state=42)),
        ('Gradient Boost', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('XGBoost', XGBRegressor(random_state=42)),
        ('KNN', KNeighborsRegressor(n_neighbors=5)),
        ('Decision Tree', DecisionTreeRegressor(random_state=42)),
        ('Bagging Regressor', BaggingRegressor(n_estimators=150, random_state=42))
    ]

if st.session_state.df is None:
    st.warning("Please upload a dataset in the Home page first!")
else:
    # Add retrain button
    retrain = st.button("Retrain Models")
    
    if retrain:
        st.session_state.models_trained = False
    
    if not st.session_state.models_trained:
        df = st.session_state.df
        
        # Define features and target
        X = df.drop(labels='hg/ha_yield', axis=1)
        y = df['hg/ha_yield']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        st.session_state.training_data = (X_train, X_test, y_train, y_test)

        # Get models
        models = get_models()

        # Train and evaluate models
        results = []
        model_dict = {}  # Store trained models
        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append((name, accuracy, mse, r2))
            model_dict[name] = model  # Save trained model

        # Save results
        st.session_state.results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'MSE', 'R2_score'])
        st.session_state.model_dict = model_dict  # Save model dictionary
        
        # Save best model
        best_model_name = st.session_state.results_df.loc[st.session_state.results_df['Accuracy'].idxmax(), 'Model']
        st.session_state.model = model_dict[best_model_name]
        
        st.session_state.models_trained = True

    # Display results using Plotly
    st.write("### Model Performance")
    results_df = st.session_state.results_df.copy()
    
    # Highlight best and worst models
    best_idx = results_df['Accuracy'].idxmax()
    worst_idx = results_df['Accuracy'].idxmin()
    
    # Create Plotly table with better colors
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(results_df.columns),
            fill_color='#0E1117',  # Dark background
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[results_df[col] for col in results_df.columns],
            fill_color=[[
                '#90EE90' if i == best_idx else  # Light green
                '#FFB6C1' if i == worst_idx else  # Light pink
                '#1F2937' for i in range(len(results_df))  # Dark blue-grey
            ]],
            font=dict(color='white'),
            align='left',
            height=30
        )
    )])
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        height=250
    )
    st.plotly_chart(fig)
    
    # Plot model comparison using Plotly
    fig = px.bar(results_df, x='Model', y='Accuracy',
                 title='Model Accuracy Comparison')
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font_color='white'
    )
    st.plotly_chart(fig)
    
    # Plot actual vs predicted for each model using Plotly
    X_train, X_test, y_train, y_test = st.session_state.training_data
    for name in results_df['Model']:
        model = st.session_state.model_dict[name]  # Use stored model
        y_pred = model.predict(X_test)
        
        fig = px.scatter(x=y_test, y=y_pred, 
                        title=f'{name} - Actual vs Predicted',
                        labels={'x': 'Actual Values', 'y': 'Predicted Values'})
        
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font_color='white'
        )
        st.plotly_chart(fig)