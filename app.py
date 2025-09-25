# app.py

import streamlit as st
import joblib
import pandas as pd
import datetime
import numpy as np
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Delhi Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background: #0F2027;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem;
        border: 1px solid rgba(255,255,255,0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        width: 100%;
        box-sizing: border-box;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        letter-spacing: -1px;
        line-height: 1.2;
        margin: 0.5rem 0;
        min-height: 60px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        line-height: 1.2;
        height: 20%;
    }
    
    .metric-description {
        font-size: 0.8rem;
        opacity: 0.8;
        line-height: 1.3;
        height: 30%;
    }

    /* Equal column widths */
    [data-testid="column"] {
        min-width: 0;
        flex: 1 1 0;
        width: 100%;
    }
    
    /* Responsive grid */
    @media (max-width: 1200px) {
        [data-testid="column"] {
            min-width: 200px !important;
        }
    }
    
    .stButton>button {
        background: #2CA58D;
        color: white;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background: #218f79;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(44,165,141,0.3);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(195deg, #0F2027 0%, #203A43 100%);
    }
    
    .status-good {
        background: rgba(44,165,141,0.15);
        border: 1px solid #2CA58D;
    }
    
    .status-bad {
        background: rgba(242,108,79,0.15);
        border: 1px solid #F26C4F;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #2CA58D 0%, #4B8BBE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model Data
try:
    model = joblib.load("energy_forecast_model.pkl")
    features = joblib.load("model_features.pkl")
    metrics = joblib.load("model_metrics.pkl")
    current_demand = joblib.load("last_demand.pkl")
except FileNotFoundError:
    st.error("Model files missing! Please train model first.")
    st.stop()

# Sidebar - About Section
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 2rem 1rem; text-align: center;">
        <h1 class="gradient-text" style="font-size: 2rem;">Grid Intelligence</h1>
        <div style="margin: 2rem 0;">
            <p style="color: #4B8BBE; font-size: 1.1rem;">Developed by</p>
            <p style="font-size: 1.2rem;">Vikram & Ankit</p>
            <div style="margin: 1.5rem 0;">
                <div style="font-size: 0.9rem; color: #2CA58D;">Version 3.2</div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Advanced Grid Analytics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Content
st.title("⚡ Delhi Grid Stabilization Forecast")
st.markdown("### AI-Powered Energy Demand Prediction System")

# Performance Metrics Dashboard
with st.container():
    cols = st.columns(5)
    metrics_data = [
        ("R² Score", f"{metrics['r2']:.3f}", "#2CA58D", "Variance explained"),
        ("RMSE", f"{metrics['rmse']:.1f} kWh", "#4B8BBE", "Root Mean Squared Error"),
        ("MAE", f"{metrics['mae']:.1f} kWh", "#6C5B7B", "Mean Absolute Error"),
        ("Training", f"{metrics['train_time']:.1f}s", "#F26C4F", "Optimization time"),
        ("Data", f"{metrics['data_points']//1000}k+", "#9B59B6", "Training samples")
    ]
    
    for col, (label, value, color, desc) in zip(cols, metrics_data):
        with col:
            st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color: {color};">{value}</div>
                    <div class="metric-description">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

# Prediction Interface
with st.container():
    st.header("Forecast Parameters")
    inputs = {}
    
    with st.expander("⏰ Time Configuration", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            date = st.date_input("Select Date", datetime.date.today())
            time = st.time_input("Select Time", datetime.time(12, 0))
            
        with cols[1]:
            timestamp = pd.to_datetime(f"{date} {time}")
            inputs.update({
                "hour": timestamp.hour,
                "dayofweek": timestamp.dayofweek,
                "month": timestamp.month,
                "Weekday": 1 if timestamp.dayofweek < 5 else 0,
                "Holiday_Indicator": st.checkbox("Public Holiday", False)
            })

    with st.expander("🌤️ Weather Conditions"):
        cols = st.columns(2)
        with cols[0]:
            inputs.update({
                "Temperature_C": st.slider("Temperature (°C)", -5, 45, 25),
                "Humidity_%": st.slider("Humidity (%)", 0, 100, 60)
            })
        with cols[1]:
            inputs.update({
                "Wind_Speed_mps": st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0),
                "Solar_Radiation_Wm2": st.slider("Solar Radiation (W/m²)", 0, 1000, 500)
            })

    with st.expander("🏭 System Parameters"):
        cols = st.columns(3)
        with cols[0]:
            inputs.update({
                "Industrial_Usage_kWh": st.slider("Industrial Usage (kWh)", 0, 2000, 800),
                "Residential_Usage_kWh": st.slider("Residential Usage (kWh)", 0, 1500, 500)
            })
        with cols[1]:
            inputs.update({
                "Commercial_Usage_kWh": st.slider("Commercial Usage (kWh)", 0, 1000, 300),
                "Voltage_Level_V": st.slider("Voltage Level (V)", 220, 440, 240)
            })
        with cols[2]:
            inputs.update({
                "Grid_Frequency_Hz": st.slider("Grid Frequency (Hz)", 49.0, 51.0, 50.0),
                "Renewable_Energy_Contribution_%": st.slider("Renewables Contribution (%)", 0, 100, 20)
            })

    # Validation check
    missing_features = [f for f in features if f not in inputs]
    if missing_features:
        st.error(f"Missing required inputs: {', '.join(missing_features)}")
        st.stop()

    # Prediction Execution
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Analyzing grid patterns..."):
            input_df = pd.DataFrame([inputs])[features]
            prediction = model.predict(input_df)[0]
            
            st.success("Forecast Generated!")
            cols = st.columns(2)
            
            with cols[0]:
                # Demand Comparison Chart
                fig = px.bar(
                    x=["Current Demand", "Predicted Demand"],
                    y=[current_demand, prediction],
                    color=["Current", "Predicted"],
                    color_discrete_map={"Current": "#4B8BBE", "Predicted": "#2CA58D"},
                    labels={"y": "Energy (kWh)", "x": ""},
                    title="Demand Comparison"
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with cols[1]:
                # Status Indicator
                status = "good" if prediction < 1000 else "bad"
                status_color = "#2CA58D" if status == "good" else "#F26C4F"
                
                st.markdown(f"""
                    <div class="metric-card status-{status}">
                        <div style="font-size: 1.5rem; margin-bottom: 1.5rem;">
                            {"✅ Grid Stable" if status == "good" else "⚠️ Grid Critical"}
                        </div>
                        <div style="display: flex; align-items: center; gap: 1.5rem;">
                            <div style="font-size: 2.5rem; color: {status_color};">
                                {prediction:,.0f} kWh
                            </div>
                            <div style="flex-grow: 1;">
                                <progress 
                                    value="{prediction}" 
                                    max="1000" 
                                    style="width: 100%; height: 15px; accent-color: {status_color};"
                                ></progress>
                                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                                    <small>0 kWh</small>
                                    <small>1000 kWh</small>
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 1.5rem; color: {status_color}; font-size: 1.1rem;">
                            {f"{(prediction/1000)*100:.1f}% of capacity" if status == "bad" else "Within safe limits"}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="margin-top: 3rem; text-align: center; opacity: 0.7;">
        <small>© 2024 Delhi Grid Intelligence | Version 3.2</small>
    </div>
""", unsafe_allow_html=True)