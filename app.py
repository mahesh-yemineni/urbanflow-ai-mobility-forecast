# ================================================================
# UrbanFlow-AI Dashboard (Streamlit)
# ================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import lightgbm as lgb
import joblib
from model_utils import load_model, preprocess

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="UrbanFlow-AI Mobility Forecast",
                   layout="wide",
                   page_icon="🚦")

st.title("🚦 UrbanFlow-AI — Smart Mobility Forecast Dashboard")
st.markdown("### Powered by LightGBM | Real-time Demand Forecasting & Analytics")

# ---------------------------------------------------------------
# LOAD MODEL & DATA
# ---------------------------------------------------------------
model = load_model()

# Default data preview
try:
    df = pd.read_csv("data/processed_full.csv", parse_dates=['hour'])
except:
    st.warning("No sample data found — please upload your dataset below.")
    df = None

# ---------------------------------------------------------------
# DATA UPLOAD SECTION
# ---------------------------------------------------------------
uploaded = st.file_uploader("📂 Upload your own mobility dataset (CSV)", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=['hour'])

if df is not None:
    st.success("✅ Data loaded successfully.")
    df = preprocess(df)

    # -----------------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------------
    feature_cols = [c for c in df.columns if c not in ['hour','demand','date']]
    preds = model.predict(df[feature_cols])
    df['Predicted'] = preds

    # -----------------------------------------------------------
    # KPIs
    # -----------------------------------------------------------
    avg_demand = round(df['Predicted'].mean(), 1)
    peak_hour = df.loc[df['Predicted'].idxmax(), 'hour']
    corr_temp = round(df['Predicted'].corr(df['temperature']), 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Demand", f"{avg_demand}")
    col2.metric("Peak Hour", str(peak_hour))
    col3.metric("Temp–Demand Corr", f"{corr_temp}")

    st.divider()

    # -----------------------------------------------------------
    # VISUALIZATIONS
    # -----------------------------------------------------------

    # 1️⃣ Forecast line chart
    st.subheader("📈 Forecast: Actual vs Predicted Demand")
    if 'demand' in df.columns:
        fig1 = px.line(df, x='hour', y=['demand','Predicted'],
                       labels={'value':'Demand','hour':'Timestamp'},
                       title="Actual vs Predicted Demand")
    else:
        fig1 = px.line(df, x='hour', y='Predicted',
                       title="Predicted Demand Forecast")
    st.plotly_chart(fig1, use_container_width=True)

    # 2️⃣ Heatmap: Hour vs Day
    st.subheader("🔥 Demand Pattern Heatmap")
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek
    heat_df = df.groupby(['day_of_week','hour_of_day'])['Predicted'].mean().reset_index()
    fig2 = px.density_heatmap(heat_df, x='hour_of_day', y='day_of_week', z='Predicted',
                              color_continuous_scale='Viridis',
                              title='Average Predicted Demand by Hour & Day')
    st.plotly_chart(fig2, use_container_width=True)

    # 3️⃣ Weather correlation
    st.subheader("🌡️ Temperature vs Demand Correlation")
    if 'temperature' in df.columns:
        fig3 = px.scatter(df, x='temperature', y='Predicted',
                          trendline="ols", title="Temperature vs Predicted Demand")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Temperature column not found — skipping weather correlation chart.")

    # 4️⃣ Feature importance
    st.subheader("🧠 Model Feature Importance")
    importance = model.feature_importance()
    feature_names = model.feature_name()
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False).head(15)
    fig4 = px.bar(imp_df, x='importance', y='feature', orientation='h',
                  title="Top 15 Feature Importances")
    st.plotly_chart(fig4, use_container_width=True)

    st.success("✅ Dashboard generated successfully.")
else:
    st.info("👆 Upload a CSV file or include a sample dataset in the `data/` folder to begin.")
