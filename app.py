# ================================================================
# UrbanFlow-AI 2.0 — Advanced Streamlit Dashboard
# ================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
import joblib
from model_utils import load_model, preprocess

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="UrbanFlow-AI 2.0",
                   layout="wide",
                   page_icon="🚦")

st.title("🚦 UrbanFlow-AI — Smart Mobility Forecast & Analytics Dashboard")
st.markdown("### Powered by LightGBM | Real-Time Demand Forecasting & Deep Analytics")

# ---------------------------------------------------------------
# LOAD MODEL & DATA
# ---------------------------------------------------------------
model = load_model()

try:
    df = pd.read_csv("data/processed_full.csv", parse_dates=['hour'])
except:
    st.warning("⚠️ No sample data found — please upload your dataset below.")
    df = None

# ---------------------------------------------------------------
# DATA UPLOAD SECTION
# ---------------------------------------------------------------
uploaded = st.file_uploader("📂 Upload your mobility dataset (CSV)", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=['hour'])

if df is not None:
    st.success("✅ Data loaded successfully.")
    df = preprocess(df)

    # -----------------------------------------------------------
    # MODEL PREDICTION
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
    weekend_share = df[df['is_weekend']==1]['Predicted'].sum() / df['Predicted'].sum() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌆 Average Demand", f"{avg_demand}")
    col2.metric("🕒 Peak Hour", str(peak_hour))
    col3.metric("🌡️ Temp–Demand Corr", f"{corr_temp}")
    col4.metric("🎉 Weekend Demand Share", f"{weekend_share:.1f}%")

    st.divider()

    # -----------------------------------------------------------
    # MAIN FORECAST VISUALIZATION
    # -----------------------------------------------------------
    st.subheader("📈 Demand Forecast — Actual vs Predicted")
    if 'demand' in df.columns:
        fig1 = px.line(df, x='hour', y=['demand','Predicted'],
                       labels={'value':'Demand','hour':'Timestamp'},
                       title="Actual vs Predicted Demand (Time Series)")
    else:
        fig1 = px.line(df, x='hour', y='Predicted',
                       title="Predicted Demand Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------------------------------------
    # ADDITIONAL VISUALS
    # -----------------------------------------------------------
    st.subheader("📊 Analytical Insights")

    col_a, col_b = st.columns(2)

    # 1️⃣ Bar Chart: Avg Demand by Day of Week
    df['day_of_week'] = df['hour'].dt.day_name()
    avg_by_day = df.groupby('day_of_week')['Predicted'].mean().reindex([
        'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'
    ])
    fig_bar = px.bar(avg_by_day, x=avg_by_day.index, y='Predicted',
                     title="Average Predicted Demand by Day of Week",
                     labels={'Predicted':'Avg Demand','day_of_week':'Day'})
    col_a.plotly_chart(fig_bar, use_container_width=True)

    # 2️⃣ Pie Chart: Weekend vs Weekday
    weekend_sum = df.groupby('is_weekend')['Predicted'].sum().reset_index()
    weekend_sum['Label'] = weekend_sum['is_weekend'].map({0:'Weekday',1:'Weekend'})
    fig_pie = px.pie(weekend_sum, values='Predicted', names='Label',
                     title='Demand Share: Weekday vs Weekend',
                     color_discrete_sequence=px.colors.sequential.RdBu)
    col_b.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # 3️⃣ Heatmap — Hour vs Day
    st.subheader("🔥 Demand Intensity Pattern (Hour × Day)")
    df['hour_of_day'] = df['hour'].dt.hour
    df['dow'] = df['hour'].dt.dayofweek
    heat_df = df.groupby(['dow','hour_of_day'])['Predicted'].mean().reset_index()
    fig_heat = px.density_heatmap(heat_df, x='hour_of_day', y='dow', z='Predicted',
                                  color_continuous_scale='Turbo',
                                  title='Average Predicted Demand by Hour & Day')
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # 4️⃣ Temperature vs Demand (Scatter)
    st.subheader("🌡️ Temperature Impact on Demand")
    if 'temperature' in df.columns:
        try:
            import statsmodels.api
            fig_scatter = px.scatter(df, x='temperature', y='Predicted',
                                     trendline='ols',
                                     title='Temperature vs Predicted Demand (with Trendline)')
        except:
            fig_scatter = px.scatter(df, x='temperature', y='Predicted',
                                     title='Temperature vs Predicted Demand')
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Temperature data not available.")

    # 5️⃣ Histogram — Distribution of Predicted Demand
    st.subheader("📉 Distribution of Predicted Demand")
    fig_hist = px.histogram(df, x='Predicted', nbins=30,
                            title="Demand Distribution (All Hours)",
                            color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # 6️⃣ Feature Importance
    st.subheader("🧠 Feature Importance (Model Explainability)")
    importance = model.feature_importance()
    feature_names = model.feature_name()
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False).head(15)
    fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h',
                     title="Top 15 Influential Features")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.success("✅ Dashboard fully generated — all visuals loaded.")
else:
    st.info("👆 Upload a CSV file or include a sample dataset in the `data/` folder to begin.")
