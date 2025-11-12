# ================================================================
# UrbanFlow-AI — Smart Mobility Forecast & Analytics Dashboard (Pro)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, os

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="UrbanFlow-AI Dashboard", layout="wide", page_icon="🚦")
st.title("🚦 UrbanFlow-AI — Smart Mobility Forecast & Analytics Dashboard")
st.caption("**Interactive, Filterable, and Forecast-Driven Dashboard powered by LightGBM**")

# ---------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path="models/urbanflow_model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# ---------------------------------------------------------------
# DATA LOADING (Sample or Uploaded)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path):
    """Safely load CSV file"""
    if not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            st.warning("⚠️ Sample file exists but is empty.")
            return None
        df = pd.read_csv(path)
        if df.empty:
            st.warning("⚠️ CSV file is empty.")
            return None
        return df
    except pd.errors.EmptyDataError:
        st.warning("⚠️ Invalid CSV — no data found.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Failed to load CSV: {e}")
        return None

# Load default sample dataset
default_path = "data/processed_full.csv"
df = load_csv(default_path)

# Allow user upload (takes priority)
uploaded = st.file_uploader("📂 Upload CSV to override sample dataset", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

if df is None:
    st.warning("⚠️ No valid dataset found. Please upload a CSV or add one to `data/processed_full.csv`.")
    st.stop()

st.success(f"✅ Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

# ---------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------
def detect_timestamp(df):
    for c in df.columns:
        if any(x in c.lower() for x in ['hour', 'time', 'date', 'timestamp', 'start']):
            return c
    return None

def prepare_data(df):
    df = df.copy()
    ts_col = detect_timestamp(df)
    if ts_col is None:
        st.error("❌ No timestamp-like column found.")
        st.stop()
    df['hour'] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=['hour']).sort_values('hour')

    if 'demand' not in df.columns:
        df['demand'] = 1

    # Create temperature if not present
    if not any('temp' in c.lower() for c in df.columns):
        df['temperature'] = np.random.normal(20, 5, len(df))

    df['day_of_week'] = df['hour'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['hour_of_day'] = df['hour'].dt.hour
    df['month'] = df['hour'].dt.month
    df['year'] = df['hour'].dt.year
    df['day'] = df['hour'].dt.day

    # Lag features
    for lag in [1,2,3,6,12,24]:
        df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
    df['rolling_mean_6h'] = df['demand'].rolling(window=6, min_periods=1).mean()
    df['rolling_mean_24h'] = df['demand'].rolling(window=24, min_periods=1).mean()
    df = df.dropna().reset_index(drop=True)
    return df

df = prepare_data(df)

# Align features with model
feature_cols = [f for f in df.columns if f not in ['hour','demand','date']]
try:
    model_features = model.feature_name()
    for f in model_features:
        if f not in df.columns:
            df[f] = 0
    X = df[model_features]
except Exception:
    X = df[feature_cols]

# Predict
try:
    df['Predicted'] = model.predict(X)
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

df['day_name'] = df['hour'].dt.day_name()

# ---------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------
st.sidebar.header("🔍 Filters")

# Date range
min_date, max_date = df['hour'].min().date(), df['hour'].max().date()
date_range = st.sidebar.date_input("📅 Date Range", [min_date, max_date])
if isinstance(date_range, list) and len(date_range) == 2:
    df = df[(df['hour'].dt.date >= date_range[0]) & (df['hour'].dt.date <= date_range[1])]

# Temperature
tmin, tmax = float(df['temperature'].min()), float(df['temperature'].max())
temp_range = st.sidebar.slider("🌡️ Temperature", tmin, tmax, (tmin, tmax))
df = df[(df['temperature'] >= temp_range[0]) & (df['temperature'] <= temp_range[1])]

# Days of Week
days = df['day_name'].unique().tolist()
selected_days = st.sidebar.multiselect("📅 Days of Week", days, default=days)
df = df[df['day_name'].isin(selected_days)]

# Hour Filter
hour_min, hour_max = df['hour_of_day'].min(), df['hour_of_day'].max()
hour_range = st.sidebar.slider("🕒 Hour Range", 0, 23, (int(hour_min), int(hour_max)))
df = df[(df['hour_of_day'] >= hour_range[0]) & (df['hour_of_day'] <= hour_range[1])]

st.sidebar.success(f"Filtered Rows: {len(df)}")

# ---------------------------------------------------------------
# KPI METRICS
# ---------------------------------------------------------------
avg_demand = round(df['Predicted'].mean(), 1)
peak_hour = df.loc[df['Predicted'].idxmax(), 'hour']
avg_temp = round(df['temperature'].mean(), 1)
corr_temp = round(df['Predicted'].corr(df['temperature']), 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚗 Avg Demand", avg_demand)
c2.metric("🔥 Peak Hour", str(peak_hour))
c3.metric("🌡️ Avg Temp", avg_temp)
c4.metric("📈 Temp–Demand Corr", corr_temp)
st.divider()

# ---------------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Overview", "📊 Patterns", "📈 Forecast", "🔗 Correlations", "🧠 Model Insights"]
)

# 🏠 Overview
with tab1:
    colA, colB = st.columns(2)
    # Pie: Weekend vs Weekday
    wk = df.groupby('is_weekend')['Predicted'].sum().reset_index()
    wk['Label'] = wk['is_weekend'].map({0:'Weekday',1:'Weekend'})
    fig_pie = px.pie(wk, values='Predicted', names='Label', title="Demand Share: Weekday vs Weekend")
    colA.plotly_chart(fig_pie, use_container_width=True)

    # Histogram
    fig_hist = px.histogram(df, x='Predicted', nbins=30, title="Predicted Demand Distribution")
    colB.plotly_chart(fig_hist, use_container_width=True)

# 📊 Patterns
with tab2:
    colC, colD = st.columns(2)
    # Bar: Avg by Day
    avg_day = df.groupby('day_name')['Predicted'].mean().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    ).reset_index()
    fig_bar = px.bar(avg_day, x='day_name', y='Predicted', title="Average Demand by Day of Week")
    colC.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap
    heat = df.groupby(['day_name','hour_of_day'])['Predicted'].mean().reset_index()
    fig_heat = px.density_heatmap(heat, x='hour_of_day', y='day_name', z='Predicted',
                                  title="Hourly Demand Intensity by Day", color_continuous_scale='Turbo')
    colD.plotly_chart(fig_heat, use_container_width=True)

# 📈 Forecast
with tab3:
    st.subheader("Forecast: Actual vs Predicted Demand")
    if 'demand' in df.columns and df['demand'].sum() > 0:
        fig_line = px.line(df, x='hour', y=['demand','Predicted'],
                           labels={'value':'Demand','hour':'Time'},
                           title="Actual vs Predicted Demand Over Time")
    else:
        fig_line = px.line(df, x='hour', y='Predicted', title="Predicted Demand Over Time")
    st.plotly_chart(fig_line, use_container_width=True)

# 🔗 Correlations
with tab4:
    st.subheader("Temperature vs Demand Correlation")
    try:
        import statsmodels.api
        fig_sc = px.scatter(df, x='temperature', y='Predicted', trendline='ols',
                            title="Temperature vs Predicted Demand (Trendline)")
    except Exception:
        fig_sc = px.scatter(df, x='temperature', y='Predicted', title="Temperature vs Predicted Demand")
    st.plotly_chart(fig_sc, use_container_width=True)

    # Correlation heatmap
    corr = df.select_dtypes('number').corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

# 🧠 Model Insights
with tab5:
    st.subheader("Feature Importance (Top 15)")
    try:
        imp = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False).head(15)
        fig_imp = px.bar(imp, x='importance', y='feature', orientation='h', title="Top Feature Importances")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")

st.success("✅ Dashboard loaded successfully.")
st.caption("Upload a new CSV anytime to instantly update forecasts and visuals.")
