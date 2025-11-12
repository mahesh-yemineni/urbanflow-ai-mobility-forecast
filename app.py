# app.py
# UrbanFlow-AI Pro — robust Streamlit dashboard that:
# - Loads sample data from data/processed_full.csv by default
# - Accepts an uploaded CSV and uses it instead
# - Preprocesses (hourly aggregation if needed), creates lag/rolling features
# - Aligns features to the trained LightGBM model and predicts
# - Renders multi-tab visuals with sidebar filters and KPI row
# - Handles missing optional libs (statsmodels) gracefully

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime, date

# ---------------------------
# Utilities / Preprocessing
# ---------------------------

@st.cache_data(show_spinner=False)
def load_model(path="models/urbanflow_model.pkl"):
    if not os.path.exists(path):
        return None, f"Model not found at {path}"
    try:
        model = joblib.load(path)
        # lightgbm.Booster from joblib supports .feature_name() and .feature_importance()
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

def detect_timestamp_column(df):
    # common timestamp names
    candidates = [c for c in df.columns if c.lower() in (
        'hour','timestamp','started_at','starttime','start_time','date','datetime','time')]
    if candidates:
        return candidates[0]
    # fuzzy: contains 'time' or 'date' or 'start'
    for c in df.columns:
        low = c.lower()
        if 'time' in low or 'date' in low or 'start' in low:
            return c
    return None

def ensure_hour_column(df, ts_col):
    # convert and create 'hour' floored to hour resolution
    df = df.copy()
    df['hour'] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=['hour'])
    df['hour'] = df['hour'].dt.floor('H')
    return df

def aggregate_to_hourly(df):
    """If dataset is trip-level (many rows per hour), aggregate to counts per hour."""
    # heuristic: if many rows per unique hour -> trip-level
    uniq_hours = df['hour'].nunique()
    if len(df) > uniq_hours * 2:
        # aggregate count of trips starting that hour
        # if 'demand' exists and is already aggregated, skip
        agg = df.groupby('hour').agg(
            demand = ('hour','size'),
            temperature = (lambda x: np.nan)  # placeholder; we'll fill if temperature exists below
        ).reset_index()
        # If original had temperature or temp-like, attempt to avg it
        temp_cols = [c for c in df.columns if 'temp' in c.lower() or c.lower() in ('t1','t2','temperature')]
        if temp_cols:
            temp = df.groupby('hour')[temp_cols[0]].mean().reset_index(name='temperature')
            agg = agg.drop(columns=['temperature'])
            agg = agg.merge(temp, on='hour', how='left')
        else:
            agg['temperature'] = np.nan
        return agg
    else:
        # assume already hourly aggregated; keep columns hour,demand,temperature if present
        cols = ['hour']
        if 'demand' in df.columns:
            cols.append('demand')
        if any('temp' in c.lower() for c in df.columns):
            temp_col = [c for c in df.columns if 'temp' in c.lower()][0]
            df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
            df = df.rename(columns={temp_col: 'temperature'})
            cols.append('temperature')
        # ensure demand exists
        if 'demand' not in df.columns:
            # if not present, treat each row as 1 (but here we are in hourly assumption, so fill with 0)
            df['demand'] = 0
            cols.append('demand')
        return df[cols].copy()

def create_time_features(df):
    df = df.copy()
    df['day_of_week'] = df['hour'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['hour_of_day'] = df['hour'].dt.hour
    df['month'] = df['hour'].dt.month
    df['year'] = df['hour'].dt.year
    df['day'] = df['hour'].dt.day
    return df

def create_lag_roll_features(df, lags=(1,2,3,6,12,24)):
    df = df.sort_values('hour').reset_index(drop=True)
    for lag in lags:
        df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
    df['rolling_mean_6h'] = df['demand'].rolling(window=6, min_periods=1).mean()
    df['rolling_mean_24h'] = df['demand'].rolling(window=24, min_periods=1).mean()
    # drop rows with NaNs caused by lags
    max_lag = max(lags)
    df = df.dropna().reset_index(drop=True)
    return df

def align_features_with_model(df, model):
    """Ensure df contains all features expected by model.feature_name(); fill missing with sensible defaults."""
    # Get model feature names if possible
    try:
        expected = list(model.feature_name())
    except Exception:
        # If model doesn't expose feature names, assume df columns are OK
        return df, []
    missing = [f for f in expected if f not in df.columns]
    if missing:
        # Fill missing features with zeros or mean if numeric
        for m in missing:
            df[m] = 0
    # Reorder columns to expected
    df = df.copy()
    try:
        df = df[expected]
    except Exception:
        # fallback: return df with all columns
        pass
    return df, missing

# ---------------------------
# App UI & Flow
# ---------------------------

st.set_page_config(layout="wide", page_title="UrbanFlow-AI Pro Dashboard", page_icon="🚦")
st.title("🚦 UrbanFlow-AI — Mobility Forecast & Analytics (Pro)")

# Load model
model, model_err = load_model()
if model is None:
    st.error(f"Model load error: {model_err}\nPlace your trained model at models/urbanflow_model.pkl and redeploy.")
    st.stop()

# Load default sample dataset from repo data folder (if present)
default_path = "data/processed_full.csv"
df = None
if os.path.exists(default_path):
    try:
        df = pd.read_csv(default_path, parse_dates=['hour'])
    except Exception:
        # fallback: try without parse_dates and detect ts later
        df = pd.read_csv(default_path)
else:
    # no sample file in repo
    df = None

# File uploader (uploaded file takes precedence)
uploaded = st.file_uploader("Upload CSV to override sample dataset (optional)", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()

if df is None:
    st.warning("No sample dataset found in `data/processed_full.csv`. Please either upload a CSV or add a sample file to the repo `data/` folder.")
    st.stop()

st.success("Dataframe loaded. Preprocessing...")

# Basic preprocessing pipeline
# 1) Ensure we have a timestamp column -> 'hour'
ts_col = detect_timestamp_column(df)
if ts_col is None:
    st.error("Could not detect timestamp column. Ensure your CSV has a column like 'hour','timestamp','started_at', or 'date'.")
    st.stop()

df = ensure_hour_column(df, ts_col)

# 2) Aggregate / ensure hourly
df = aggregate_to_hourly(df)

# 3) Ensure numeric demand and temperature columns
if 'demand' not in df.columns:
    df['demand'] = 0
if 'temperature' not in df.columns:
    df['temperature'] = np.nan
else:
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')

# 4) Time features
df = create_time_features(df)

# 5) If data seems too short for lags, we still create lags but drop only first rows in create_lag_roll_features
try:
    df = create_lag_roll_features(df)
except Exception as e:
    st.warning(f"Could not create all lag features due to dataset length: {e}")
    # attempt minimal lags
    df['demand_lag_1'] = df['demand'].shift(1)
    df['rolling_mean_6h'] = df['demand'].rolling(window=6, min_periods=1).mean()
    df = df.dropna().reset_index(drop=True)

# Keep a copy for visuals before alignment
df_visual = df.copy()

# 6) Align features with model
X_for_model, missing_features = align_features_with_model(df, model)
if missing_features:
    st.warning(f"Model expects features that were missing in input; they were filled with zeros: {missing_features}")

# Predict
try:
    preds = model.predict(X_for_model)
    df_visual['Predicted'] = preds
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# -------------
# SIDEBAR FILTERS
# -------------
st.sidebar.header("Filters")
min_date, max_date = df_visual['hour'].min().date(), df_visual['hour'].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
# sanitize
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = min_date
    end_date = max_date

df_visual = df_visual[(df_visual['hour'].dt.date >= start_date) & (df_visual['hour'].dt.date <= end_date)]

# Temperature slider
temp_min = float(np.nanmin(df_visual['temperature'].fillna(0)))
temp_max = float(np.nanmax(df_visual['temperature'].fillna(0)))
if temp_min == temp_max:
    temp_min, temp_max = temp_min - 1, temp_max + 1
temp_range = st.sidebar.slider("Temperature range", temp_min, temp_max, (temp_min, temp_max))
df_visual = df_visual[(df_visual['temperature'] >= temp_range[0]) & (df_visual['temperature'] <= temp_range[1])]

# Day of week multiselect
day_labels = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df_visual['day_name'] = df_visual['hour'].dt.day_name()
selected_days = st.sidebar.multiselect("Days of week", options=day_labels, default=day_labels)
df_visual = df_visual[df_visual['day_name'].isin(selected_days)]

# Hour-of-day range
hmin, hmax = int(df_visual['hour_of_day'].min()), int(df_visual['hour_of_day'].max())
hour_slider = st.sidebar.slider("Hour of day", 0, 23, (hmin, hmax))
df_visual = df_visual[(df_visual['hour_of_day'] >= hour_slider[0]) & (df_visual['hour_of_day'] <= hour_slider[1])]

st.sidebar.markdown(f"**Filtered rows:** {len(df_visual)}")

# -------------
# KPIs Row
# -------------
avg_demand = df_visual['Predicted'].mean()
peak_idx = df_visual['Predicted'].idxmax() if len(df_visual)>0 else None
peak_time = df_visual.loc[peak_idx, 'hour'] if peak_idx is not None else "N/A"
avg_temp = df_visual['temperature'].mean()
temp_corr = df_visual['Predicted'].corr(df_visual['temperature']) if 'temperature' in df_visual.columns else np.nan
weekend_share = df_visual[df_visual['is_weekend']==1]['Predicted'].sum() / (df_visual['Predicted'].sum()+1e-9) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Predicted Demand", f"{avg_demand:.1f}")
c2.metric("Peak Time (Predicted)", str(peak_time))
c3.metric("Avg Temp", f"{avg_temp:.1f}")
c4.metric("Weekend Demand Share", f"{weekend_share:.1f}%")

st.markdown("---")

# -------------
# Tabs & Visuals
# -------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Patterns", "Forecast", "Correlations", "Model Insights"])

with tab1:
    st.subheader("Overview")
    colA, colB = st.columns(2)
    # Pie: Weekend vs Weekday
    wk = df_visual.groupby('is_weekend')['Predicted'].sum().reset_index()
    wk['label'] = wk['is_weekend'].map({0:'Weekday',1:'Weekend'})
    fig_pie = px.pie(wk, names='label', values='Predicted', title="Demand Share: Weekday vs Weekend")
    colA.plotly_chart(fig_pie, use_container_width=True)

    # Histogram of predicted demand
    fig_hist = px.histogram(df_visual, x='Predicted', nbins=30, title="Predicted Demand Distribution")
    colB.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("Patterns")
    c1_, c2_ = st.columns(2)
    # Bar: Avg predicted by day name
    avg_day = df_visual.groupby('day_name')['Predicted'].mean().reindex(day_labels).reset_index()
    fig_bar = px.bar(avg_day, x='day_name', y='Predicted', title="Avg Predicted Demand by Day")
    c1_.plotly_chart(fig_bar, use_container_width=True)
    # Heatmap: hour vs day
    heat = df_visual.groupby(['day_name','hour_of_day'])['Predicted'].mean().reset_index()
    # make day_name categorical ordered by day_labels
    heat['day_name'] = pd.Categorical(heat['day_name'], categories=day_labels, ordered=True)
    heat = heat.sort_values(['day_name','hour_of_day'])
    fig_heat = px.density_heatmap(heat, x='hour_of_day', y='day_name', z='Predicted',
                                  title="Avg Predicted Demand (Hour × Day)", color_continuous_scale='Turbo')
    c2_.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.subheader("Forecast Time Series")
    if 'demand' in df_visual.columns and df_visual['demand'].sum() > 0:
        fig_line = px.line(df_visual, x='hour', y=['demand','Predicted'], labels={'value':'Demand','hour':'Time'},
                           title="Actual vs Predicted Demand Over Time")
    else:
        fig_line = px.line(df_visual, x='hour', y='Predicted', title="Predicted Demand Over Time")
    st.plotly_chart(fig_line, use_container_width=True)

with tab4:
    st.subheader("Correlations & Relationships")
    if df_visual['temperature'].notna().sum() > 0:
        try:
            import statsmodels.api
            fig_sc = px.scatter(df_visual, x='temperature', y='Predicted', trendline='ols', title="Temperature vs Predicted Demand (OLS)")
        except Exception:
            fig_sc = px.scatter(df_visual, x='temperature', y='Predicted', title="Temperature vs Predicted Demand")
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("No temperature data available to show correlation.")

    # Additional correlation heatmap (numeric features)
    num_cols = df_visual.select_dtypes(include='number').columns.tolist()
    if len(num_cols) >= 2:
        corr = df_visual[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

with tab5:
    st.subheader("Model Insights")
    try:
        feat_imp = model.feature_importance()
        feat_names = model.feature_name()
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': feat_imp}).sort_values('importance', ascending=False).head(20)
        fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h', title="Top Feature Importances")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")

st.markdown("---")
st.caption("Tip: Upload a new CSV to test predictions on different data. Ensure your CSV has a timestamp column (e.g., 'started_at' / 'timestamp' / 'hour').")

# End of app
