"""
Ethiopia Malaria Burden & Supply Chain Dashboard.

Local run:
    streamlit run app.py

Cloud deployment (Streamlit Community Cloud):
    Push this repo to GitHub, connect it at https://share.streamlit.io,
    and set the credentials below via the app's "Secrets" panel instead of
    a local secrets.toml file (see .streamlit/secrets.toml.example).
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

st.set_page_config(page_title="Ethiopia Malaria Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# AUTHENTICATION
# Credentials live in .streamlit/secrets.toml (local) or the Streamlit Cloud
# "Secrets" panel (deployed) -- never hardcoded here and never committed.
# See .streamlit/secrets.toml.example for the expected format.
# ---------------------------------------------------------------------------
if "credentials" not in st.secrets:
    st.error(
        "No credentials configured. Copy .streamlit/secrets.toml.example to "
        ".streamlit/secrets.toml (local) or paste its contents into your "
        "Streamlit Cloud app's Secrets panel, then edit the usernames/passwords."
    )
    st.stop()

credentials = {"usernames": dict(st.secrets["credentials"]["usernames"])}

authenticator = stauth.Authenticate(
    credentials,
    cookie_name=st.secrets["cookie"]["name"],
    cookie_key=st.secrets["cookie"]["key"],
    cookie_expiry_days=st.secrets["cookie"].get("expiry_days", 7),
)

authenticator.login()

if st.session_state.get("authentication_status") is False:
    st.error("Username or password is incorrect.")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password.")
    st.stop()

authenticator.logout("Logout", "sidebar")
st.sidebar.success(f"Logged in as {st.session_state['name']}")

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
st.title("Ethiopia Malaria Burden & Supply Chain Dashboard")


@st.cache_data
def load_data():
    model_frame = pd.read_csv(os.path.join(DATA_DIR, "model_frame.csv"), parse_dates=["date"])
    try:
        forecast_df = pd.read_csv(os.path.join(OUT_DIR, "forecast_df.csv"), parse_dates=["date"])
    except FileNotFoundError:
        forecast_df = pd.DataFrame(columns=["ADM3_PCODE", "ADM3_EN", "date", "forecast_cases", "step_ahead"])
    try:
        annual = pd.read_csv(os.path.join(OUT_DIR, "woreda_need_vs_allocation_2024.csv"))
    except FileNotFoundError:
        annual = pd.DataFrame()
    return model_frame, forecast_df, annual


model_frame, forecast_df, annual = load_data()

tab1, tab2, tab3 = st.tabs(["Woreda drill-down", "National map", "Supply chain"])

with tab1:
    woreda = st.selectbox("Woreda", sorted(model_frame["ADM3_EN"].dropna().unique()))
    sub = model_frame[model_frame["ADM3_EN"] == woreda].sort_values("date")
    pcode = sub["ADM3_PCODE"].iloc[0]
    fc = forecast_df[forecast_df["ADM3_PCODE"] == pcode].sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["cases"], name="Actual cases"))
    if not fc.empty:
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_cases"], name="Forecast",
                                  line=dict(color="firebrick")))
    fig.update_layout(title=f"{woreda}: malaria cases", height=450,
                       xaxis_title="Date", yaxis_title="Cases")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if {"shp_lat", "shp_lon"}.issubset(model_frame.columns) and not forecast_df.empty:
        latest = forecast_df[forecast_df["step_ahead"] == 1].merge(
            model_frame[["ADM3_PCODE", "shp_lat", "shp_lon", "ADM1_EN", "burden_tier"]]
            .drop_duplicates("ADM3_PCODE"),
            on="ADM3_PCODE", how="left")
        fig = px.scatter_mapbox(
            latest, lat="shp_lat", lon="shp_lon", size="forecast_cases",
            color="burden_tier",
            category_orders={"burden_tier": ["very high", "high", "moderate", "low"]},
            hover_name="ADM3_EN", hover_data=["ADM1_EN", "forecast_cases"],
            zoom=4.6, height=550, size_max=28,
            title="Next-month forecast cases by woreda",
        )
        fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run src/generate_forecast.py to populate this map.")

with tab3:
    if not annual.empty:
        region = st.selectbox(
            "Region", ["All regions"] + sorted(annual["ADM1_EN"].dropna().unique().tolist())
        )
        sub = annual if region == "All regions" else annual[annual["ADM1_EN"] == region]
        agg = (sub.groupby("burden_tier")
               .agg(need=("annual_need_al", "sum"), allocation=("al_allocation", "sum"))
               .reindex(["very high", "high", "moderate", "low"]).dropna(how="all").reset_index())

        fig = go.Figure()
        fig.add_bar(x=agg["burden_tier"], y=agg["need"], name="Annual need (AL courses)")
        fig.add_bar(x=agg["burden_tier"], y=agg["allocation"], name="Optimized allocation")
        fig.update_layout(barmode="group",
                           title=f"AL courses: need vs optimized allocation — {region}", height=420)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Woredas", len(sub))
        c2.metric("Total need", f"{sub['annual_need_al'].sum():,.0f}")
        c3.metric("Avg coverage", f"{sub['al_coverage_pct'].mean():.1f}%")
    else:
        st.info("Run src/optimize_supply_chain.py to populate this tab.")
