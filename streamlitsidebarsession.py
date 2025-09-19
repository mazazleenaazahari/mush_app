import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# -------------------------------
# Dummy dataset
# -------------------------------
def make_dummy_data(n_days=60, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    temp = 23 + 2*np.sin(np.linspace(0, 3*np.pi, n_days)) + rng.normal(0, 0.5, n_days)
    hum = 72 + 3*np.cos(np.linspace(0, 2*np.pi, n_days)) + rng.normal(0, 1.0, n_days)
    co2 = 420 + 10*np.sin(np.linspace(0, 4*np.pi, n_days)) + rng.normal(0, 3.0, n_days)
    return pd.DataFrame({
        "timestamp": dates,
        "temperature_C": np.round(temp, 2),
        "humidity_%": np.round(hum, 2),
        "CO2_ppm": np.round(co2, 1),
    })


df = make_dummy_data()

# -------------------------------
# Sidebar navigation with buttons
# -------------------------------
st.sidebar.title("Navigation")

# Initialize session_state
if "page" not in st.session_state:
    st.session_state.page = "Overview"

# Sidebar buttons (fixed size via CSS)
st.sidebar.markdown(
    """
    <style>
    div[data-testid="stSidebar"] button {
        width: 200px;
        height: 40px;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("Overview"):
    st.session_state.page = "Overview"
if st.sidebar.button("Explore Data"):
    st.session_state.page = "Explore Data"
if st.sidebar.button("Forecast (Prophet)"):
    st.session_state.page = "Forecast (Prophet)"

page = st.session_state.page

# -------------------------------
# Pages
# -------------------------------
if page == "Overview":
    st.title("🍄 Mushroom Analytics – Simple App")
    st.subheader("General Information")
    st.markdown(
        """
This beginner app demonstrates:
- Sidebar **buttons** for navigation (instead of radio buttons).  
- A built-in dummy mushroom dataset.  
- Quick data visualization.  
- A simple Prophet forecasting demo.
        """
    )
    st.info("👉 Use the sidebar buttons on the left to switch pages.")

elif page == "Explore Data":
    st.title("📊 Explore Data")
    st.dataframe(df)

    x_col = st.selectbox("X-axis", df.columns, index=0)
    y_col = st.selectbox("Y-axis", df.columns, index=1)
    chart_type = st.radio("Chart type", ["Line", "Bar", "Scatter"], horizontal=True)

    if chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")

    st.plotly_chart(fig, use_container_width=True)

elif page == "Forecast (Prophet)":
    st.title("🔮 Forecast with Prophet")

    if not PROPHET_AVAILABLE:
        st.error("Prophet not installed. Run `pip install prophet` first.")
    else:
        target = st.selectbox("Variable", ["temperature_C", "humidity_%", "CO2_ppm"])
        periods = st.slider("Days to forecast", 7, 60, 21, step=7)

        if st.button("Run Forecast"):
            prophet_df = df[["timestamp", target]].rename(columns={"timestamp": "ds", target: "y"})
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=periods, freq="D")
            forecast = model.predict(future)

            st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
            st.dataframe(forecast.tail(10))
