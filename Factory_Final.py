import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.io as pio

# ------------------ THEME ------------------
pio.templates.default = "plotly_dark"

COLORS = ["#00C6FF", "#0072FF", "#00FF94", "#FFD700", "#FF4B4B"]

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Nassau Candy | Deep Insights", layout="wide")

# ------------------ CSS ------------------
st.markdown("""
<style>
.stTabs [data-baseweb="tab"] {
    color: #FFFF00 !important;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #004a99 !important;
    color: #ffffff !important;
}
[data-testid="stMetricValue"] {
    font-size: 24px;
    color: #00C6FF;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')

    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    df['profit_margin'] = (df['gross_profit'] / df['sales']) * 100

    return df

df = load_data()
model = joblib.load('model.pkl')

# ------------------ SIDEBAR ------------------
st.sidebar.title("🛠️ Filters")

product = st.sidebar.selectbox("Product", sorted(df['product_name'].unique()))
region = st.sidebar.multiselect("Region", df['region'].unique(), default=df['region'].unique())
ship_mode = st.sidebar.multiselect("Ship Mode", df['ship_mode'].unique(), default=df['ship_mode'].unique())

date_range = st.sidebar.date_input(
    "Date Range",
    [df['order_date'].min(), df['order_date'].max()]
)

min_sales = st.sidebar.slider(
    "Min Sales",
    int(df['sales'].min()),
    int(df['sales'].max()),
    int(df['sales'].quantile(0.25))
)

lead_time_range = st.sidebar.slider(
    "Lead Time",
    0,
    int(df['lead_time'].max()),
    (0, int(df['lead_time'].max()))
)

priority = st.sidebar.slider("Optimization Priority (Speed vs Profit)", 0, 100, 50)

# ------------------ FILTER ------------------
mask = (
    (df['product_name'] == product) &
    (df['sales'] >= min_sales) &
    (df['lead_time'].between(lead_time_range[0], lead_time_range[1]))
)

if region:
    mask &= df['region'].isin(region)

if ship_mode:
    mask &= df['ship_mode'].isin(ship_mode)

if len(date_range) == 2:
    mask &= (df['order_date'] >= pd.Timestamp(date_range[0])) & (df['order_date'] <= pd.Timestamp(date_range[1]))

f_df = df[mask]

# ------------------ HEADER ------------------
st.title("📊 Nassau Candy Distributor Dashboard")
st.caption(f"Filtered Data: {len(f_df)} rows")

if f_df.empty:
    st.warning("⚠️ No data available for selected filters. Adjust filters.")

# ------------------ KPIs ------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Avg Lead Time", f"{f_df['lead_time'].mean():.1f}" if not f_df.empty else "N/A")
k2.metric("Total Profit", f"${f_df['gross_profit'].sum():,.0f}" if not f_df.empty else "N/A")
k3.metric("Avg Margin", f"{f_df['profit_margin'].mean():.1f}%" if not f_df.empty else "N/A")
k4.metric("Total Orders", len(f_df))

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Simulator",
    "🔄 What-If Analysis",
    "📊 Deep Insights",
    "📝 Recommendations",
    "⚠️ Risk Panel"
])

# ------------------ TAB 1 ------------------
with tab1:
    st.subheader("Factory Simulation")

    if not f_df.empty:
        factories = ["Lot's O' Nuts", "Wicked Choccy's", "Sugar Shack", "Secret Factory"]

        sim_data = pd.DataFrame({
            'Factory': factories,
            'Lead Time': [f_df['lead_time'].mean() * np.random.uniform(0.7, 1.2) for _ in factories]
        })

        fig = px.bar(sim_data, x='Factory', y='Lead Time', color='Factory',
                     color_discrete_sequence=COLORS)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display")

# ------------------ TAB 2 ------------------
with tab2:
    st.subheader("What-If Scenario Analysis")

    if not f_df.empty:
        current = f_df['lead_time'].mean()
        current_profit = f_df['gross_profit'].mean()

        optimized = current * (1 - (priority / 200))
        adjusted_profit = current_profit * (1 - (priority / 300))

        col1, col2, col3 = st.columns(3)

        col1.metric("Lead Time", f"{optimized:.2f}",
                    f"{((current - optimized)/current)*100:.1f}% ↓")

        col2.metric("Profit Impact", f"${adjusted_profit:,.0f}",
                    f"{((adjusted_profit - current_profit)/current_profit)*100:.1f}%")

        col3.metric("Priority", f"{priority}%",
                    "Speed" if priority > 50 else "Profit")

        comparison_df = pd.DataFrame({
            "Scenario": ["Current", "Optimized"],
            "Lead Time": [current, optimized],
            "Profit": [current_profit, adjusted_profit]
        })

        fig = px.bar(comparison_df, x="Scenario",
                     y=["Lead Time", "Profit"],
                     barmode="group",
                     color_discrete_sequence=["#00C6FF", "#FF4B4B"])

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available")

# ------------------ TAB 3 ------------------
with tab3:
    st.subheader("Deep Insights")

    if not f_df.empty:

        col1, col2 = st.columns(2)

        fig1 = px.scatter(f_df, x="units", y="lead_time",
                          color="ship_mode",
                          color_discrete_sequence=COLORS)
        col1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(f_df, x="lead_time",
                            color_discrete_sequence=["#00FF94"])
        col2.plotly_chart(fig2, use_container_width=True)

        ship_perf = f_df.groupby('ship_mode')['lead_time'].mean().reset_index()

        fig3 = px.bar(ship_perf, x='ship_mode', y='lead_time',
                      color='ship_mode',
                      color_discrete_sequence=COLORS)

        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("No data available")

# ------------------ TAB 4 ------------------
with tab4:
    st.subheader("Recommendations")

    if not f_df.empty:
        factories = ["Lot's O' Nuts", "Wicked Choccy's", "Sugar Shack", "Secret Factory"]

        sim_data = pd.DataFrame({
            'Factory': factories,
            'Lead Time': [f_df['lead_time'].mean() * np.random.uniform(0.7, 1.2) for _ in factories]
        })

        best_factory = sim_data.sort_values('Lead Time').iloc[0]['Factory']

        st.success(f"✅ Move production to **{best_factory}**")

        st.info(f"""
        - Highest sales region: {f_df.groupby('region')['sales'].sum().idxmax()}
        - Slowest ship mode: {f_df.groupby('ship_mode')['lead_time'].mean().idxmax()}
        """)
    else:
        st.info("No data available")

# ------------------ TAB 5 ------------------
with tab5:
    st.subheader("Risk & Impact Panel")

    if not f_df.empty:
        risk_df = f_df.copy()

        risk_df['risk_level'] = np.where(
            risk_df['lead_time'] > risk_df['lead_time'].quantile(0.75),
            "High",
            np.where(
                risk_df['lead_time'] > risk_df['lead_time'].quantile(0.5),
                "Medium",
                "Low"
            )
        )

        risk_counts = risk_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]

        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Count",
            color="Risk",
            color_discrete_map={
                "Low": "#00FF94",
                "Medium": "#FFD700",
                "High": "#FF4B4B"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        high_risk = (risk_df['risk_level'] == "High").sum()

        if high_risk > 0:
            st.error(f"⚠️ {high_risk} High-Risk Assignments Detected")
        else:
            st.success("✅ No High-Risk Issues")

        st.download_button("Download Report", risk_df.to_csv(), "report.csv")

    else:
        st.info("No data available")