from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Hospital Bed Occupancy Dashboard
# Run from the project root:
#   streamlit run streamlit_app.py
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Hospital Bed Occupancy",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, compact dashboard styling
st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #10243a; }
    [data-testid="stSidebar"] * { color: #f4f7fb !important; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e6eaf0;
        padding: 16px 18px; border-radius: 12px;
        box-shadow: 0 2px 8px rgba(16,36,58,.04);
    }
    h1, h2, h3 { color: #142b45; }
    .subtitle { color: #607086; margin-top: -12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    required = {
        "daily": DATA / "daily_occupancy.csv",
        "fact": DATA / "fact_admission.csv",
        "departments": DATA / "dim_department.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required processed CSV file(s) not found:\n" + "\n".join(missing)
            + "\n\nRun: python python/clean_data.py"
        )

    daily = pd.read_csv(required["daily"], parse_dates=["date"])
    fact = pd.read_csv(required["fact"], parse_dates=["admit_date", "discharge_date"])
    departments = pd.read_csv(required["departments"])

    daily["department_id"] = pd.to_numeric(daily["department_id"], errors="coerce")
    daily["occupied_beds"] = pd.to_numeric(daily["occupied_beds"], errors="coerce").fillna(0)
    daily["bed_capacity"] = pd.to_numeric(daily["bed_capacity"], errors="coerce").fillna(0)
    daily["occupancy_pct"] = pd.to_numeric(daily["occupancy_pct"], errors="coerce")
    fact["department_id"] = pd.to_numeric(fact["department_id"], errors="coerce")
    fact["los_days"] = pd.to_numeric(fact["los_days"], errors="coerce")
    fact["age_at_admission"] = pd.to_numeric(fact["age_at_admission"], errors="coerce")
    fact["is_valid"] = fact["is_valid"].astype(str)
    fact["status"] = fact["status"].astype(str)
    departments["department_id"] = pd.to_numeric(departments["department_id"], errors="coerce")
    return daily, fact, departments


try:
    daily, fact, departments = load_data()
except Exception as exc:
    st.error(f"Could not load dashboard data: {exc}")
    st.stop()

st.title("🏥 Hospital Bed Occupancy Dashboard")
st.markdown(
    '<p class="subtitle">Bed utilization, patient length of stay, and department-level capacity monitoring</p>',
    unsafe_allow_html=True,
)

# Sidebar filters
st.sidebar.title("Dashboard Filters")
min_date = daily["date"].min().date()
max_date = daily["date"].max().date()
date_range = st.sidebar.date_input(
    "Analysis period", value=(min_date, max_date),
    min_value=min_date, max_value=max_date
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)

department_names = departments.sort_values("department_name")["department_name"].dropna().tolist()
selected_departments = st.sidebar.multiselect(
    "Departments", department_names, default=department_names
)
threshold = st.sidebar.slider("High occupancy alert (%)", 50, 100, 90, 1)
st.sidebar.caption("KPIs follow the definitions in the project documentation.")

dep_map = departments.set_index("department_name")["department_id"].to_dict()
selected_ids = [dep_map[n] for n in selected_departments if n in dep_map]

d = daily[
    daily["date"].between(start_date, end_date)
    & daily["department_id"].isin(selected_ids)
].copy()

# Hospital-wide figures should use one row per date; department selection
# controls which departments contribute to the selected view.
if d.empty:
    st.warning("No records match these filters. Select at least one department and a valid date range.")
    st.stop()

hospital = (
    d.groupby("date", as_index=False)
    .agg(occupied_beds=("occupied_beds", "sum"),
         bed_capacity=("bed_capacity", "sum"))
)
hospital["occupancy_pct"] = (
    hospital["occupied_beds"] / hospital["bed_capacity"].replace(0, pd.NA) * 100
).round(1)

f = fact[
    fact["department_id"].isin(selected_ids)
    & fact["admit_date"].between(start_date, end_date)
    & (fact["is_valid"] == "Y")
    & (fact["status"] == "DISCHARGED")
].copy()

total_bed_days = d["bed_capacity"].sum()
occupied_bed_days = d["occupied_beds"].sum()
overall_occupancy = (occupied_bed_days / total_bed_days * 100) if total_bed_days else 0
avg_los = f["los_days"].mean() if not f.empty else 0
same_day_pct = (f["same_day_flag"].eq("Y").mean() * 100) if not f.empty else 0
high_days = int((hospital["occupancy_pct"] >= threshold).sum())

# KPI row
st.subheader("Key Performance Indicators")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Bed occupancy", f"{overall_occupancy:.1f}%")
k2.metric("Average LOS", f"{avg_los:.2f} days")
k3.metric("Occupied bed-days", f"{occupied_bed_days:,.0f}")
k4.metric("Completed admissions", f"{len(f):,}")
k5.metric(f"High-load days (≥{threshold}%)", f"{high_days:,}")

st.divider()

# Main charts
left, right = st.columns([1.6, 1])
with left:
    st.subheader("Daily occupancy trend")
    fig = px.line(
        hospital, x="date", y="occupancy_pct",
        markers=False, labels={"date": "Date", "occupancy_pct": "Occupancy (%)"},
        template="plotly_white",
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e05a47",
                  annotation_text=f"Alert threshold {threshold}%")
    fig.update_layout(height=370, margin=dict(l=10, r=10, t=25, b=10),
                      yaxis_ticksuffix="%", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Occupancy by department")
    dep_summary = (
        d.groupby("department_name", as_index=False)
        .agg(occupied_bed_days=("occupied_beds", "sum"),
             available_bed_days=("bed_capacity", "sum"))
    )
    dep_summary["occupancy_pct"] = (
        dep_summary["occupied_bed_days"] / dep_summary["available_bed_days"] * 100
    ).round(1)
    dep_summary = dep_summary.sort_values("occupancy_pct", ascending=True)
    fig_dep = px.bar(
        dep_summary, x="occupancy_pct", y="department_name", orientation="h",
        text="occupancy_pct", labels={"occupancy_pct": "Occupancy (%)",
                                      "department_name": "Department"},
        template="plotly_white",
    )
    fig_dep.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_dep.update_layout(height=370, margin=dict(l=10, r=35, t=25, b=10),
                          xaxis_ticksuffix="%")
    st.plotly_chart(fig_dep, use_container_width=True)

# Additional analysis
left2, right2 = st.columns(2)
with left2:
    st.subheader("Length of stay distribution")
    if f.empty:
        st.info("No completed valid admissions in this selection.")
    else:
        fig_los = px.histogram(
            f, x="los_days", nbins=25,
            labels={"los_days": "Length of stay (days)", "count": "Admissions"},
            template="plotly_white",
        )
        fig_los.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig_los, use_container_width=True)

with right2:
    st.subheader("Monthly occupancy")
    monthly = hospital.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly = monthly.groupby("month", as_index=False).agg(
        occupied_bed_days=("occupied_beds", "sum"),
        available_bed_days=("bed_capacity", "sum"),
    )
    monthly["occupancy_pct"] = (
        monthly["occupied_bed_days"] / monthly["available_bed_days"] * 100
    ).round(1)
    fig_month = px.bar(
        monthly, x="month", y="occupancy_pct", text="occupancy_pct",
        labels={"month": "Month", "occupancy_pct": "Occupancy (%)"},
        template="plotly_white",
    )
    fig_month.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_month.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10),
                            yaxis_ticksuffix="%")
    st.plotly_chart(fig_month, use_container_width=True)

# Peak load days and department details
st.subheader("Peak occupancy days")
peak = hospital.sort_values("occupancy_pct", ascending=False).head(10).copy()
peak["date"] = peak["date"].dt.strftime("%d %b %Y")
peak = peak.rename(columns={
    "date": "Date", "occupied_beds": "Occupied beds",
    "bed_capacity": "Bed capacity", "occupancy_pct": "Occupancy (%)"
})
st.dataframe(peak, use_container_width=True, hide_index=True)

st.subheader("Department performance")
st.dataframe(
    dep_summary.sort_values("occupancy_pct", ascending=False).rename(columns={
        "department_name": "Department",
        "occupied_bed_days": "Occupied bed-days",
        "available_bed_days": "Available bed-days",
        "occupancy_pct": "Occupancy (%)",
    }),
    use_container_width=True, hide_index=True
)

with st.expander("View filtered admission records"):
    cols = [c for c in [
        "admission_id", "patient_id", "department_id", "admit_date",
        "discharge_date", "age_at_admission", "los_days", "status",
        "same_day_flag", "dq_flag"
    ] if c in f.columns]
    st.dataframe(f[cols], use_container_width=True, hide_index=True)
    csv = f[cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered admissions CSV", csv,
                       "filtered_admissions.csv", "text/csv")

st.caption(
    "Data source: project processed CSV files. Occupancy is calculated from occupied bed-days / "
    "available bed-days. LOS is the mean of valid completed admissions; same-day discharge has LOS = 0."
)

