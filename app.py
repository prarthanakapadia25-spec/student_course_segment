import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import generate_data
from src.database import is_demo_mode, load_tables_from_db
from src.feature_engineering import build_learner_features
from src.clustering import run_kmeans, get_cluster_summary, SEGMENT_COLORS, SEGMENT_DESCRIPTIONS

st.set_page_config(
    page_title="EduPro — Learner Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Global CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #F8FAFC; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 4px solid #2563EB;
    }
    .metric-card.green  { border-left-color: #10B981; }
    .metric-card.purple { border-left-color: #8B5CF6; }
    .metric-card.amber  { border-left-color: #F59E0B; }

    .metric-value { font-size: 2rem; font-weight: 700; color: #1E293B; line-height: 1; }
    .metric-label { font-size: 0.85rem; color: #64748B; margin-top: 4px; font-weight: 500; }
    .metric-delta { font-size: 0.75rem; color: #10B981; font-weight: 600; margin-top: 2px; }

    .segment-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-left: 5px solid #2563EB;
    }
    .segment-title { font-size: 1rem; font-weight: 600; color: #1E293B; }
    .segment-desc  { font-size: 0.82rem; color: #64748B; margin-top: 4px; line-height: 1.5; }

    .page-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 28px;
    }
    .page-title    { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .page-subtitle { font-size: 0.95rem; opacity: 0.85; margin-top: 6px; }

    .demo-badge {
        background: #FEF3C7;
        color: #92400E;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 16px;
    }

    div[data-testid="stSidebar"] { background: #1E293B; }
    div[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    div[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
    hr { border-color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)


# ---------- Data Loader (cached) ----------
@st.cache_data(show_spinner=False)
def load_all_data():
    if not is_demo_mode():
        users, courses, txns = load_tables_from_db()
        if users is not None:
            return users, courses, txns, False
    users, courses, txns = generate_data(n_users=300)
    return users, courses, txns, True


@st.cache_data(show_spinner=False)
def get_features_and_clusters(_users, _courses, _txns):
    users = pd.DataFrame(_users)
    courses = pd.DataFrame(_courses)
    txns = pd.DataFrame(_txns)
    features = build_learner_features(users, courses, txns)
    features, km, scaler, sil_score, X_scaled = run_kmeans(features, n_clusters=4)
    return features, sil_score


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎓 EduPro")
    st.markdown("**Learner Intelligence Platform**")
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    - 🏠 **Overview** ← you are here
    - 👤 Learner Explorer
    - 📊 Cluster Dashboard
    - 💡 Recommendations
    - 🔍 Segment Comparison
    """)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Powered by K-Means clustering and content-based filtering to deliver personalized learning experiences.")


# ---------- Load Data ----------
with st.spinner("Loading learner data..."):
    users_df, courses_df, txns_df, demo_mode = load_all_data()
    features_df, sil_score = get_features_and_clusters(
        users_df.to_dict(), courses_df.to_dict(), txns_df.to_dict()
    )

# Store in session
st.session_state["users_df"] = users_df
st.session_state["courses_df"] = courses_df
st.session_state["txns_df"] = txns_df
st.session_state["features_df"] = features_df
st.session_state["sil_score"] = sil_score

# ---------- Header ----------
st.markdown("""
<div class="page-header">
    <div class="page-title">🎓 EduPro Learner Intelligence Platform</div>
    <div class="page-subtitle">Student Segmentation & Personalized Course Recommendation System</div>
</div>
""", unsafe_allow_html=True)

if demo_mode:
    st.markdown('<div class="demo-badge">⚡ Demo Mode — Set DATABASE_URL in .env for live data</div>', unsafe_allow_html=True)

# ---------- KPI Cards ----------
total_learners = len(users_df)
total_courses = len(courses_df)
total_txns = len(txns_df)
total_revenue = txns_df["Amount"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{total_learners:,}</div>
        <div class="metric-label">Total Learners</div>
        <div class="metric-delta">↑ Active Profiles</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card green">
        <div class="metric-value">{total_courses}</div>
        <div class="metric-label">Courses Available</div>
        <div class="metric-delta">8 Categories</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card purple">
        <div class="metric-value">{total_txns:,}</div>
        <div class="metric-label">Total Enrollments</div>
        <div class="metric-delta">↑ Growing</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card amber">
        <div class="metric-value">${total_revenue:,.0f}</div>
        <div class="metric-label">Platform Revenue</div>
        <div class="metric-delta">↑ Increasing</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{sil_score:.2f}</div>
        <div class="metric-label">Silhouette Score</div>
        <div class="metric-delta">Cluster Quality</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Row 1: Segment Distribution + Enrollment Trend ----------
col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.markdown("#### Learner Segments")
    seg_counts = features_df["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    colors = [SEGMENT_COLORS.get(s, "#94A3B8") for s in seg_counts["Segment"]]
    fig_pie = go.Figure(go.Pie(
        labels=seg_counts["Segment"],
        values=seg_counts["Count"],
        hole=0.55,
        marker_colors=colors,
        textinfo="percent+label",
        textfont_size=13,
    ))
    fig_pie.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False, paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.markdown("#### Monthly Enrollment Trend")
    txns_trend = txns_df.copy()
    txns_trend["TransactionDate"] = pd.to_datetime(txns_trend["TransactionDate"])
    txns_trend["Month"] = txns_trend["TransactionDate"].dt.to_period("M").astype(str)
    monthly = txns_trend.groupby("Month").agg(
        Enrollments=("TransactionID", "count"),
        Revenue=("Amount", "sum"),
    ).reset_index()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly["Month"], y=monthly["Enrollments"],
        fill="tozeroy", line=dict(color="#2563EB", width=2.5),
        fillcolor="rgba(37,99,235,0.1)", name="Enrollments",
    ))
    fig_trend.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=False, tickangle=-45, tickfont_size=10),
        yaxis=dict(gridcolor="#F1F5F9"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ---------- Row 2: Category Distribution + Age Distribution ----------
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Top Course Categories")
    merged = txns_df.merge(courses_df, on="CourseID")
    cat_pop = merged["CourseCategory"].value_counts().reset_index()
    cat_pop.columns = ["Category", "Enrollments"]
    fig_bar = px.bar(
        cat_pop, x="Enrollments", y="Category",
        orientation="h",
        color="Enrollments",
        color_continuous_scale=["#DBEAFE", "#2563EB"],
    )
    fig_bar.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    st.markdown("#### Learner Age Distribution")
    fig_hist = px.histogram(
        users_df, x="Age", nbins=20,
        color_discrete_sequence=["#8B5CF6"],
    )
    fig_hist.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        bargap=0.05,
        xaxis=dict(showgrid=False, title="Age"),
        yaxis=dict(gridcolor="#F1F5F9", title="Number of Learners"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------- Segment Cards ----------
st.markdown("#### Segment Overview")
cluster_summary = get_cluster_summary(features_df)
cols = st.columns(4)
for i, row in cluster_summary.iterrows():
    seg = row["segment"]
    color = SEGMENT_COLORS.get(seg, "#2563EB")
    desc = SEGMENT_DESCRIPTIONS.get(seg, "")
    with cols[i % 4]:
        st.markdown(f"""
        <div class="segment-card" style="border-left-color:{color}">
            <div class="segment-title">{seg}</div>
            <div style="font-size:1.4rem;font-weight:700;color:{color};margin:6px 0">{int(row['learner_count'])}</div>
            <div style="font-size:0.78rem;color:#64748B">learners &nbsp;|&nbsp; avg {row['avg_courses']:.1f} courses</div>
            <div class="segment-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:0.8rem;'>EduPro Learner Intelligence Platform · Powered by K-Means Clustering</div>",
    unsafe_allow_html=True,
)
