import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import generate_data
from src.database import is_demo_mode, load_tables_from_db
from src.feature_engineering import build_learner_features, NUMERIC_FEATURES
from src.clustering import run_kmeans, get_cluster_summary, SEGMENT_COLORS, SEGMENT_DESCRIPTIONS

st.set_page_config(page_title="Segment Comparison — EduPro", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F8FAFC; }
.page-header {
    background: linear-gradient(135deg, #1E1B4B 0%, #4F46E5 100%);
    border-radius: 16px; padding: 28px 32px; color: white; margin-bottom: 28px;
}
.page-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
.page-subtitle { font-size: 0.95rem; opacity: 0.85; margin-top: 6px; }
.seg-header {
    border-radius: 12px; padding: 16px 20px; color: white; margin-bottom: 14px; text-align: center;
}
.seg-metric { font-size: 1.6rem; font-weight: 700; }
.seg-label  { font-size: 0.78rem; opacity: 0.9; }
.insight-box {
    background: white; border-radius: 10px; padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07); margin: 6px 0;
    border-left: 4px solid #4F46E5;
}
.insight-text { font-size: 0.85rem; color: #334155; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_all_data():
    if not is_demo_mode():
        u, c, t = load_tables_from_db()
        if u is not None:
            return u, c, t
    return generate_data(n_users=300)


@st.cache_data(show_spinner=False)
def get_features(_u, _c, _t):
    u = pd.DataFrame(_u)
    c = pd.DataFrame(_c)
    t = pd.DataFrame(_t)
    f = build_learner_features(u, c, t)
    f, _, _, sil, _ = run_kmeans(f, n_clusters=4)
    return f


users_df, courses_df, txns_df = load_all_data()
features_df = get_features(users_df.to_dict(), courses_df.to_dict(), txns_df.to_dict())
merged = txns_df.merge(courses_df, on="CourseID").merge(features_df[["UserID", "segment"]], on="UserID")

st.markdown("""
<div class="page-header">
    <div class="page-title">🔍 Segment Comparison Panels</div>
    <div class="page-subtitle">Side-by-side analysis of learner segments — behavioral patterns, preferences, and engagement metrics</div>
</div>
""", unsafe_allow_html=True)

segments = sorted(features_df["segment"].unique().tolist())

# ---------- Select Segments to Compare ----------
sel_segs = st.multiselect("Select Segments to Compare", segments, default=segments)
if len(sel_segs) < 2:
    st.warning("Select at least 2 segments to compare.")
    st.stop()

filtered = features_df[features_df["segment"].isin(sel_segs)]
filtered_merged = merged[merged["segment"].isin(sel_segs)]

# ---------- Summary Headers ----------
cols = st.columns(len(sel_segs))
for i, seg in enumerate(sel_segs):
    seg_df = filtered[filtered["segment"] == seg]
    color = SEGMENT_COLORS.get(seg, "#4F46E5")
    with cols[i]:
        count = len(seg_df)
        avg_courses = seg_df["total_courses"].mean()
        avg_spend = seg_df["avg_spending"].mean()
        st.markdown(f"""
        <div class="seg-header" style="background:linear-gradient(135deg,{color},{color}cc)">
            <div style="font-size:1.1rem;font-weight:700;margin-bottom:8px">{seg}</div>
            <div class="seg-metric">{count}</div>
            <div class="seg-label">Learners</div>
            <div style="margin-top:10px;display:flex;justify-content:space-around">
                <div><div class="seg-metric" style="font-size:1.1rem">{avg_courses:.1f}</div><div class="seg-label">Avg Courses</div></div>
                <div><div class="seg-metric" style="font-size:1.1rem">${avg_spend:.0f}</div><div class="seg-label">Avg Spend</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box"><div class="insight-text">{SEGMENT_DESCRIPTIONS.get(seg,"")}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Tabs ----------
t1, t2, t3, t4 = st.tabs(["📊 Key Metrics", "📚 Course Preferences", "💰 Spending Analysis", "📅 Engagement Over Time"])

# ===== TAB 1 =====
with t1:
    st.markdown("#### Feature Comparison Across Segments")
    metric_choice = st.selectbox("Select Metric", NUMERIC_FEATURES,
                                 format_func=lambda x: x.replace("_", " ").title())

    seg_metric = filtered.groupby("segment")[metric_choice].mean().reset_index()
    colors = [SEGMENT_COLORS.get(s, "#888") for s in seg_metric["segment"]]
    fig = go.Figure(go.Bar(
        x=seg_metric["segment"],
        y=seg_metric[metric_choice],
        marker_color=colors,
        text=seg_metric[metric_choice].round(2),
        textposition="outside",
    ))
    fig.update_layout(height=360, paper_bgcolor="white", plot_bgcolor="white",
                      margin=dict(l=0, r=0, t=20, b=0),
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F1F5F9"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Box Plot — Distribution Comparison")
    fig_box = px.box(filtered[filtered["segment"].isin(sel_segs)],
                     x="segment", y=metric_choice, color="segment",
                     color_discrete_map=SEGMENT_COLORS)
    fig_box.update_layout(height=380, showlegend=False,
                          paper_bgcolor="white", plot_bgcolor="white",
                          margin=dict(l=0, r=0, t=10, b=0),
                          xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F1F5F9"))
    st.plotly_chart(fig_box, use_container_width=True)

# ===== TAB 2 =====
with t2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Preferred Course Category")
        cat_pref = (
            filtered.groupby(["segment", "preferred_category"])
            .size().reset_index(name="count")
        )
        fig_cat = px.bar(cat_pref, x="preferred_category", y="count", color="segment",
                         barmode="group", color_discrete_map=SEGMENT_COLORS)
        fig_cat.update_layout(height=340, paper_bgcolor="white", plot_bgcolor="white",
                               margin=dict(l=0, r=0, t=10, b=40),
                               xaxis=dict(tickangle=-30, showgrid=False),
                               yaxis=dict(gridcolor="#F1F5F9"),
                               legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        st.markdown("#### Course Level Preference")
        lvl_pref = (
            filtered.groupby(["segment", "preferred_level"])
            .size().reset_index(name="count")
        )
        fig_lvl = px.bar(lvl_pref, x="preferred_level", y="count", color="segment",
                         barmode="group", color_discrete_map=SEGMENT_COLORS,
                         category_orders={"preferred_level": ["Beginner", "Intermediate", "Advanced"]})
        fig_lvl.update_layout(height=340, paper_bgcolor="white", plot_bgcolor="white",
                               margin=dict(l=0, r=0, t=10, b=40),
                               xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F1F5F9"),
                               legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_lvl, use_container_width=True)

    st.markdown("#### Top Courses Enrolled Per Segment")
    top_courses_data = []
    for seg in sel_segs:
        seg_merged = filtered_merged[filtered_merged["segment"] == seg]
        top = seg_merged["CourseCategory"].value_counts().head(5).reset_index()
        top.columns = ["Category", "Enrollments"]
        top["Segment"] = seg
        top_courses_data.append(top)
    if top_courses_data:
        all_top = pd.concat(top_courses_data)
        fig_top = px.bar(all_top, x="Category", y="Enrollments", color="Segment",
                         barmode="group", color_discrete_map=SEGMENT_COLORS)
        fig_top.update_layout(height=320, paper_bgcolor="white", plot_bgcolor="white",
                               margin=dict(l=0, r=0, t=10, b=40),
                               xaxis=dict(tickangle=-20, showgrid=False),
                               yaxis=dict(gridcolor="#F1F5F9"),
                               legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_top, use_container_width=True)

# ===== TAB 3 =====
with t3:
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### Average Spending Distribution")
        fig_spend = px.violin(filtered[filtered["segment"].isin(sel_segs)],
                              x="segment", y="avg_spending", color="segment",
                              color_discrete_map=SEGMENT_COLORS, box=True)
        fig_spend.update_layout(height=360, showlegend=False,
                                paper_bgcolor="white", plot_bgcolor="white",
                                margin=dict(l=0, r=0, t=10, b=0),
                                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F1F5F9"))
        st.plotly_chart(fig_spend, use_container_width=True)

    with col_s2:
        st.markdown("#### Revenue Contribution by Segment")
        seg_rev = merged[merged["segment"].isin(sel_segs)].groupby("segment")["Amount"].sum().reset_index()
        seg_rev.columns = ["Segment", "Revenue"]
        colors_rev = [SEGMENT_COLORS.get(s, "#888") for s in seg_rev["Segment"]]
        fig_rev = go.Figure(go.Pie(
            labels=seg_rev["Segment"], values=seg_rev["Revenue"],
            hole=0.5, marker_colors=colors_rev,
            textinfo="percent+label",
        ))
        fig_rev.update_layout(height=360, paper_bgcolor="white",
                              margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("#### Spending Summary Table")
    spend_summary = (
        filtered[filtered["segment"].isin(sel_segs)]
        .groupby("segment")["avg_spending"]
        .agg(["mean", "median", "min", "max", "std"])
        .round(2)
        .reset_index()
    )
    spend_summary.columns = ["Segment", "Mean ($)", "Median ($)", "Min ($)", "Max ($)", "Std Dev"]
    st.dataframe(spend_summary, use_container_width=True, hide_index=True)

# ===== TAB 4 =====
with t4:
    st.markdown("#### Monthly Enrollment Trend by Segment")
    trend_data = filtered_merged.copy()
    trend_data["TransactionDate"] = pd.to_datetime(trend_data["TransactionDate"])
    trend_data["Month"] = trend_data["TransactionDate"].dt.to_period("M").astype(str)

    monthly_seg = (
        trend_data.groupby(["Month", "segment"])["TransactionID"]
        .count().reset_index(name="Enrollments")
    )
    fig_trend = px.line(monthly_seg, x="Month", y="Enrollments", color="segment",
                        color_discrete_map=SEGMENT_COLORS, markers=True)
    fig_trend.update_layout(
        height=400, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=60),
        xaxis=dict(tickangle=-45, showgrid=False, tickfont_size=9),
        yaxis=dict(gridcolor="#F1F5F9"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("#### Gender Distribution by Segment")
    gender_seg = filtered.groupby(["segment", "Gender"]).size().reset_index(name="Count")
    fig_gender = px.bar(gender_seg, x="segment", y="Count", color="Gender",
                        barmode="stack",
                        color_discrete_sequence=["#3B82F6", "#EC4899", "#94A3B8"])
    fig_gender.update_layout(
        height=340, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F1F5F9"),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_gender, use_container_width=True)
