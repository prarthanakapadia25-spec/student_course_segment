import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_generator import generate_data
from src.database import is_demo_mode, load_tables_from_db
from src.feature_engineering import build_learner_features
from src.clustering import run_kmeans, SEGMENT_COLORS, SEGMENT_DESCRIPTIONS
from src.recommendations import get_recommendations, get_segment_popular_courses

st.set_page_config(page_title="Recommendations — EduPro", page_icon="💡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F8FAFC; }
.page-header {
    background: linear-gradient(135deg, #78350F 0%, #F59E0B 100%);
    border-radius: 16px; padding: 28px 32px; color: white; margin-bottom: 28px;
}
.page-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
.page-subtitle { font-size: 0.95rem; opacity: 0.85; margin-top: 6px; }
.course-card {
    background: white; border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07); margin-bottom: 10px;
    border-left: 4px solid #F59E0B; transition: all 0.2s;
}
.course-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
.course-name { font-size: 0.95rem; font-weight: 600; color: #1E293B; }
.course-meta { font-size: 0.78rem; color: #64748B; margin-top: 4px; }
.score-bar-bg { background:#E2E8F0; border-radius:4px; height:6px; margin-top:6px; }
.badge-sm {
    display: inline-block; border-radius: 6px; padding: 2px 10px;
    font-size: 0.72rem; font-weight: 600; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

LEVEL_COLORS = {"Beginner": "#10B981", "Intermediate": "#3B82F6", "Advanced": "#EF4444"}
TYPE_COLORS  = {"Self-Paced": "#8B5CF6", "Live Session": "#F59E0B", "Video Series": "#0EA5E9", "Bootcamp": "#EF4444"}


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
    f, _, _, _, _ = run_kmeans(f, n_clusters=4)
    return f


users_df, courses_df, txns_df = load_all_data()
features_df = get_features(users_df.to_dict(), courses_df.to_dict(), txns_df.to_dict())

st.markdown("""
<div class="page-header">
    <div class="page-title">💡 Personalized Course Recommendations</div>
    <div class="page-subtitle">Cluster-aware recommendations using content-based filtering and peer learning patterns</div>
</div>
""", unsafe_allow_html=True)

# ---------- Controls ----------
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1.5, 1.5, 1.5])

with ctrl1:
    selected_uid = st.selectbox("Select Learner", sorted(features_df["UserID"].tolist()))
with ctrl2:
    categories = ["All Categories"] + sorted(courses_df["CourseCategory"].unique().tolist())
    filter_cat = st.selectbox("Filter by Category", categories)
with ctrl3:
    levels = ["All Levels", "Beginner", "Intermediate", "Advanced"]
    filter_level = st.selectbox("Filter by Level", levels)
with ctrl4:
    top_n = st.slider("Max Recommendations", 4, 16, 8)

# Get recommendations
recs = get_recommendations(selected_uid, features_df, courses_df, txns_df, top_n=16)

# Apply filters
if filter_cat != "All Categories":
    recs = recs[recs["CourseCategory"] == filter_cat]
if filter_level != "All Levels":
    recs = recs[recs["CourseLevel"] == filter_level]
recs = recs.head(top_n)

# ---------- User Context ----------
user_row = features_df[features_df["UserID"] == selected_uid].iloc[0]
segment = user_row["segment"]
seg_color = SEGMENT_COLORS.get(segment, "#F59E0B")

ctx1, ctx2, ctx3, ctx4 = st.columns(4)
with ctx1:
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,0.07)">
        <div style="font-size:0.75rem;color:#64748B">Learner Segment</div>
        <div style="font-size:1rem;font-weight:700;color:{seg_color};margin-top:4px">{segment}</div>
    </div>
    """, unsafe_allow_html=True)
with ctx2:
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,0.07)">
        <div style="font-size:0.75rem;color:#64748B">Preferred Category</div>
        <div style="font-size:1rem;font-weight:700;color:#1E293B;margin-top:4px">{user_row.get('preferred_category','—')}</div>
    </div>
    """, unsafe_allow_html=True)
with ctx3:
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,0.07)">
        <div style="font-size:0.75rem;color:#64748B">Preferred Level</div>
        <div style="font-size:1rem;font-weight:700;color:#1E293B;margin-top:4px">{user_row.get('preferred_level','—')}</div>
    </div>
    """, unsafe_allow_html=True)
with ctx4:
    enrolled = txns_df[txns_df["UserID"] == selected_uid]["CourseID"].nunique()
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,0.07)">
        <div style="font-size:0.75rem;color:#64748B">Already Enrolled</div>
        <div style="font-size:1rem;font-weight:700;color:#1E293B;margin-top:4px">{enrolled} courses</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Recommendations ----------
col_recs, col_chart = st.columns([1.6, 1])

with col_recs:
    st.markdown(f"#### Recommended Courses ({len(recs)} results)")
    if recs.empty:
        st.info("No recommendations match the current filters. Try adjusting the filters above.")
    else:
        for _, row in recs.iterrows():
            score_pct = int(row.get("relevance_score", 0) * 100)
            lvl_col = LEVEL_COLORS.get(row["CourseLevel"], "#94A3B8")
            typ_col = TYPE_COLORS.get(row["CourseType"], "#94A3B8")
            score_bar_w = min(score_pct, 100)
            name = row.get("CourseName", f"Course #{row['CourseID']}")
            st.markdown(f"""
            <div class="course-card">
                <div class="course-name">⭐ {row['CourseRating']} &nbsp; {name}</div>
                <div class="course-meta">
                    <span class="badge-sm" style="background:{lvl_col}22;color:{lvl_col}">{row['CourseLevel']}</span>
                    <span class="badge-sm" style="background:{typ_col}22;color:{typ_col}">{row['CourseType']}</span>
                    <span class="badge-sm" style="background:#F1F5F9;color:#475569">{row['CourseCategory']}</span>
                </div>
                <div class="score-bar-bg">
                    <div style="width:{score_bar_w}%;background:{seg_color};height:6px;border-radius:4px"></div>
                </div>
                <div style="font-size:0.72rem;color:#94A3B8;margin-top:4px">Relevance Score: {score_pct}%</div>
            </div>
            """, unsafe_allow_html=True)

with col_chart:
    st.markdown("#### Category Distribution")
    if not recs.empty:
        cat_dist = recs["CourseCategory"].value_counts().reset_index()
        cat_dist.columns = ["Category", "Count"]
        fig = px.pie(cat_dist, values="Count", names="Category", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor="white", showlegend=True,
                          legend=dict(font_size=11))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Level Distribution")
    if not recs.empty:
        lvl_dist = recs["CourseLevel"].value_counts().reset_index()
        lvl_dist.columns = ["Level", "Count"]
        lvl_dist["Color"] = lvl_dist["Level"].map(LEVEL_COLORS)
        fig2 = px.bar(lvl_dist, x="Level", y="Count",
                      color="Level",
                      color_discrete_map=LEVEL_COLORS)
        fig2.update_layout(height=220, showlegend=False,
                           margin=dict(l=0, r=0, t=0, b=0),
                           paper_bgcolor="white", plot_bgcolor="white",
                           xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F1F5F9"))
        st.plotly_chart(fig2, use_container_width=True)

# ---------- Segment Popular Courses ----------
st.markdown(f"#### Top Courses in **{segment}** Segment")
seg_popular = get_segment_popular_courses(segment, features_df, courses_df, txns_df, top_n=8)
if not seg_popular.empty:
    display_cols = [c for c in ["CourseID", "CourseName", "CourseCategory", "CourseLevel", "CourseRating", "enrollments"] if c in seg_popular.columns]
    seg_popular_display = seg_popular[display_cols].copy()
    seg_popular_display.columns = [c.replace("_", " ").title() for c in seg_popular_display.columns]
    st.dataframe(seg_popular_display, use_container_width=True, hide_index=True)
