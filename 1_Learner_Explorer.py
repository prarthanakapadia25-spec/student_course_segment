import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import generate_data
from src.database import is_demo_mode, load_tables_from_db
from src.feature_engineering import build_learner_features
from src.clustering import run_kmeans, SEGMENT_COLORS, SEGMENT_DESCRIPTIONS
from src.recommendations import get_similar_learners

def hex_to_rgba(hex_color, alpha=0.13):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config(page_title="Learner Explorer — EduPro", page_icon="👤", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F8FAFC; }
.page-header {
    background: linear-gradient(135deg, #312E81 0%, #7C3AED 100%);
    border-radius: 16px; padding: 28px 32px; color: white; margin-bottom: 28px;
}
.page-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
.page-subtitle { font-size: 0.95rem; opacity: 0.85; margin-top: 6px; }
.profile-card {
    background: white; border-radius: 12px; padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 16px;
}
.stat-box {
    background: #F8FAFC; border-radius: 8px; padding: 12px 16px;
    text-align: center; margin: 4px;
}
.stat-val { font-size: 1.4rem; font-weight: 700; color: #1E293B; }
.stat-lbl { font-size: 0.75rem; color: #64748B; margin-top: 2px; }
.badge {
    display: inline-block; border-radius: 20px; padding: 4px 14px;
    font-size: 0.82rem; font-weight: 600; margin: 2px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_all_data():
    if not is_demo_mode():
        u, c, t = load_tables_from_db()
        if u is not None:
            return u, c, t, False
    u, c, t = generate_data(n_users=300)
    return u, c, t, True


@st.cache_data(show_spinner=False)
def get_features(_u, _c, _t):
    u = pd.DataFrame(_u)
    c = pd.DataFrame(_c)
    t = pd.DataFrame(_t)
    f = build_learner_features(u, c, t)
    f, _, _, sil, _ = run_kmeans(f, n_clusters=4)
    return f


users_df, courses_df, txns_df, demo = load_all_data()
features_df = get_features(users_df.to_dict(), courses_df.to_dict(), txns_df.to_dict())

st.markdown("""
<div class="page-header">
    <div class="page-title">👤 Learner Profile Explorer</div>
    <div class="page-subtitle">Deep-dive into individual learner profiles, behavior patterns, and segment assignments</div>
</div>
""", unsafe_allow_html=True)

# ---------- Learner Selector ----------
col_sel, col_info = st.columns([1, 3])

with col_sel:
    st.markdown("### Select Learner")
    user_ids = sorted(features_df["UserID"].tolist())
    selected_uid = st.selectbox("Learner ID", user_ids, label_visibility="collapsed")

    segments = ["All"] + sorted(features_df["segment"].unique().tolist())
    filter_seg = st.selectbox("Filter by Segment", segments)
    if filter_seg != "All":
        filtered_users = features_df[features_df["segment"] == filter_seg]["UserID"].tolist()
        selected_uid = st.selectbox("From Segment", sorted(filtered_users))

    st.markdown("---")
    st.markdown("**Quick Stats**")
    total = len(features_df)
    seg_counts = features_df["segment"].value_counts()
    for seg, cnt in seg_counts.items():
        color = SEGMENT_COLORS.get(seg, "#94A3B8")
        pct = cnt / total * 100
        st.markdown(
            f'<span class="badge" style="background:{color}20;color:{color}">{seg}: {cnt} ({pct:.0f}%)</span>',
            unsafe_allow_html=True,
        )

with col_info:
    user_row = features_df[features_df["UserID"] == selected_uid].iloc[0]
    segment = user_row["segment"]
    seg_color = SEGMENT_COLORS.get(segment, "#2563EB")
    user_txns = txns_df[txns_df["UserID"] == selected_uid].merge(courses_df, on="CourseID")

    # Profile header
    st.markdown(f"""
    <div class="profile-card">
        <div style="display:flex;align-items:center;gap:16px">
            <div style="width:60px;height:60px;border-radius:50%;background:linear-gradient(135deg,{seg_color},{seg_color}88);
                        display:flex;align-items:center;justify-content:center;font-size:1.6rem;color:white;font-weight:700">
                {str(selected_uid)[-1]}
            </div>
            <div>
                <div style="font-size:1.2rem;font-weight:700;color:#1E293B">Learner #{selected_uid}</div>
                <div style="font-size:0.85rem;color:#64748B">{user_row.get('Age','N/A')} years · {user_row.get('Gender','N/A')}</div>
                <div style="margin-top:6px">
                    <span class="badge" style="background:{seg_color}20;color:{seg_color};font-size:0.85rem">{segment}</span>
                </div>
            </div>
        </div>
        <div style="margin-top:14px;font-size:0.85rem;color:#64748B;line-height:1.6">
            {SEGMENT_DESCRIPTIONS.get(segment, '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{int(user_row.get("total_courses",0))}</div><div class="stat-lbl">Courses Enrolled</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><div class="stat-val">${user_row.get("avg_spending",0):.0f}</div><div class="stat-lbl">Avg Spending</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{user_row.get("avg_rating_enrolled",0):.1f}★</div><div class="stat-lbl">Avg Rating</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{int(user_row.get("diversity_score",0))}</div><div class="stat-lbl">Categories Explored</div></div>', unsafe_allow_html=True)
    with s5:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{user_row.get("learning_depth_index",0):.2f}</div><div class="stat-lbl">Depth Index</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Enrolled Courses by Category**")
        if not user_txns.empty:
            cat_dist = user_txns["CourseCategory"].value_counts().reset_index()
            cat_dist.columns = ["Category", "Count"]
            fig = px.bar(cat_dist, x="Category", y="Count", color="Category",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=240, showlegend=False, margin=dict(l=0, r=0, t=0, b=40),
                              paper_bgcolor="white", plot_bgcolor="white",
                              xaxis=dict(showgrid=False, tickangle=-30), yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No enrollment data found.")

    with ch2:
        st.markdown("**Spending Over Time**")
        if not user_txns.empty:
            user_txns["TransactionDate"] = pd.to_datetime(user_txns["TransactionDate"])
            monthly = user_txns.set_index("TransactionDate").resample("ME")["Amount"].sum().reset_index()
            fig2 = go.Figure(go.Scatter(
                x=monthly["TransactionDate"], y=monthly["Amount"],
                fill="tozeroy", line=dict(color=seg_color, width=2),
                fillcolor=hex_to_rgba(seg_color, 0.13),
            ))
            fig2.update_layout(height=240, margin=dict(l=0, r=0, t=0, b=10),
                               paper_bgcolor="white", plot_bgcolor="white",
                               xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig2, use_container_width=True)

# ---------- Enrollment History ----------
st.markdown("### Enrollment History")
if not user_txns.empty:
    display_cols = ["CourseID", "CourseName", "CourseCategory", "CourseLevel", "CourseRating", "Amount", "TransactionDate"]
    available = [c for c in display_cols if c in user_txns.columns]
    st.dataframe(user_txns[available].sort_values("TransactionDate", ascending=False), use_container_width=True, height=250)
else:
    st.info("No transactions found for this learner.")

# ---------- Similar Learners ----------
st.markdown("### Similar Learners in Platform")
similar = get_similar_learners(selected_uid, features_df, top_n=5)
if not similar.empty:
    similar["similarity"] = similar["similarity"].apply(lambda x: f"{x:.2%}")
    similar.columns = [c.replace("_", " ").title() for c in similar.columns]
    st.dataframe(similar, use_container_width=True, hide_index=True)
