import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.data_generator import generate_data
from src.database import is_demo_mode, load_tables_from_db
from src.feature_engineering import build_learner_features, NUMERIC_FEATURES
from src.clustering import run_kmeans, get_cluster_summary, find_optimal_k, SEGMENT_COLORS
from sklearn.preprocessing import StandardScaler

def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config(page_title="Cluster Dashboard — EduPro", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F8FAFC; }
.page-header {
    background: linear-gradient(135deg, #064E3B 0%, #10B981 100%);
    border-radius: 16px; padding: 28px 32px; color: white; margin-bottom: 28px;
}
.page-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
.page-subtitle { font-size: 0.95rem; opacity: 0.85; margin-top: 6px; }
.info-card {
    background: white; border-radius: 12px; padding: 18px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07); margin-bottom: 12px;
}
.metric-mini { font-size: 1.5rem; font-weight: 700; color: #1E293B; }
.metric-mini-label { font-size: 0.8rem; color: #64748B; }
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
def get_features_clusters(_u, _c, _t):
    u = pd.DataFrame(_u)
    c = pd.DataFrame(_c)
    t = pd.DataFrame(_t)
    f = build_learner_features(u, c, t)
    f, km, scaler, sil, X_scaled = run_kmeans(f, n_clusters=4)
    return f, sil, X_scaled


users_df, courses_df, txns_df = load_all_data()
features_df, sil_score, X_scaled = get_features_clusters(
    users_df.to_dict(), courses_df.to_dict(), txns_df.to_dict()
)

st.markdown("""
<div class="page-header">
    <div class="page-title">📊 Cluster Visualization Dashboard</div>
    <div class="page-subtitle">K-Means clustering analysis — visualize and explore learner segments</div>
</div>
""", unsafe_allow_html=True)

# ---------- Silhouette + Inertia ----------
k_vals, inertias, silhouettes = find_optimal_k(X_scaled)

tab1, tab2, tab3 = st.tabs(["🔵 Cluster Scatter", "📈 Elbow & Silhouette", "🗂 Feature Profiles"])

# ===== TAB 1: Scatter =====
with tab1:
    st.markdown("#### Learner Cluster Map (PCA 2D Projection)")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    plot_df = features_df.copy()
    plot_df["PCA_1"] = coords[:, 0]
    plot_df["PCA_2"] = coords[:, 1]

    color_map = SEGMENT_COLORS
    fig_scatter = px.scatter(
        plot_df, x="PCA_1", y="PCA_2",
        color="segment",
        color_discrete_map=color_map,
        hover_data={"UserID": True, "total_courses": True, "avg_spending": ":.2f", "diversity_score": True},
        labels={"PCA_1": "Principal Component 1", "PCA_2": "Principal Component 2"},
        opacity=0.75,
    )
    fig_scatter.update_traces(marker_size=7)
    fig_scatter.update_layout(
        height=480, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    exp_var = pca.explained_variance_ratio_
    st.caption(f"PCA explains {exp_var[0]*100:.1f}% + {exp_var[1]*100:.1f}% = {sum(exp_var)*100:.1f}% of variance. Silhouette Score: **{sil_score:.3f}**")

    # Cluster summary table
    st.markdown("#### Cluster Summary Statistics")
    summary = get_cluster_summary(features_df)
    summary.columns = ["Segment", "Learners", "Avg Courses", "Avg Spending ($)", "Avg Rating", "Avg Diversity", "Avg Depth Index"]
    for col in ["Avg Courses", "Avg Spending ($)", "Avg Rating", "Avg Diversity", "Avg Depth Index"]:
        summary[col] = summary[col].round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ===== TAB 2: Elbow & Silhouette =====
with tab2:
    col_e, col_s = st.columns(2)

    with col_e:
        st.markdown("#### Elbow Method — Inertia vs K")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_vals, y=inertias,
            mode="lines+markers",
            line=dict(color="#10B981", width=2.5),
            marker=dict(size=8, color="#10B981"),
        ))
        fig_elbow.add_vline(x=4, line_dash="dash", line_color="#EF4444",
                            annotation_text="Optimal K=4", annotation_position="top right")
        fig_elbow.update_layout(
            height=340, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=40),
            xaxis=dict(title="Number of Clusters (K)", showgrid=False),
            yaxis=dict(title="Inertia", gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.caption("The 'elbow' at K=4 suggests 4 is the optimal number of clusters.")

    with col_s:
        st.markdown("#### Silhouette Score vs K")
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=k_vals, y=silhouettes,
            marker_color=["#DBEAFE" if k != 4 else "#2563EB" for k in k_vals],
        ))
        fig_sil.update_layout(
            height=340, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=40),
            xaxis=dict(title="Number of Clusters (K)", showgrid=False),
            yaxis=dict(title="Silhouette Score", gridcolor="#F1F5F9", range=[0, 0.8]),
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        st.caption(f"Current silhouette score at K=4: **{sil_score:.3f}** (higher is better, max=1.0).")

# ===== TAB 3: Feature Profiles =====
with tab3:
    st.markdown("#### Average Feature Values by Segment")
    profile_df = features_df.groupby("segment")[NUMERIC_FEATURES].mean().reset_index()

    feature_labels = {
        "total_courses": "Total Courses",
        "avg_courses_per_category": "Avg Courses/Category",
        "enrollment_frequency": "Enrollment Frequency",
        "avg_rating_enrolled": "Avg Rating",
        "avg_spending": "Avg Spending",
        "diversity_score": "Diversity Score",
        "learning_depth_index": "Depth Index",
        "preferred_level_encoded": "Level (0=Beg, 2=Adv)",
    }

    feat_choice = st.selectbox(
        "Select Feature",
        NUMERIC_FEATURES,
        format_func=lambda x: feature_labels.get(x, x),
    )

    fig_feat = px.bar(
        profile_df, x="segment", y=feat_choice,
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        text_auto=".2f",
    )
    fig_feat.update_layout(
        height=360, showlegend=False,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(title="Segment", showgrid=False),
        yaxis=dict(title=feature_labels.get(feat_choice, feat_choice), gridcolor="#F1F5F9"),
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("#### Radar Chart — Segment Profiles")
    from sklearn.preprocessing import MinMaxScaler
    radar_features = ["total_courses", "avg_spending", "diversity_score", "learning_depth_index", "avg_rating_enrolled"]
    radar_labels = ["Total Courses", "Avg Spending", "Diversity", "Depth Index", "Avg Rating"]

    mms = MinMaxScaler()
    norm_vals = mms.fit_transform(profile_df[radar_features])
    norm_df = pd.DataFrame(norm_vals, columns=radar_features)
    norm_df["segment"] = profile_df["segment"].values

    fig_radar = go.Figure()
    for _, row in norm_df.iterrows():
        seg = row["segment"]
        vals = [row[f] for f in radar_features]
        vals += [vals[0]]
        lbl = radar_labels + [radar_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=lbl, fill="toself",
            name=seg,
            line_color=SEGMENT_COLORS.get(seg, "#888"),
            fillcolor=hex_to_rgba(SEGMENT_COLORS.get(seg, "#888888"), 0.2),
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E2E8F0")),
        showlegend=True, height=420,
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
