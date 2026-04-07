import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from src.feature_engineering import NUMERIC_FEATURES

SEGMENT_NAMES = {
    0: "Explorer",
    1: "Specialist",
    2: "Career-Focused",
    3: "Casual Learner",
}

SEGMENT_COLORS = {
    "Explorer": "#3B82F6",
    "Specialist": "#8B5CF6",
    "Career-Focused": "#10B981",
    "Casual Learner": "#F59E0B",
}

SEGMENT_DESCRIPTIONS = {
    "Explorer": "Curious learners who explore diverse topics across multiple categories. They enroll in many courses and enjoy discovering new subjects.",
    "Specialist": "Deep-focus learners who master specific domains. They prefer advanced content in their preferred category.",
    "Career-Focused": "Goal-oriented learners pursuing certifications and career advancement. High spenders who choose bootcamps and advanced courses.",
    "Casual Learner": "Occasional learners who take a few beginner courses. Lower engagement but open to discovery.",
}


def find_optimal_k(X_scaled, k_range=range(2, 9)):
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(X_scaled, labels))
        else:
            silhouettes.append(0)
    return list(k_range), inertias, silhouettes


def run_kmeans(features_df, n_clusters=4):
    X = features_df[NUMERIC_FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)

    # Map cluster IDs to meaningful names based on characteristics
    temp = features_df.copy()
    temp["raw_cluster"] = labels

    cluster_profiles = temp.groupby("raw_cluster")[NUMERIC_FEATURES].mean()

    # Assign names: sort by diversity + total_courses for explorer,
    # learning_depth for specialist, spending for career, else casual
    sorted_by_diversity = cluster_profiles["diversity_score"].sort_values(ascending=False).index.tolist()
    sorted_by_depth = cluster_profiles["learning_depth_index"].sort_values(ascending=False).index.tolist()
    sorted_by_spending = cluster_profiles["avg_spending"].sort_values(ascending=False).index.tolist()

    assignment = {}
    used = set()

    # Career-Focused = highest spending
    for c in sorted_by_spending:
        if c not in used:
            assignment[c] = "Career-Focused"
            used.add(c)
            break

    # Specialist = highest depth index (among remaining)
    for c in sorted_by_depth:
        if c not in used:
            assignment[c] = "Specialist"
            used.add(c)
            break

    # Explorer = highest diversity (among remaining)
    for c in sorted_by_diversity:
        if c not in used:
            assignment[c] = "Explorer"
            used.add(c)
            break

    # Casual = whatever is left
    for c in cluster_profiles.index:
        if c not in used:
            assignment[c] = "Casual Learner"
            used.add(c)

    features_df = features_df.copy()
    features_df["cluster_id"] = labels
    features_df["segment"] = features_df["cluster_id"].map(assignment)

    return features_df, km, scaler, sil_score, X_scaled


def get_cluster_summary(features_df):
    summary = (
        features_df.groupby("segment")
        .agg(
            learner_count=("UserID", "count"),
            avg_courses=("total_courses", "mean"),
            avg_spending=("avg_spending", "mean"),
            avg_rating=("avg_rating_enrolled", "mean"),
            avg_diversity=("diversity_score", "mean"),
            avg_depth=("learning_depth_index", "mean"),
        )
        .reset_index()
    )
    return summary
