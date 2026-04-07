import pandas as pd
import numpy as np


def get_recommendations(user_id, features_df, courses_df, transactions_df, top_n=8):
    user_row = features_df[features_df["UserID"] == user_id]
    if user_row.empty:
        return pd.DataFrame()

    segment = user_row["segment"].values[0]
    preferred_cat = user_row["preferred_category"].values[0] if "preferred_category" in user_row.columns else None
    preferred_level = user_row["preferred_level"].values[0] if "preferred_level" in user_row.columns else None

    # Courses already enrolled
    enrolled_ids = transactions_df[transactions_df["UserID"] == user_id]["CourseID"].tolist()

    # All segment peers
    segment_peers = features_df[features_df["segment"] == segment]["UserID"].tolist()
    segment_peers = [uid for uid in segment_peers if uid != user_id]

    # Courses popular in the segment (not yet enrolled)
    peer_txns = transactions_df[transactions_df["UserID"].isin(segment_peers)]
    course_popularity = (
        peer_txns.groupby("CourseID")
        .agg(peer_enrollments=("UserID", "nunique"))
        .reset_index()
    )

    # Merge with course details
    recs = courses_df.merge(course_popularity, on="CourseID", how="left").fillna({"peer_enrollments": 0})

    # Remove already enrolled
    recs = recs[~recs["CourseID"].isin(enrolled_ids)]

    # Score = weighted: 60% rating + 40% peer popularity (normalized)
    max_pop = recs["peer_enrollments"].max() if recs["peer_enrollments"].max() > 0 else 1
    recs["popularity_norm"] = recs["peer_enrollments"] / max_pop
    recs["relevance_score"] = (0.60 * recs["CourseRating"] / 5.0) + (0.40 * recs["popularity_norm"])

    # Boost preferred category and level
    if preferred_cat:
        recs.loc[recs["CourseCategory"] == preferred_cat, "relevance_score"] += 0.15
    if preferred_level:
        recs.loc[recs["CourseLevel"] == preferred_level, "relevance_score"] += 0.10

    recs = recs.sort_values("relevance_score", ascending=False)
    return recs.head(top_n).reset_index(drop=True)


def get_segment_popular_courses(segment, features_df, courses_df, transactions_df, top_n=10):
    segment_users = features_df[features_df["segment"] == segment]["UserID"].tolist()
    seg_txns = transactions_df[transactions_df["UserID"].isin(segment_users)]

    popularity = (
        seg_txns.groupby("CourseID")
        .agg(enrollments=("UserID", "nunique"), avg_amount=("Amount", "mean"))
        .reset_index()
    )
    result = courses_df.merge(popularity, on="CourseID", how="left").fillna(0)
    result = result.sort_values(["enrollments", "CourseRating"], ascending=False)
    return result.head(top_n).reset_index(drop=True)


def get_similar_learners(user_id, features_df, top_n=5):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    from src.feature_engineering import NUMERIC_FEATURES

    if user_id not in features_df["UserID"].values:
        return pd.DataFrame()

    X = features_df[NUMERIC_FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    user_idx = features_df[features_df["UserID"] == user_id].index[0]
    user_vec = X_scaled[user_idx].reshape(1, -1)

    similarities = cosine_similarity(user_vec, X_scaled)[0]
    features_df = features_df.copy()
    features_df["similarity"] = similarities
    similar = features_df[features_df["UserID"] != user_id].sort_values("similarity", ascending=False)
    return similar.head(top_n)[["UserID", "Age", "Gender", "segment", "total_courses", "avg_spending", "similarity"]]
