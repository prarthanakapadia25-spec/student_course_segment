import pandas as pd
import numpy as np


def build_learner_features(users_df, courses_df, transactions_df):
    merged = transactions_df.merge(courses_df, on="CourseID").merge(users_df, on="UserID")

    # ---- Engagement Features ----
    total_courses = transactions_df.groupby("UserID")["CourseID"].nunique().rename("total_courses")

    avg_per_category = (
        merged.groupby(["UserID", "CourseCategory"])["CourseID"]
        .nunique()
        .reset_index()
        .groupby("UserID")["CourseID"]
        .mean()
        .rename("avg_courses_per_category")
    )

    merged["TransactionDate"] = pd.to_datetime(merged["TransactionDate"])
    date_span = merged.groupby("UserID")["TransactionDate"].agg(lambda x: (x.max() - x.min()).days + 1)
    enrollment_frequency = (total_courses / date_span.replace(0, 1)).rename("enrollment_frequency")

    # ---- Preference Features ----
    preferred_category = (
        merged.groupby(["UserID", "CourseCategory"])["CourseID"]
        .count()
        .reset_index()
        .sort_values("CourseID", ascending=False)
        .drop_duplicates("UserID")
        .set_index("UserID")["CourseCategory"]
        .rename("preferred_category")
    )

    preferred_level = (
        merged.groupby(["UserID", "CourseLevel"])["CourseID"]
        .count()
        .reset_index()
        .sort_values("CourseID", ascending=False)
        .drop_duplicates("UserID")
        .set_index("UserID")["CourseLevel"]
        .rename("preferred_level")
    )

    avg_rating = merged.groupby("UserID")["CourseRating"].mean().rename("avg_rating_enrolled")

    # ---- Behavioral Features ----
    avg_spending = transactions_df.groupby("UserID")["Amount"].mean().rename("avg_spending")

    diversity_score = (
        merged.groupby("UserID")["CourseCategory"].nunique().rename("diversity_score")
    )

    beginner_count = (
        merged[merged["CourseLevel"] == "Beginner"]
        .groupby("UserID")["CourseID"]
        .count()
        .rename("beginner_count")
    )
    advanced_count = (
        merged[merged["CourseLevel"] == "Advanced"]
        .groupby("UserID")["CourseID"]
        .count()
        .rename("advanced_count")
    )
    depth_df = pd.concat([beginner_count, advanced_count], axis=1).fillna(0)
    depth_df["learning_depth_index"] = depth_df["advanced_count"] / (depth_df["beginner_count"] + depth_df["advanced_count"] + 1)

    # ---- Combine ----
    features = pd.concat(
        [total_courses, avg_per_category, enrollment_frequency,
         preferred_category, preferred_level, avg_rating,
         avg_spending, diversity_score, depth_df["learning_depth_index"]],
        axis=1,
    ).reset_index()

    features = features.merge(users_df, on="UserID", how="left")
    features = features.fillna(0)

    # Encode categorical columns
    level_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    features["preferred_level_encoded"] = features["preferred_level"].map(level_map).fillna(0)

    cat_codes = {cat: i for i, cat in enumerate(sorted(features["preferred_category"].unique()))}
    features["preferred_category_encoded"] = features["preferred_category"].map(cat_codes).fillna(0)

    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    features["gender_encoded"] = features["Gender"].map(gender_map).fillna(0)

    return features


NUMERIC_FEATURES = [
    "total_courses",
    "avg_courses_per_category",
    "enrollment_frequency",
    "avg_rating_enrolled",
    "avg_spending",
    "diversity_score",
    "learning_depth_index",
    "preferred_level_encoded",
]
