import numpy as np
import pandas as pd
from datetime import date, timedelta

CATEGORIES = ["Technology", "Business", "Data Science", "Design", "Marketing", "Health & Wellness", "Language", "Arts & Music"]
COURSE_TYPES = ["Self-Paced", "Live Session", "Video Series", "Bootcamp"]
LEVELS = ["Beginner", "Intermediate", "Advanced"]

COURSES_DATA = [
    ("Python for Beginners", "Technology", "Self-Paced", "Beginner", 4.5),
    ("Advanced Machine Learning", "Data Science", "Bootcamp", "Advanced", 4.8),
    ("Web Development Bootcamp", "Technology", "Bootcamp", "Intermediate", 4.6),
    ("Business Analytics", "Business", "Video Series", "Intermediate", 4.3),
    ("UI/UX Design Fundamentals", "Design", "Self-Paced", "Beginner", 4.4),
    ("Digital Marketing Mastery", "Marketing", "Live Session", "Intermediate", 4.2),
    ("Data Visualization with Tableau", "Data Science", "Self-Paced", "Intermediate", 4.5),
    ("Yoga & Mindfulness", "Health & Wellness", "Video Series", "Beginner", 4.7),
    ("Spanish for Beginners", "Language", "Self-Paced", "Beginner", 4.3),
    ("Guitar Masterclass", "Arts & Music", "Video Series", "Beginner", 4.6),
    ("Deep Learning with TensorFlow", "Data Science", "Bootcamp", "Advanced", 4.9),
    ("Excel for Business", "Business", "Self-Paced", "Beginner", 4.1),
    ("React.js Advanced", "Technology", "Self-Paced", "Advanced", 4.7),
    ("Brand Strategy", "Marketing", "Live Session", "Advanced", 4.4),
    ("Graphic Design Pro", "Design", "Video Series", "Intermediate", 4.5),
    ("SQL & Databases", "Technology", "Self-Paced", "Intermediate", 4.4),
    ("Project Management PMP", "Business", "Bootcamp", "Advanced", 4.6),
    ("Nutrition & Diet Science", "Health & Wellness", "Self-Paced", "Intermediate", 4.3),
    ("Mandarin Chinese", "Language", "Live Session", "Beginner", 4.5),
    ("Photography Essentials", "Arts & Music", "Video Series", "Beginner", 4.4),
    ("Cloud Computing AWS", "Technology", "Bootcamp", "Intermediate", 4.7),
    ("Financial Modeling", "Business", "Self-Paced", "Advanced", 4.5),
    ("NLP with Python", "Data Science", "Self-Paced", "Advanced", 4.8),
    ("SEO & Content Marketing", "Marketing", "Video Series", "Beginner", 4.2),
    ("Motion Graphics", "Design", "Self-Paced", "Advanced", 4.6),
    ("Cybersecurity Essentials", "Technology", "Live Session", "Intermediate", 4.5),
    ("Entrepreneurship 101", "Business", "Video Series", "Beginner", 4.3),
    ("Power BI Analytics", "Data Science", "Self-Paced", "Intermediate", 4.6),
    ("Social Media Marketing", "Marketing", "Self-Paced", "Beginner", 4.1),
    ("Watercolor Painting", "Arts & Music", "Video Series", "Beginner", 4.5),
    ("Django REST API", "Technology", "Self-Paced", "Advanced", 4.7),
    ("Investment & Stocks", "Business", "Live Session", "Intermediate", 4.4),
    ("Statistics for Data Science", "Data Science", "Self-Paced", "Intermediate", 4.6),
    ("French Language A1", "Language", "Self-Paced", "Beginner", 4.2),
    ("3D Modeling Blender", "Design", "Video Series", "Intermediate", 4.5),
    ("Fitness & Strength Training", "Health & Wellness", "Video Series", "Beginner", 4.6),
    ("Email Marketing", "Marketing", "Self-Paced", "Beginner", 4.0),
    ("Piano for Beginners", "Arts & Music", "Live Session", "Beginner", 4.7),
    ("Docker & Kubernetes", "Technology", "Bootcamp", "Advanced", 4.8),
    ("Leadership & Management", "Business", "Live Session", "Advanced", 4.5),
]


def generate_data(n_users=300, seed=42):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # --- Users ---
    user_ids = list(range(1, n_users + 1))
    ages = rng.integers(18, 55, size=n_users).tolist()
    genders = rng.choice(["Male", "Female", "Other"], size=n_users, p=[0.45, 0.50, 0.05]).tolist()
    users_df = pd.DataFrame({"UserID": user_ids, "Age": ages, "Gender": genders})

    # --- Courses ---
    courses_records = []
    for idx, (name, cat, ctype, level, rating) in enumerate(COURSES_DATA, start=1):
        courses_records.append({
            "CourseID": idx,
            "CourseName": name,
            "CourseCategory": cat,
            "CourseType": ctype,
            "CourseLevel": level,
            "CourseRating": rating,
        })
    courses_df = pd.DataFrame(courses_records)

    # --- Transactions ---
    # Assign user types to create realistic clusters
    user_types = rng.choice(
        ["Explorer", "Specialist", "Career-Focused", "Casual"],
        size=n_users,
        p=[0.28, 0.30, 0.25, 0.17],
    )

    transactions = []
    start_date = date(2023, 1, 1)
    end_date = date(2024, 12, 31)
    date_range = (end_date - start_date).days
    tid = 1

    for i, uid in enumerate(user_ids):
        utype = user_types[i]
        # Determine how many courses each learner type buys
        if utype == "Casual":
            n_enrollments = int(rng.integers(1, 4))
        elif utype == "Explorer":
            n_enrollments = int(rng.integers(5, 12))
        elif utype == "Specialist":
            n_enrollments = int(rng.integers(4, 10))
        else:  # Career-Focused
            n_enrollments = int(rng.integers(6, 15))

        # Pick courses based on type
        if utype == "Specialist":
            preferred_cat = rng.choice(CATEGORIES)
            cat_courses = courses_df[courses_df["CourseCategory"] == preferred_cat]["CourseID"].tolist()
            other_courses = courses_df[courses_df["CourseCategory"] != preferred_cat]["CourseID"].tolist()
            pool = (cat_courses * 5) + other_courses
        elif utype == "Career-Focused":
            cert_courses = courses_df[courses_df["CourseType"] == "Bootcamp"]["CourseID"].tolist()
            adv_courses = courses_df[courses_df["CourseLevel"].isin(["Intermediate", "Advanced"])]["CourseID"].tolist()
            pool = (cert_courses * 4) + (adv_courses * 2)
        elif utype == "Explorer":
            pool = courses_df["CourseID"].tolist()
        else:
            beg_courses = courses_df[courses_df["CourseLevel"] == "Beginner"]["CourseID"].tolist()
            pool = beg_courses * 3 + courses_df["CourseID"].tolist()

        chosen = list(set(rng.choice(pool, size=min(n_enrollments, len(set(pool))), replace=False).tolist()))

        for cid in chosen:
            course_info = courses_df[courses_df["CourseID"] == cid].iloc[0]
            if utype == "Career-Focused":
                amount = round(float(rng.uniform(80, 250)), 2)
            elif utype == "Specialist":
                amount = round(float(rng.uniform(50, 180)), 2)
            elif utype == "Explorer":
                amount = round(float(rng.uniform(20, 120)), 2)
            else:
                amount = round(float(rng.uniform(10, 60)), 2)

            txn_date = start_date + timedelta(days=int(rng.integers(0, date_range)))
            transactions.append({
                "TransactionID": tid,
                "UserID": uid,
                "CourseID": int(cid),
                "TransactionDate": txn_date,
                "Amount": amount,
            })
            tid += 1

    transactions_df = pd.DataFrame(transactions)
    return users_df, courses_df, transactions_df
