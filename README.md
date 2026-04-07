# EduPro — Learner Intelligence Platform

Student Segmentation and Personalized Course Recommendation System

---

## Features

| Module | Description |
|---|---|
| 🏠 Overview Dashboard | KPIs, segment distribution, enrollment trends |
| 👤 Learner Explorer | Individual learner profiles, history, similar learners |
| 📊 Cluster Dashboard | K-Means scatter, Elbow/Silhouette, feature radar |
| 💡 Recommendations | Personalized, filterable course recommendations |
| 🔍 Segment Comparison | Side-by-side behavioral analysis across segments |

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure database (optional)

Copy `.env.example` to `.env` and set your MySQL connection string:

```
DATABASE_URL=mysql://root:yourpassword@localhost:3306/edupro
```

If you skip this step, the app runs in **Demo Mode** with 300 synthetic learners.

### 3. Set up the database (if using MySQL)

```sql
-- Run sql/schema.sql in MySQL Workbench or command line
mysql -u root -p < sql/schema.sql
```

### 4. Run the app

```bash
streamlit run app.py
```

Or on Windows:
```bash
run.bat
```

---

## Project Structure

```
edupro-system/
├── app.py                        # Main dashboard (Overview)
├── pages/
│   ├── 1_Learner_Explorer.py    # Individual learner profiles
│   ├── 2_Cluster_Dashboard.py  # Cluster visualizations
│   ├── 3_Recommendations.py    # Personalized recommendations
│   └── 4_Segment_Comparison.py # Segment analysis
├── src/
│   ├── database.py              # MySQL connection + fallback
│   ├── data_generator.py        # Synthetic data generation
│   ├── feature_engineering.py  # Learner feature creation
│   ├── clustering.py            # K-Means + Hierarchical
│   └── recommendations.py      # Recommendation engine
├── sql/
│   └── schema.sql               # MySQL schema
├── requirements.txt
├── .env.example
└── README.md
```

---

## Learner Segments

| Segment | Description |
|---|---|
| 🔵 Explorer | Diverse learners across many categories |
| 🟣 Specialist | Deep-focus, advanced content in one domain |
| 🟢 Career-Focused | Bootcamp & certification oriented, high spenders |
| 🟡 Casual Learner | Low engagement, beginner courses |

---

## Tech Stack

- **Frontend**: Streamlit + Plotly
- **ML**: scikit-learn (K-Means, PCA, StandardScaler)
- **Database**: MySQL + SQLAlchemy
- **Data**: pandas + numpy
