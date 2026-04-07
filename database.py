import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_engine():
    if not DATABASE_URL:
        return None
    try:
        from sqlalchemy import create_engine
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception:
        return None


def is_demo_mode():
    return DATABASE_URL is None


def load_tables_from_db():
    engine = get_engine()
    if engine is None:
        return None, None, None
    try:
        users = pd.read_sql("SELECT * FROM users", engine)
        courses = pd.read_sql("SELECT * FROM courses", engine)
        transactions = pd.read_sql("SELECT * FROM transactions", engine)
        return users, courses, transactions
    except Exception:
        return None, None, None


def seed_database(users_df, courses_df, transactions_df):
    engine = get_engine()
    if engine is None:
        return False
    try:
        users_df.to_sql("users", engine, if_exists="append", index=False)
        courses_df.to_sql("courses", engine, if_exists="append", index=False)
        transactions_df.to_sql("transactions", engine, if_exists="append", index=False)
        return True
    except Exception as e:
        print(f"Seed error: {e}")
        return False
