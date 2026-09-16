import pandas as pd

# Load the preprocessed Spotify dataset
df = pd.read_csv("data/processed/spotifycares_preprocessed.csv")

print("===== DATASET OVERVIEW =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n===== COLUMN NAMES =====")
for column in df.columns:
    print("-", column)

print("\n===== FIRST 5 CUSTOMER MESSAGES =====")
print(df["clean_customer"].head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== UNIQUE CUSTOMER MESSAGES =====")
print(df["clean_customer"].nunique())

print("\n===== DATE RANGE =====")
print("First:", df["customer_created_at"].min())
print("Last :", df["customer_created_at"].max())