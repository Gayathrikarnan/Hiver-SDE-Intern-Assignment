import pandas as pd

df = pd.read_csv("data/golden/spotifycares_golden_set_labeled.csv")

print("===== GOLDEN SET =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== LABEL DISTRIBUTION =====")
print(df["gold_label"].value_counts())

print("\n===== LABEL PERCENTAGES =====")
print(
    (df["gold_label"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\n===== SAMPLE FROM EACH LABEL =====")

for label in sorted(df["gold_label"].dropna().unique()):
    print("\n" + "=" * 60)
    print("LABEL:", label)
    print("=" * 60)

    examples = df[df["gold_label"] == label]["clean_customer"].head(5)

    for example in examples:
        print("-", example)