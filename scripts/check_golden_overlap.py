import pandas as pd

TRAIN_FILE = "data/processed/spotify_weak_labels.csv"
GOLDEN_FILE = "data/golden/spotifycares_golden_set_labeled.csv"

train_df = pd.read_csv(TRAIN_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)

# Remove missing messages
train_df = train_df.dropna(subset=["clean_customer"]).copy()
golden_df = golden_df.dropna(subset=["clean_customer"]).copy()

# Normalize text for comparison
train_text = (
    train_df["clean_customer"]
    .astype(str)
    .str.strip()
    .str.lower()
)

golden_text = (
    golden_df["clean_customer"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Find exact message overlap
overlap = golden_text.isin(set(train_text))

print("=" * 60)
print("GOLDEN SET vs TRAINING DATA OVERLAP CHECK")
print("=" * 60)

print("Training examples:", len(train_df))
print("Golden examples:", len(golden_df))
print("Exact overlapping golden examples:", overlap.sum())

if overlap.sum() > 0:
    print("\nOverlapping examples:")
    
    overlapping_messages = golden_df.loc[
        overlap,
        ["example_id", "clean_customer", "gold_label"]
    ]
    
    print(overlapping_messages.to_string(index=False))
else:
    print("\nNo exact message overlap found.")

print("\nSUMMARY")
print("Exact overlapping golden examples:", overlap.sum())