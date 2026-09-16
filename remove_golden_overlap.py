import pandas as pd

TRAIN_FILE = "data/processed/spotify_weak_labels.csv"
GOLDEN_FILE = "data/golden/spotifycares_golden_set_labeled.csv"
OUTPUT_FILE = "data/processed/spotify_weak_labels_clean.csv"

train_df = pd.read_csv(TRAIN_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)

# Remove missing messages
train_df = train_df.dropna(subset=["clean_customer"]).copy()
golden_df = golden_df.dropna(subset=["clean_customer"]).copy()

# Normalize text before comparison
train_df["_normalized_text"] = (
    train_df["clean_customer"]
    .astype(str)
    .str.strip()
    .str.lower()
)

golden_text = set(
    golden_df["clean_customer"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Identify training examples that overlap with golden set
overlap_mask = train_df["_normalized_text"].isin(golden_text)

print("=" * 60)
print("REMOVING GOLDEN SET OVERLAP")
print("=" * 60)

print("Original training examples:", len(train_df))
print("Overlapping training examples:", overlap_mask.sum())

clean_df = train_df.loc[~overlap_mask].copy()

# Remove helper column
clean_df = clean_df.drop(columns=["_normalized_text"])

clean_df.to_csv(OUTPUT_FILE, index=False)

print("Clean training examples:", len(clean_df))
print("Saved to:", OUTPUT_FILE)