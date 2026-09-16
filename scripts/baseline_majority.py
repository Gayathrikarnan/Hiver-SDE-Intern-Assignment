import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. Load training data
# --------------------------------------------------

TRAIN_FILE = "data/processed/spotify_weak_labels.csv"
GOLDEN_FILE = "data/golden/spotifycares_golden_set_labeled.csv"

train_df = pd.read_csv(TRAIN_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)


# --------------------------------------------------
# 2. Find the majority class
# --------------------------------------------------

majority_class = train_df["weak_label"].value_counts().idxmax()

print("=" * 60)
print("MAJORITY CLASS BASELINE")
print("=" * 60)

print("Majority class:", majority_class)


# --------------------------------------------------
# 3. Predict the same class for every golden example
# --------------------------------------------------

y_true = golden_df["gold_label"]

y_pred = [majority_class] * len(golden_df)


# --------------------------------------------------
# 4. Evaluate
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

print("\nAccuracy:", round(accuracy, 4))


print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 5. Confusion Matrix
# --------------------------------------------------

labels = sorted(golden_df["gold_label"].unique())

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print("Labels:", labels)
print(cm)