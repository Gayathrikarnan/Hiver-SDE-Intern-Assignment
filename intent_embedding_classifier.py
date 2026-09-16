import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

TRAIN_FILE = "data/processed/spotify_weak_labels_clean.csv"
GOLDEN_FILE = "data/golden/spotifycares_golden_set_labeled.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 60)
print("EMBEDDING + LOGISTIC REGRESSION INTENT CLASSIFIER")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)

# Remove rows without usable text/labels
train_df = train_df.dropna(
    subset=["clean_customer", "weak_label"]
).copy()

golden_df = golden_df.dropna(
    subset=["clean_customer", "gold_label"]
).copy()

X_train = train_df["clean_customer"].astype(str)
y_train = train_df["weak_label"]

X_test = golden_df["clean_customer"].astype(str)
y_test = golden_df["gold_label"]

print("Training examples:", len(X_train))
print("Golden examples:", len(X_test))


# ---------------------------------------------------------
# LOAD SENTENCE TRANSFORMER MODEL
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ---------------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------------

print("\nCreating training embeddings...")

X_train_embeddings = model.encode(
    X_train.tolist(),
    batch_size=32,
    show_progress_bar=True
)

print("Training embedding shape:")
print(X_train_embeddings.shape)


print("\nCreating golden-set embeddings...")

X_test_embeddings = model.encode(
    X_test.tolist(),
    batch_size=32,
    show_progress_bar=True
)

print("Golden embedding shape:")
print(X_test_embeddings.shape)


# ---------------------------------------------------------
# TRAIN CLASSIFIER
# ---------------------------------------------------------

print("\nTraining Logistic Regression classifier...")

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(
    X_train_embeddings,
    y_train
)

print("Classifier trained.")


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

y_pred = classifier.predict(X_test_embeddings)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")

print("Labels:", labels)
print(cm)