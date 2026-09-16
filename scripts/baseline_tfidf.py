import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

TRAIN_FILE = "data/processed/spotify_weak_labels_clean.csv"
GOLDEN_FILE = "data/golden/spotifycares_golden_set_labeled.csv"

train_df = pd.read_csv(TRAIN_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)


# --------------------------------------------------
# 2. Prepare text and labels
# --------------------------------------------------

X_train = train_df["clean_customer"].astype(str)
y_train = train_df["weak_label"]

golden_df = golden_df.dropna(subset=["clean_customer", "gold_label"]).copy()

X_test = golden_df["clean_customer"].astype(str)
y_test = golden_df["gold_label"]


print("=" * 60)
print("TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 60)

print("Training examples:", len(X_train))
print("Golden examples:", len(X_test))


# --------------------------------------------------
# 3. Convert text into TF-IDF features
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF feature shape:")
print("Training:", X_train_tfidf.shape)
print("Golden:", X_test_tfidf.shape)


# --------------------------------------------------
# 4. Train Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# --------------------------------------------------
# 5. Predict on untouched golden set
# --------------------------------------------------

y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 7. Confusion Matrix
# --------------------------------------------------

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print("Labels:", labels)
print(cm)