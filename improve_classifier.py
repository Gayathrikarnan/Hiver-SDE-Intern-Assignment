import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = "data/processed/spotify_weak_labels_clean.csv"
GOLDEN_FILE = "data/golden/spotifycares_golden_set_labeled.csv"


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
golden = pd.read_csv(GOLDEN_FILE)

# Training file uses "weak_label"
train = train.dropna(
    subset=["clean_customer", "weak_label"]
)

# Golden set uses "gold_label"
golden = golden.dropna(
    subset=["clean_customer", "gold_label"]
)

X_train = train["clean_customer"].astype(str).values
y_train = train["weak_label"].astype(str).values

X_test = golden["clean_customer"].astype(str).values
y_test = golden["gold_label"].astype(str).values


print("=" * 60)
print("CLASSIFIER IMPROVEMENT EXPERIMENT")
print("=" * 60)

print("Training examples:", len(X_train))
print("Golden examples:", len(X_test))


# ============================================================
# EXPERIMENT 1
# TF-IDF WORD + LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("[1] TF-IDF WORD + LINEAR SVM")
print("=" * 60)

word_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

print("Training feature shape:", X_train_word.shape)

svm_word = LinearSVC(
    class_weight="balanced",
    C=1.5
)

svm_word.fit(X_train_word, y_train)

pred_word = svm_word.predict(X_test_word)

acc_word = accuracy_score(y_test, pred_word)
f1_word = f1_score(
    y_test,
    pred_word,
    average="macro"
)

print("Accuracy:", round(acc_word, 4))
print("Macro F1:", round(f1_word, 4))


# ============================================================
# EXPERIMENT 2
# TF-IDF WORD + CHARACTER + LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("[2] TF-IDF WORD + CHARACTER + LINEAR SVM")
print("=" * 60)

features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char_wb",
            max_features=15000,
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True
        )
    )
])

X_train_combined = features.fit_transform(X_train)
X_test_combined = features.transform(X_test)

print("Combined feature shape:", X_train_combined.shape)

svm_combined = LinearSVC(
    class_weight="balanced",
    C=1.0
)

svm_combined.fit(
    X_train_combined,
    y_train
)

pred_combined = svm_combined.predict(
    X_test_combined
)

acc_combined = accuracy_score(
    y_test,
    pred_combined
)

f1_combined = f1_score(
    y_test,
    pred_combined,
    average="macro"
)

print("Accuracy:", round(acc_combined, 4))
print("Macro F1:", round(f1_combined, 4))


# ============================================================
# EXPERIMENT 3
# MINILM EMBEDDINGS + LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("[3] MiniLM EMBEDDINGS + LINEAR SVM")
print("=" * 60)

print("Loading MiniLM model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Creating training embeddings...")

X_train_emb = model.encode(
    X_train,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Creating golden-set embeddings...")

X_test_emb = model.encode(
    X_test,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Training Linear SVM...")

svm_emb = LinearSVC(
    class_weight="balanced",
    C=1.0
)

svm_emb.fit(
    X_train_emb,
    y_train
)

pred_emb = svm_emb.predict(
    X_test_emb
)

acc_emb = accuracy_score(
    y_test,
    pred_emb
)

f1_emb = f1_score(
    y_test,
    pred_emb,
    average="macro"
)

print("Accuracy:", round(acc_emb, 4))
print("Macro F1:", round(f1_emb, 4))


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    "Current MiniLM + Logistic Regression"
    f" : Accuracy 0.5101 | Macro F1 0.5300"
)

print(
    "TF-IDF + Linear SVM"
    f"                  : Accuracy {acc_word:.4f}"
    f" | Macro F1 {f1_word:.4f}"
)

print(
    "Word + Char TF-IDF + Linear SVM"
    f"      : Accuracy {acc_combined:.4f}"
    f" | Macro F1 {f1_combined:.4f}"
)

print(
    "MiniLM + Linear SVM"
    f"                   : Accuracy {acc_emb:.4f}"
    f" | Macro F1 {f1_emb:.4f}"
)


# ============================================================
# FIND BEST MODEL
# ============================================================

results = [
    (
        "Current MiniLM + Logistic Regression",
        0.5101,
        0.5300
    ),
    (
        "TF-IDF + Linear SVM",
        acc_word,
        f1_word
    ),
    (
        "Word + Char TF-IDF + Linear SVM",
        acc_combined,
        f1_combined
    ),
    (
        "MiniLM + Linear SVM",
        acc_emb,
        f1_emb
    )
]

best = max(
    results,
    key=lambda x: x[2]
)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best[0])
print("Accuracy:", round(best[1], 4))
print("Macro F1:", round(best[2], 4))


# ============================================================
# CLASSIFICATION REPORT FOR BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

if best[0] == "TF-IDF + Linear SVM":

    print(
        classification_report(
            y_test,
            pred_word,
            zero_division=0
        )
    )

elif best[0] == "Word + Char TF-IDF + Linear SVM":

    print(
        classification_report(
            y_test,
            pred_combined,
            zero_division=0
        )
    )

elif best[0] == "MiniLM + Linear SVM":

    print(
        classification_report(
            y_test,
            pred_emb,
            zero_division=0
        )
    )

else:

    print(
        "The current model's classification report "
        "was already recorded."
    )


print("\nExperiment completed.")