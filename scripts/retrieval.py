import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_FILE = "data/processed/spotify_weak_labels_clean.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("=" * 60)
print("SPOTIFY HISTORICAL CASE RETRIEVAL")
print("=" * 60)

df = pd.read_csv(TRAIN_FILE)

df = df.dropna(subset=["clean_customer"]).copy()

df["clean_customer"] = df["clean_customer"].astype(str)

print("Historical examples:", len(df))


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# --------------------------------------------------
# Create historical embeddings
# --------------------------------------------------

print("\nCreating historical embeddings...")

historical_embeddings = model.encode(
    df["clean_customer"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embedding shape:")
print(historical_embeddings.shape)


# --------------------------------------------------
# Retrieval function
# --------------------------------------------------

def retrieve_similar_cases(customer_message, top_k=TOP_K):

    query_embedding = model.encode(
        [customer_message],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        historical_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = df.iloc[top_indices].copy()

    results["similarity"] = similarities[top_indices]

    return results


# --------------------------------------------------
# Test retrieval
# --------------------------------------------------

test_message = (
    "I paid for Spotify Premium but my account is still showing "
    "the free plan."
)

print("\nCustomer message:")
print(test_message)

print("\nTop similar historical cases:")

results = retrieve_similar_cases(test_message)

for i, (_, row) in enumerate(results.iterrows(), start=1):

    print("\n" + "-" * 60)
    print("Result", i)
    print("Similarity:", round(row["similarity"], 4))
    print("Intent:", row["weak_label"])
    print("Historical message:")
    print(row["clean_customer"])