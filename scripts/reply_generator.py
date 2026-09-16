import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_FILE = "data/processed/spotify_weak_labels_clean.csv"
RESPONSE_FILE = "data/processed/spotifycares_preprocessed.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

print("=" * 60)
print("SPOTIFY HISTORICAL CASE RETRIEVAL")
print("=" * 60)

# Load clean training corpus
df = pd.read_csv(TRAIN_FILE)

# Load original file containing historical Spotify replies
response_df = pd.read_csv(RESPONSE_FILE)

# Keep only the columns needed for the reply
response_df = response_df[["example_id", "spotify"]]

# Merge historical Spotify replies using example_id
df = df.merge(
    response_df,
    on="example_id",
    how="left"
)

# Remove rows without customer messages
df = df.dropna(subset=["clean_customer"]).copy()

df["clean_customer"] = df["clean_customer"].astype(str)

print("Historical examples:", len(df))
print("Historical replies available:", df["spotify"].notna().sum())

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")

print("\nCreating historical embeddings...")

historical_embeddings = model.encode(
    df["clean_customer"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embedding shape:")
print(historical_embeddings.shape)


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
import re


def clean_historical_reply(reply):
    """
    Remove Twitter-specific and customer-specific metadata
    from a historical support response.
    """

    if not isinstance(reply, str):
        return ""

    # Remove leading Twitter username
    reply = re.sub(
        r"^@\S+\s*",
        "",
        reply
    )

    # Remove agent initials such as /CH or /GS
    reply = re.sub(
        r"\s*/[A-Z]{2,3}\s*$",
        "",
        reply
    )

    # Remove URLs from historical responses
    reply = re.sub(
        r"https?://\S+",
        "",
        reply
    )

    return reply.strip()

def make_reply_generic(reply):
    """
    Convert a historical response into a safer generic
    support response.
    """

    # Remove common greeting + customer-name pattern
    reply = re.sub(
        r"^(Hey|Hi|Hello)\s+[A-Za-z]+[,!]\s*",
        "",
        reply,
        flags=re.IGNORECASE
    )

    reply = reply.strip()

    # Add a neutral support opening
    if reply:
        reply = "Thanks for reaching out! " + reply

    return reply

def generate_grounded_reply(customer_message, top_k=TOP_K):

    results = retrieve_similar_cases(
        customer_message,
        top_k=top_k
    )

    if results.empty:
        return {
            "reply": (
                "Thanks for reaching out. Could you share "
                "a few more details about the issue?"
            ),
            "similarity": 0.0,
            "intent": "OTHER_UNCLEAR",
            "source_case": None
        }

    # Select the most similar historical case
    best_case = results.iloc[0]

    similarity = float(best_case["similarity"])
    intent = best_case["weak_label"]

    historical_reply = best_case["spotify"]

    if not isinstance(historical_reply, str) or not historical_reply.strip():

        reply = (
            "Thanks for reaching out. Could you share "
            "a few more details about the issue?"
        )

    else:
        reply = clean_historical_reply(historical_reply)

        reply = make_reply_generic(reply)
    return {
        "reply": reply,
        "similarity": similarity,
        "intent": intent,
        "source_case": best_case["example_id"]
    }


# Test retrieval
if __name__ == "__main__":

    test_message = (
        "I paid for Spotify Premium but my account is still "
        "showing the free plan."
    )

    print("\nCustomer message:")
    print(test_message)

    result = generate_grounded_reply(test_message)

    print("\nPredicted intent:")
    print(result["intent"])

    print(
        "\nHistorical similarity:",
        round(result["similarity"], 4)
    )

    print("\nDraft reply:")
    print(result["reply"])

    print("\nGrounding case:")
    print(result["source_case"])