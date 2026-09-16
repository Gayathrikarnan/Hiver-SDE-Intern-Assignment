import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# FILES
# =========================================================

TRAIN_FILE = "data/processed/spotify_weak_labels_clean.csv"
RESPONSE_FILE = "data/processed/spotifycares_preprocessed.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

# Decision thresholds
INTENT_THRESHOLD = 0.60
RETRIEVAL_THRESHOLD = 0.75


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("HIVER SPOTIFY AI SUPPORT AGENT")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)

response_df = pd.read_csv(RESPONSE_FILE)

response_df = response_df[
    ["example_id", "spotify"]
]

# Add historical Spotify response
train_df = train_df.merge(
    response_df,
    on="example_id",
    how="left"
)

train_df = train_df.dropna(
    subset=["clean_customer", "weak_label"]
).copy()

train_df["clean_customer"] = (
    train_df["clean_customer"].astype(str)
)

print("\nTraining examples:", len(train_df))


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded.")


# =========================================================
# CREATE TRAINING EMBEDDINGS
# =========================================================

print("\nCreating training embeddings...")

X_train = model.encode(
    train_df["clean_customer"].tolist(),
    batch_size=32,
    show_progress_bar=True
)

y_train = train_df["weak_label"]


# =========================================================
# TRAIN INTENT CLASSIFIER
# =========================================================

print("\nTraining intent classifier...")

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(
    X_train,
    y_train
)

print("Classifier trained.")


# =========================================================
# HISTORICAL RETRIEVAL
# =========================================================

print("\nPreparing historical retrieval...")

historical_embeddings = model.encode(
    train_df["clean_customer"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Retrieval ready.")


# =========================================================
# INTENT CLASSIFICATION
# =========================================================

def classify_intent(customer_message):

    embedding = model.encode(
        [customer_message]
    )

    probabilities = classifier.predict_proba(
        embedding
    )[0]

    best_index = np.argmax(probabilities)

    intent = classifier.classes_[best_index]

    confidence = float(
        probabilities[best_index]
    )

    return intent, confidence


# =========================================================
# RETRIEVE HISTORICAL CASES
# =========================================================

def retrieve_cases(customer_message, top_k=TOP_K):

    query_embedding = model.encode(
        [customer_message],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        historical_embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = train_df.iloc[
        top_indices
    ].copy()

    results["similarity"] = (
        similarities[top_indices]
    )

    return results


# =========================================================
# CLEAN HISTORICAL RESPONSE
# =========================================================

def clean_reply(reply):

    if not isinstance(reply, str):
        return ""

    # Remove Twitter username
    reply = re.sub(
        r"^@\S+\s*",
        "",
        reply
    )

    # Remove URLs
    reply = re.sub(
        r"https?://\S+",
        "",
        reply
    )

    # Remove agent initials
    reply = re.sub(
        r"\s*/[A-Z]{2,3}\s*$",
        "",
        reply
    )

    # Remove common greeting + customer name
    reply = re.sub(
        r"^(Hey|Hi|Hello)\s+[A-Za-z]+[,!]\s*",
        "",
        reply,
        flags=re.IGNORECASE
    )

    reply = reply.strip()

    if reply:
        reply = (
            "Thanks for reaching out! "
            + reply
        )

    return reply


# =========================================================
# ESCALATION POLICY
# =========================================================

def decide_escalation(
    intent,
    confidence,
    similarity,
    customer_message
):

    message_lower = customer_message.lower()

    # 1. Uncertain classification
    if confidence < INTENT_THRESHOLD:

        return (
            "ESCALATE",
            "Low intent-classification confidence"
        )

    # 2. No sufficiently similar historical case
    if similarity < RETRIEVAL_THRESHOLD:

        return (
            "ESCALATE",
            "No sufficiently similar historical support case"
        )

    # 3. Account-specific requests
    account_terms = [
        "refund",
        "charged",
        "payment",
        "billing",
        "cancel",
        "hacked",
        "account email",
        "change email"
    ]

    if any(
        term in message_lower
        for term in account_terms
    ):

        return (
            "ESCALATE",
            "Account or payment-specific action may require human support"
        )

    # Otherwise handle automatically
    return (
        "AUTO-HANDLE",
        "High intent confidence and strong historical support evidence"
    )


# =========================================================
# COMPLETE AGENT
# =========================================================

def run_agent(customer_message):

    # Step 1: classify intent
    intent, confidence = classify_intent(
        customer_message
    )

    # Step 2: retrieve similar cases
    results = retrieve_cases(
        customer_message
    )

    if results.empty:

        return {
            "customer_message": customer_message,
            "intent": "OTHER_UNCLEAR",
            "confidence": 0.0,
            "similarity": 0.0,
            "reply": (
                "Thanks for reaching out. "
                "Could you share a few more details "
                "about the issue?"
            ),
            "decision": "ESCALATE",
            "reason": "No historical evidence available"
        }

    best_case = results.iloc[0]

    similarity = float(
        best_case["similarity"]
    )

    historical_reply = best_case["spotify"]

    # Step 3: generate grounded response
    reply = clean_reply(
        historical_reply
    )

    if not reply:

        reply = (
            "Thanks for reaching out. "
            "Could you share a few more details "
            "about the issue?"
        )

    # Step 4: escalation
    decision, reason = decide_escalation(
        intent,
        confidence,
        similarity,
        customer_message
    )

    return {
        "customer_message": customer_message,
        "intent": intent,
        "confidence": confidence,
        "similarity": similarity,
        "reply": reply,
        "decision": decision,
        "reason": reason,
        "source_case": best_case["example_id"]
    }


# =========================================================
# DEMO
# =========================================================

if __name__ == "__main__":

    test_messages = [

        "I paid for Spotify Premium but my account is still showing the free plan.",

        "Spotify keeps crashing whenever I open the app.",

        "Can you add lyrics translation to Spotify?",

        "Where can I find my playlists?"
    ]

    for message in test_messages:

        print("\n")
        print("=" * 60)

        result = run_agent(message)

        print("CUSTOMER:")
        print(result["customer_message"])

        print("\nINTENT:")
        print(result["intent"])

        print(
            "\nCONFIDENCE:",
            round(result["confidence"], 4)
        )

        print(
            "\nHISTORICAL SIMILARITY:",
            round(result["similarity"], 4)
        )

        print("\nDRAFT REPLY:")
        print(result["reply"])

        print("\nDECISION:")
        print(result["decision"])

        print("\nREASON:")
        print(result["reason"])

        print("\nGROUNDING CASE:")
        print(result["source_case"])