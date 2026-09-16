import pandas as pd
import re

# --------------------------------------------------
# 1. Load the Spotify dataset
# --------------------------------------------------

INPUT_FILE = "data/processed/spotifycares_preprocessed.csv"
OUTPUT_FILE = "data/processed/spotify_weak_labels.csv"

df = pd.read_csv(INPUT_FILE)

# Remove rows where customer message is missing
df = df.dropna(subset=["clean_customer"]).copy()

# Convert message to lowercase
df["text"] = df["clean_customer"].astype(str).str.lower()


# --------------------------------------------------
# 2. Define high-confidence intent rules
# --------------------------------------------------

def assign_intent(text):

    # Account / login
    if re.search(
        r"\b(log ?in|login|sign ?in|password|facebook login|account access|can't access account)\b",
        text
    ):
        return "ACCOUNT_LOGIN"

    # Billing / subscription
    if re.search(
        r"\b(charged|charge|billing|payment|refund|subscription|premium|student discount|family plan)\b",
        text
    ):
        return "BILLING_SUBSCRIPTION"

    # Playback
    if re.search(
        r"\b(not playing|won't play|wont play|can't play|cant play|skipping|skip songs|pause|playback|shuffle|repeat)\b",
        text
    ):
        return "PLAYBACK_ISSUE"

    # Playlist / library
    if re.search(
        r"\b(playlist|playlists|saved songs|saved albums|library|my songs|collection)\b",
        text
    ):
        return "PLAYLIST_LIBRARY"

    # App crash / performance
    if re.search(
        r"\b(crash|crashing|crashed|freeze|freezing|frozen|slow|lag|cpu|unstable|not responding)\b",
        text
    ):
        return "APP_CRASH_PERFORMANCE"

    # Connectivity / sync
    if re.search(
        r"\b(sync|synced|synchroni|connect|connected|connection|internet|offline|chromecast|fire tv|smart tv)\b",
        text
    ):
        return "CONNECTIVITY_SYNC"

    # Availability / region
    if re.search(
        r"\b(not available|unavailable|available in|country|region|india|hong kong|usa|united states|vietnam)\b",
        text
    ):
        return "AVAILABILITY_REGION"

    # Feature requests
    if re.search(
        r"\b(please add|please bring back|bring back|add a feature|feature request|would love|can you add|wish you had)\b",
        text
    ):
        return "FEATURE_REQUEST"

    # General feedback
    if re.search(
        r"\b(love spotify|great job|thank you|thanks|amazing|awesome|hate|terrible|frustrating|disappointed)\b",
        text
    ):
        return "FEEDBACK"

    # Generic bug report
    if re.search(
        r"\b(bug|error|issue|problem|broken|doesn't work|doesnt work|not working|fix this|fix it)\b",
        text
    ):
        return "BUG_REPORT"

    # No confident label
    return None


# --------------------------------------------------
# 3. Apply the rules
# --------------------------------------------------

df["weak_label"] = df["text"].apply(assign_intent)

# Keep only messages where we found a confident label
weak_df = df[df["weak_label"].notna()].copy()


# --------------------------------------------------
# 4. Remove duplicates
# --------------------------------------------------

weak_df = weak_df.drop_duplicates(subset=["clean_customer"])


# --------------------------------------------------
# 5. Save the training dataset
# --------------------------------------------------

columns_to_save = [
    "example_id",
    "clean_customer",
    "customer_created_at",
    "weak_label"
]

weak_df[columns_to_save].to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 6. Print summary
# --------------------------------------------------

print("=" * 60)
print("WEAKLY LABELLED TRAINING DATA")
print("=" * 60)

print("Original rows:", len(df))
print("Rows with labels:", len(weak_df))

print("\nLabel distribution:")
print(weak_df["weak_label"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)