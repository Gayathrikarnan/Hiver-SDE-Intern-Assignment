import pandas as pd
from collections import Counter
import re

# Load dataset
df = pd.read_csv("data/processed/spotifycares_preprocessed.csv")

# Remove rows without customer messages
messages = df["clean_customer"].dropna().astype(str)

print("===== BASIC MESSAGE STATISTICS =====")
print("Total rows:", len(df))
print("Usable customer messages:", len(messages))
print("Unique customer messages:", messages.nunique())

print("\n===== MOST COMMON WORDS =====")

# Combine all messages
text = " ".join(messages).lower()

# Keep words with at least 3 characters
words = re.findall(r"\b[a-zA-Z]{3,}\b", text)

# Common English words that don't tell us much
stopwords = {
    "the", "and", "for", "you", "that", "this",
    "have", "with", "are", "but", "not", "was",
    "from", "can", "your", "they", "will", "would",
    "what", "when", "how", "why", "its", "has",
    "been", "get", "all", "just", "like", "there",
    "about", "does", "did", "out", "my", "our",
    "spotify"
}

filtered_words = [
    word for word in words
    if word not in stopwords
]

word_counts = Counter(filtered_words)

for word, count in word_counts.most_common(50):
    print(f"{word:20} {count}")

print("\n===== SAMPLE CUSTOMER MESSAGES =====")

for message in messages.sample(30, random_state=42):
    print("-", message)