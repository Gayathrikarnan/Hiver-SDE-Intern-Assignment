import pandas as pd

df = pd.read_csv("data/processed/spotifycares_preprocessed.csv")

# Remove missing customer messages
df = df.dropna(subset=["clean_customer"])

# Convert to lowercase for searching
df["message_lower"] = df["clean_customer"].str.lower()

# Themes we want to investigate
themes = {
    "Billing / Payment": [
        "charged", "charge", "payment", "refund",
        "billing", "money", "credit card"
    ],

    "Premium / Subscription": [
        "premium", "subscription", "subscribe",
        "cancel", "renew"
    ],

    "Playlist": [
        "playlist", "playlists"
    ],

    "Playback": [
        "play", "playing", "pause", "shuffle",
        "repeat", "skip"
    ],

    "Account": [
        "account", "login", "password", "username",
        "email"
    ],

    "Student / Family": [
        "student", "family", "hulu"
    ],

    "Availability / Region": [
        "available", "availability", "country",
        "region", "vietnam"
    ],

    "Songs / Content": [
        "song", "songs", "album", "artist",
        "missing", "add"
    ],

    "Device / App": [
        "iphone", "android", "phone", "chromecast",
        "app", "update"
    ]
}

for theme, keywords in themes.items():

    mask = df["message_lower"].apply(
        lambda text: any(keyword in text for keyword in keywords)
    )

    matching = df[mask]

    print("\n" + "=" * 70)
    print(theme)
    print("Matching messages:", len(matching))
    print("=" * 70)

    # Show up to 10 examples
    for message in matching["clean_customer"].head(10):
        print("-", message)