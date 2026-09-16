import pandas as pd

df = pd.read_csv("data/golden/spotifycares_golden_set_labeled.csv")

unclear = df[df["gold_label"] == "OTHER_UNCLEAR"]

print("===== OTHER_UNCLEAR EXAMPLES =====")
print("Total:", len(unclear))

for i, row in unclear.iterrows():
    print("\n" + "=" * 70)
    print("Example ID:", row["example_id"])
    print("Message:", row["clean_customer"])