import pandas as pd

df = pd.read_csv("dataset.csv")
print(df["label"].value_counts())
print(f"\nClass balance: {df['label'].mean():.2%} feasible")