import pandas as pd

PATH = "data/raw/breast_cancer.csv"

df = pd.read_csv(PATH)
df = df.drop(df.sample(n=30, random_state=42).index)
df.to_csv(PATH, index=False)
print(f"New row count: {len(df)}")