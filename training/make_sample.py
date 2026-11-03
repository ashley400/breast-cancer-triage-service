import json

from sklearn.datasets import load_breast_cancer

d = load_breast_cancer()
names = [n.replace(" ", "_") for n in d.feature_names]
with open("sample.json", "w") as f:
    json.dump(dict(zip(names, map(float, d.data[0]))), f, indent=2)
print("Wrote sample.json")