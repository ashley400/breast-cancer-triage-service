from pathlib import Path

from sklearn.datasets import load_breast_cancer

OUT = Path("data/raw/breast_cancer.csv")


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]

    feature_cols = [c for c in df.columns if c != "target"]
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"
    assert len(feature_cols) == 30, f"Expected 30 features, got {len(feature_cols)}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows, {len(feature_cols)} features -> {OUT}")


if __name__ == "__main__":
    main()