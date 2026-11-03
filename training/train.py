import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MALIGNANT = 0  # in this dataset target 0 = malignant (positive class)
MIN_TEST_RECALL = 0.95


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-version", required=True)
    p.add_argument("--params", default="params.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.params) as f:
        params = yaml.safe_load(f)
    seed = params["seed"]
    hp = params["train"]

    df = pd.read_csv(args.data_path)
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].to_numpy()
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=hp["test_size"], stratify=y, random_state=seed
    )
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")

    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=hp["logreg_C"], max_iter=hp["logreg_max_iter"], random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=hp["rf_n_estimators"], max_depth=hp["rf_max_depth"],
                random_state=seed)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = {
        "recall": make_scorer(recall_score, pos_label=MALIGNANT),
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",  # AUC is the same whichever class is called positive
    }

    results = {}
    for name, pipe in candidates.items():
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba_benign = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_malignant_recall": float(s["test_recall"].mean()),
            "cv_accuracy": float(s["test_accuracy"].mean()),
            "cv_roc_auc": float(s["test_roc_auc"].mean()),
            "test_malignant_recall": float(recall_score(y_test, pred, pos_label=MALIGNANT)),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_roc_auc": float(roc_auc_score(y_test, proba_benign)),
        }

    best = max(results, key=lambda n: results[n]["cv_malignant_recall"])

    print("\n=== Results (malignant = positive class) ===")
    header = f"{'model':22}{'cvRecall':>9}{'cvAcc':>8}{'cvAUC':>8}{'tRecall':>9}{'tAcc':>8}{'tAUC':>8}"
    print(header)
    for n, r in results.items():
        mark = "  <-- selected" if n == best else ""
        print(f"{n:22}{r['cv_malignant_recall']:9.4f}{r['cv_accuracy']:8.4f}"
              f"{r['cv_roc_auc']:8.4f}{r['test_malignant_recall']:9.4f}"
              f"{r['test_accuracy']:8.4f}{r['test_roc_auc']:8.4f}{mark}")

    test_recall = results[best]["test_malignant_recall"]
    print(f"\nSelected: {best}  test malignant recall = {test_recall:.4f}")
    if test_recall < MIN_TEST_RECALL:
        print(f"FAIL: test malignant recall {test_recall:.4f} < {MIN_TEST_RECALL}",
              file=sys.stderr)
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_model = candidates[best]
    joblib.dump(final_model, out / "model.joblib")

    selected = results[best]
    card = {
        "model_version": args.model_version,
        "model_family": final_model[-1].__class__.__name__,
        "feature_names": feature_names,
        "metrics": {k: v for k, v in selected.items() if k.startswith("test_")},
        "seed": seed,
        "scikit_learn_version": sklearn.__version__,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"selected_model": best, "selected": selected, "candidates": results}, indent=2))
    print(f"Saved model.joblib, model_card.json, metrics.json -> {out}")


if __name__ == "__main__":
    main()