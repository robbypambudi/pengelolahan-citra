"""Machine-learning part of the paper (Tables 1, 4 and 5, Fig. 6).

For each subset (100*7 originals, 200*7 with horizontal flips):
  1. resize every image to 220x220 and extract the global feature
     (colour histogram + Haralick + Hu moments), scale to [0, 1] and store
     features + labels in HDF5;
  2. split 80% train / 20% test;
  3. 10-fold cross-validate the seven classifiers of Table 1 on the training
     split -> mean accuracy (Table 4);
  4. fit on the full training split and report accuracy and weighted
     precision / recall / F1 on the test split (Table 5).

`--leak-free` keeps every lesion (and therefore an image and its flipped copy)
in a single split/fold. The paper does not do this; see README.md.
"""
import argparse
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import CLASSES, DATA_DIR, OUTPUT_DIR, SEED, load_manifest, read_image
from features import global_feature

ML_IMAGE_SIZE = 220
TEST_SIZE = 0.20
N_FOLDS = 10
SUBSET_LABELS = {
    "original": "Without augmentation (100*7 images)",
    "augmented": "With horizontal flip augmentation (200*7 images)",
}


def get_models():
    """Classifiers and hyper-parameters of Table 1."""
    return {
        "LR": LogisticRegression(random_state=9, max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(solver="svd"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DT": DecisionTreeClassifier(random_state=9),
        "RF": RandomForestClassifier(n_estimators=200, random_state=0),
        "NB": GaussianNB(var_smoothing=1e-09),
        "SVM": SVC(kernel="linear", C=1, random_state=0),
    }


def _extract(path):
    return global_feature(read_image(DATA_DIR / path, ML_IMAGE_SIZE))


def build_features(subset: str, out_dir):
    """Extract (or load cached) scaled global features for one subset."""
    h5_path = out_dir / f"features_{subset}.h5"
    manifest = load_manifest(subset)
    if h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            features, labels = f["features"][()], f["labels"][()]
        print(f"[{subset}] loaded cached features {features.shape} from {h5_path.name}")
    else:
        raw = Parallel(n_jobs=-1)(delayed(_extract)(p) for p in manifest["path"])
        encoder = LabelEncoder().fit(CLASSES)
        labels = encoder.transform(manifest["dx"])
        features = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(raw))
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("features", data=features)
            f.create_dataset("labels", data=labels)
        print(f"[{subset}] extracted features {features.shape} -> {h5_path.name}")
    return features, labels, manifest["lesion_id"].to_numpy()


def split(features, labels, groups, leak_free):
    if leak_free:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        train_idx, test_idx = next(splitter.split(features, labels, groups))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=TEST_SIZE, random_state=SEED)
    return train_idx, test_idx


def run_subset(subset, out_dir, leak_free):
    features, labels, groups = build_features(subset, out_dir)
    train_idx, test_idx = split(features, labels, groups, leak_free)
    x_train, y_train, x_test, y_test = features[train_idx], labels[train_idx], features[test_idx], labels[test_idx]
    print(f"[{subset}] train {len(train_idx)} / test {len(test_idx)}")

    if leak_free:
        cv = GroupKFold(n_splits=N_FOLDS)
        cv_kwargs = {"groups": groups[train_idx]}
    else:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_kwargs = {}

    rows, reports = [], {}
    for name, model in get_models().items():
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, **cv_kwargs)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
        rows.append({
            "subset": subset, "model": name,
            "cv_accuracy_mean": scores.mean(), "cv_accuracy_std": scores.std(),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision_w": p, "test_recall_w": r, "test_f1_w": f1,
        })
        reports[name] = classification_report(
            y_test, pred, labels=range(len(CLASSES)), target_names=CLASSES, zero_division=0)
        print(f"  {name:4s} 10-fold CV acc {scores.mean():.5f} (+/- {scores.std():.3f}) | "
              f"test acc {rows[-1]['test_accuracy']:.4f}  P {p:.2f}  R {r:.2f}  F1 {f1:.2f}")

    with open(out_dir / f"classification_reports_{subset}.txt", "w") as f:
        for name, report in reports.items():
            f.write(f"===== {name} ({SUBSET_LABELS[subset]}) =====\n{report}\n")
    return rows


def plot_results(results: pd.DataFrame, out_dir):
    aug = results[results["subset"] == "augmented"].sort_values("test_accuracy", ascending=False)
    metrics = {"test_accuracy": "Accuracy", "test_precision_w": "Precision",
               "test_recall_w": "Recall", "test_f1_w": "F1-score"}
    x = np.arange(len(aug))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (col, label) in enumerate(metrics.items()):
        ax.bar(x + (i - 1.5) * width, aug[col] * 100, width, label=label)
    ax.set_xticks(x, aug["model"])
    ax.set_ylabel("%")
    ax.set_title("ML models' performance (200*7 images, test split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ml_performance.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", choices=["original", "augmented", "both"], default="both")
    parser.add_argument("--leak-free", action="store_true",
                        help="group-aware split/CV by lesion_id (not in the paper)")
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / ("ml_leak_free" if args.leak_free else "ml")
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["original", "augmented"] if args.subset == "both" else [args.subset]
    rows = []
    for subset in subsets:
        rows += run_subset(subset, out_dir, args.leak_free)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)

    table4 = results.pivot(index="subset", columns="model", values="cv_accuracy_mean")
    table4 = table4.reindex(index=subsets, columns=list(get_models()))
    table4.index = [SUBSET_LABELS[s] for s in table4.index]
    print("\nTable 4 - mean 10-fold CV accuracy")
    print(table4.round(5).to_string())

    if "augmented" in subsets:
        aug = results[results["subset"] == "augmented"].set_index("model")
        table5 = (aug[["test_accuracy", "test_precision_w", "test_recall_w", "test_f1_w"]] * 100).round(0).T
        table5.index = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)"]
        print("\nTable 5 - test split, weighted averages (200*7 images)")
        print(table5.to_string())
        plot_results(results, out_dir)

    with open(out_dir / "table4.json", "w") as f:
        json.dump(table4.to_dict(orient="index"), f, indent=2)
    print(f"\nresults written to {out_dir}")


if __name__ == "__main__":
    main()
