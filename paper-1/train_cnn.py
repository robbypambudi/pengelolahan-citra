"""Train / evaluate the customised CNN (Tables 3, 6, 7 and Fig. 5).

Data: the 200*7 image set (100 originals + 100 horizontal flips per class),
resized to 96x96x3, converted to NumPy arrays and scaled to [0, 1]; labels are
one-hot encoded with LabelBinarizer. 80% is used for training and 20% (280
images) for testing. During training random transformations (rotation, shift,
shear, zoom, flip) are applied on the fly, with one pass over the training
images per epoch.

Hyper-parameters (Table 3): Adam, categorical cross-entropy, 150 epochs,
batch size 32, learning rate 1e-3 decreasing to at most 1e-5.

`--folds K` additionally runs K-fold cross-validation and reports the mean
accuracy of the K models. `--leak-free` keeps every lesion (and therefore an
image and its flipped copy) on one side of each split; the paper does not.
"""
import argparse
import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer

from cnn_model import build_cnn
from common import CLASS_NAMES, CLASSES, DATA_DIR, OUTPUT_DIR, load_manifest, read_image

IMAGE_SIZE = 96
EPOCHS = 150
BATCH_SIZE = 32
INIT_LR = 1e-3
MIN_LR = 1e-5
TEST_SIZE = 0.20


def load_data(subset, size=IMAGE_SIZE):
    manifest = load_manifest(subset)
    images = np.array([cv2.cvtColor(read_image(DATA_DIR / p, size), cv2.COLOR_BGR2RGB)
                       for p in manifest["path"]], dtype="float32") / 255.0
    lb = LabelBinarizer().fit(CLASSES)
    labels = lb.transform(manifest["dx"])
    return images, labels, manifest["lesion_id"].to_numpy(), lb


def make_augmenter(seed):
    """Random rotation (+/-25 deg), shift (10%), shear, zoom (20%) and horizontal flip."""
    return keras.Sequential([
        layers.RandomRotation(25 / 360, fill_mode="nearest", seed=seed),
        layers.RandomTranslation(0.1, 0.1, fill_mode="nearest", seed=seed + 1),
        layers.RandomShear(x_factor=0.2, y_factor=0.2, fill_mode="nearest", seed=seed + 2),
        layers.RandomZoom(0.2, 0.2, fill_mode="nearest", seed=seed + 3),
        layers.RandomFlip("horizontal", seed=seed + 4),
    ], name="augmentation")


class AugmentedBatches(keras.utils.PyDataset):
    """One pass over the training images per epoch, each batch randomly transformed."""

    def __init__(self, x, y, batch_size, seed):
        super().__init__()
        self.x, self.y, self.batch_size = x, y, batch_size
        self.augment = make_augmenter(seed)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(x))

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        x = keras.ops.convert_to_numpy(self.augment(self.x[idx], training=True))
        return x, self.y[idx]

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.x))


def train_model(x_train, y_train, x_val, y_val, epochs, batch_size, seed, verbose=2):
    keras.utils.set_random_seed(seed)
    model = build_cnn(IMAGE_SIZE, IMAGE_SIZE, 3, len(CLASSES))
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=INIT_LR),
                  metrics=["accuracy"])
    # Learning rate 0.001 -> 0.00001 (Table 3). Driven by the training loss so the
    # test split is never used to make training decisions.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, min_lr=MIN_LR, verbose=0)
    history = model.fit(
        AugmentedBatches(x_train, y_train, batch_size, seed),
        validation_data=(x_val, y_val),
        epochs=epochs, callbacks=[reduce_lr], verbose=verbose)
    return model, history.history


def split(images, labels, groups, leak_free, seed):
    idx = np.arange(len(images))
    if leak_free:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        return next(splitter.split(idx, groups=groups))
    return train_test_split(idx, test_size=TEST_SIZE, random_state=seed)


def plot_history(history, path):
    """Fig. 5: training loss and accuracy."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["loss"], label="train_loss")
    ax.plot(epochs, history["val_loss"], label="val_loss")
    ax.plot(epochs, history["accuracy"], label="train_acc")
    ax.plot(epochs, history["val_accuracy"], label="val_acc")
    ax.set_title("Training Loss and Accuracy")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss/Accuracy")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_holdout(images, labels, groups, lb, args, out_dir):
    train_idx, test_idx = split(images, labels, groups, args.leak_free, args.seed)
    x_train, y_train, x_test, y_test = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]
    print(f"train {len(x_train)} / test {len(x_test)}")

    model, history = train_model(x_train, y_train, x_test, y_test, args.epochs, args.batch_size, args.seed)
    model.save(out_dir / "customised_cnn.keras")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index_label="epoch")
    plot_history(history, out_dir / "training_history.png")

    # Table 6: accuracy and loss on the (non-augmented) train and test sets.
    train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=args.batch_size, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
    table6 = {
        "train_accuracy": train_acc, "test_accuracy": test_acc,
        "train_loss": train_loss, "test_loss": test_loss,
        "last_epoch_train_accuracy_augmented": history["accuracy"][-1],
        "last_epoch_train_loss_augmented": history["loss"][-1],
    }
    print("\nTable 6 - CNN model training and validation")
    print(f"  Accuracy (%)  train {train_acc * 100:.2f}   test {test_acc * 100:.2f}")
    print(f"  Loss          train {train_loss:.4f}   test {test_loss:.4f}")

    # Table 7: multi-class classification report on the test set.
    pred = model.predict(x_test, batch_size=args.batch_size, verbose=0).argmax(axis=1)
    true = y_test.argmax(axis=1)
    names = [f"{i}-{CLASS_NAMES[c]}" for i, c in enumerate(lb.classes_)]
    report = classification_report(true, pred, labels=range(len(CLASSES)), target_names=names, digits=2, zero_division=0)
    print("\nTable 7 - multi-class classification report of the customised CNN")
    print(report)
    cm = confusion_matrix(true, pred, labels=range(len(CLASSES)))
    plot_confusion(cm, out_dir / "confusion_matrix.png")

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)
    with open(out_dir / "table6.json", "w") as f:
        json.dump(table6, f, indent=2)


def run_kfold(images, labels, groups, args, out_dir):
    y = labels.argmax(axis=1)
    if args.leak_free:
        folds = GroupKFold(n_splits=args.folds).split(images, y, groups)
    else:
        folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(images, y)
    accs = []
    for k, (train_idx, val_idx) in enumerate(folds, 1):
        model, _ = train_model(images[train_idx], labels[train_idx], images[val_idx], labels[val_idx],
                               args.epochs, args.batch_size, args.seed + k, verbose=0)
        _, acc = model.evaluate(images[val_idx], labels[val_idx], batch_size=args.batch_size, verbose=0)
        accs.append(acc)
        print(f"  fold {k:2d}/{args.folds}: accuracy {acc * 100:.2f}%")
        keras.backend.clear_session()
    print(f"{args.folds}-fold CV accuracy: {np.mean(accs) * 100:.2f}% (+/- {np.std(accs) * 100:.2f})")
    with open(out_dir / f"kfold_{args.folds}.json", "w") as f:
        json.dump({"fold_accuracy": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", choices=["original", "augmented"], default="augmented")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=0, help="also run K-fold CV (e.g. 10)")
    parser.add_argument("--leak-free", action="store_true",
                        help="group-aware split by lesion_id (not in the paper)")
    args = parser.parse_args()

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Keras {keras.__version__} ({keras.backend.backend()} backend) on {device}")
    out_dir = OUTPUT_DIR / ("cnn_leak_free" if args.leak_free else "cnn")
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels, groups, lb = load_data(args.subset)
    print(f"loaded {len(images)} images of shape {images.shape[1:]} from the '{args.subset}' subset")

    run_holdout(images, labels, groups, lb, args, out_dir)
    if args.folds > 1:
        run_kfold(images, labels, groups, args, out_dir)
    print(f"\nresults written to {out_dir}")


if __name__ == "__main__":
    main()
