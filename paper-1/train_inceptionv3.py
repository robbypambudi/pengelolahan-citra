"""InceptionV3 comparison (paper section "Results and discussion": 95.14%).

The paper gives no training details for InceptionV3, so a standard transfer-
learning recipe is used on the same 200*7 images and the same 80/20 split as
the customised CNN:
  1. ImageNet-pretrained InceptionV3 (299x299), frozen, with a new
     GlobalAveragePooling -> Dropout -> Dense(7, softmax) head, trained at 1e-3;
  2. the last two Inception blocks (from `mixed9`) are unfrozen and fine-tuned
     at 1e-5.
"""
import argparse
import json
import time

from sklearn.metrics import classification_report

from train_cnn import BATCH_SIZE, AugmentedBatches, load_data, split  # sets KERAS_BACKEND first
import keras
from keras import layers

from common import CLASS_NAMES, CLASSES, OUTPUT_DIR

IMAGE_SIZE = 299


def build_inception(classes):
    base = keras.applications.InceptionV3(include_top=False, weights="imagenet",
                                          input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.trainable = False
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)      # [0, 1] -> [-1, 1] (InceptionV3 preprocessing)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="inceptionv3_transfer"), base


def fit(model, x_train, y_train, x_test, y_test, lr, epochs, seed):
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=["accuracy"])
    model.fit(AugmentedBatches(x_train, y_train, BATCH_SIZE, seed),
              validation_data=(x_test, y_test), epochs=epochs, verbose=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leak-free", action="store_true")
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / ("inceptionv3_leak_free" if args.leak_free else "inceptionv3")
    out_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(args.seed)

    images, labels, groups, lb = load_data("augmented", IMAGE_SIZE)
    train_idx, test_idx = split(images, labels, groups, args.leak_free, args.seed)
    x_train, y_train, x_test, y_test = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

    start = time.time()
    model, base = build_inception(len(CLASSES))
    fit(model, x_train, y_train, x_test, y_test, 1e-3, args.head_epochs, args.seed)

    base.trainable = True
    for layer in base.layers[: base.layers.index(base.get_layer("mixed9"))]:
        layer.trainable = False
    fit(model, x_train, y_train, x_test, y_test, 1e-5, args.finetune_epochs, args.seed)
    elapsed = time.time() - start

    _, train_acc = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    pred = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    names = [f"{i}-{CLASS_NAMES[c]}" for i, c in enumerate(lb.classes_)]
    report = classification_report(y_test.argmax(axis=1), pred, labels=range(len(CLASSES)),
                                   target_names=names, zero_division=0)
    print(f"\nInceptionV3  train acc {train_acc * 100:.2f}%  test acc {test_acc * 100:.2f}%  "
          f"training time {elapsed / 60:.1f} min")
    print(report)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"train_accuracy": train_acc, "test_accuracy": test_acc, "training_seconds": elapsed}, f, indent=2)


if __name__ == "__main__":
    main()
