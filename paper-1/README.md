# Paper 1 — reproduction

**Shetty, B. et al. "Skin lesion classification of dermoscopic images using machine learning and
convolutional neural network." *Scientific Reports* 12, 18134 (2022).**
https://doi.org/10.1038/s41598-022-22644-9 (PDF: `s41598-022-22644-9.pdf`)

## What the paper does

| Step | Paper | Code |
|---|---|---|
| Step 1: ordering the dataset | sort HAM10000 into one folder per `dx`; take **100 images per class** (df only has 115) → 700 images | `prepare_dataset.py` |
| Step 3: data augmentation (ML) | **horizontal flip** of every image → 200 per class → 1400 images (Fig. 3) | `prepare_dataset.py` |
| Step 2: image resizing | 220×220 for the ML models, 96×96×3 scaled to [0,1] for the CNN, `LabelBinarizer` labels | `ml_pipeline.py`, `train_cnn.py` |
| Feature extraction | global descriptors: colour histogram (HSV), Haralick texture, Hu moments → concatenated, saved to HDF5 | `features.py`, `ml_pipeline.py` |
| Data split | 80% train / 20% test (560/140 and 1120/280) | both pipelines |
| ML classifiers | LR, LDA, KNN, DT, RF, NB, SVM with the Table 1 hyperparameters, 10-fold CV | `ml_pipeline.py` |
| CNN | Table 2 architecture, Table 3 hyperparameters (Adam, categorical cross-entropy, 150 epochs, batch 32, LR 1e-3 → 1e-5), random rotation/shift/shear/zoom/flip during training | `cnn_model.py`, `train_cnn.py` |
| InceptionV3 comparison | pretrained InceptionV3 (reported 95.14%) | `train_inceptionv3.py` |

## Setup (uv)

`paper-1` is its own uv project (Keras 3 on the PyTorch backend; the CUDA 12.8 torch wheels support the RTX 50xx GPUs):

```bash
cd paper-1
uv sync
```

The HAM10000 images and `HAM10000_metadata` are read from `../dataverse_files/`.

## Running

```bash
uv run python prepare_dataset.py          # data/original (700) + data/augmented (1400)
uv run python ml_pipeline.py              # Tables 4 and 5 (ML), outputs/ml/
uv run python train_cnn.py                # Tables 6 and 7, Fig. 5, outputs/cnn/
uv run python train_cnn.py --folds 10     # also 10-fold CV of the CNN
uv run python train_inceptionv3.py        # InceptionV3 comparison, outputs/inceptionv3/
```

Every training script accepts `--leak-free`, which keeps all images of a lesion (and so an image and its
flipped copy) on the same side of every split or fold. The paper does not do this; see below.

## Results

Single runs on an RTX 5090. "Leak-free" means the split was grouped by `lesion_id`.

### Table 4: ML models, mean 10-fold CV accuracy on the training split

| | LR | LDA | KNN | DT | RF | NB | SVM |
|---|---|---|---|---|---|---|---|
| Paper, without augmentation (100×7) | 0.489 | 0.352 | 0.418 | 0.443 | 0.586 | 0.341 | 0.421 |
| **Ours**, without augmentation | 0.380 | 0.321 | 0.311 | 0.396 | 0.514 | 0.300 | 0.384 |
| Paper, with horizontal flip (200×7) | 0.581 | 0.576 | 0.484 | 0.687 | 0.873 | 0.363 | 0.531 |
| **Ours**, with horizontal flip | 0.501 | 0.544 | 0.421 | 0.679 | 0.857 | 0.361 | 0.510 |
| Ours, with horizontal flip, leak-free | 0.396 | 0.343 | 0.301 | 0.353 | 0.479 | 0.319 | 0.371 |

The model ranking and the large jump from flip augmentation both reproduce (RF best, then DT; NB worst).
The leak-free row shows that the jump comes from leakage (see caveats).

### Table 5: test split (280 images), weighted averages, in %

| | CNN | RF | DT | LR | LDA | SVM | KNN | NB |
|---|---|---|---|---|---|---|---|---|
| Paper, accuracy | 94* | 87* | 68* | 58* | 57* | 53* | 48* | 36* |
| **Ours**, accuracy | 88 | 91 | 73 | 53 | 55 | 54 | 45 | 35 |
| **Ours**, F1 | 87 | 91 | 73 | 53 | 56 | 54 | 46 | 34 |

\* For the ML models the paper's Table 5 accuracy simply repeats the Table 4 CV accuracy.

### Tables 6 and 7: customised CNN (1400 images, 1120 train / 280 test, 150 epochs)

| | Paper | Ours | Ours, leak-free |
|---|---|---|---|
| Train accuracy | 95.18% | 98.93% (93.75% on augmented batches, last epoch) | 92.44% |
| **Test accuracy** | **86.43%** | **87.50%** | **47.10%** |
| Train loss / test loss | 0.148 / 0.469 | 0.050 / 0.694 | 0.280 / 3.698 |
| Weighted P / R / F1 (Table 7) | 0.88 / 0.85 / 0.86 | 0.88 / 0.88 / 0.87 | 0.59 / 0.47 / 0.50 |

`outputs/cnn/training_history.png` reproduces Fig. 5, including the large `val_loss` spikes.
`outputs/cnn/classification_report.txt` holds the per-class report (Table 7).

## Implementation notes and assumptions

The paper leaves some details out. Where it does, these choices were made:

- **Subset selection.** The paper doesn't say which 100 images per class were used, so they are sampled at random (`random_state=9`).
- **Global features.** HSV 3-D histogram with 8 bins per channel (512 values, `cv2.normalize`), Haralick features from `mahotas` averaged over the 4 directions (13), and Hu moments (7). That gives a 532-dim vector, scaled with `MinMaxScaler`. This is the common "global feature descriptor" recipe that the paper's description and its `random_state=9` setting point to.
- **Table 4.** These numbers are the mean 10-fold CV accuracy on the training split. For example, 0.48929 × 560 = 274 and 0.58125 × 1120 = 651, which fits CV over 560 and 1120 training images.
- **Table 5.** Weighted precision, recall and F1 are computed on the 20% test split.
- **Table 1 "DT Estimators = 100"** isn't a parameter of a decision tree, so `DecisionTreeClassifier(random_state=9)` is used. LR uses `max_iter=1000` so it converges.
- **CNN data.** The paper says "each class has 200 images", so the CNN is trained on the same 1400-image set (originals + flips). An 80/20 split gives 280 test images, which matches the support column of Table 7.
- **Learning rate "0.001–0.00001".** Implemented as `ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, min_lr=1e-5)`. It watches the training loss, so the test split never affects training.
- **Framework.** The paper used TensorFlow/Keras 2 with `ImageDataGenerator`. TensorFlow has no kernels for compute capability 12.0 (RTX 5090), and Keras 3 removed `ImageDataGenerator`. So the same Keras model runs on the PyTorch backend, and the augmentation is done with the Keras `RandomRotation(±25°)`, `RandomTranslation(0.1)`, `RandomShear(0.2)`, `RandomZoom(0.2)` and `RandomFlip` layers.
- **InceptionV3.** The paper gives no training details. A frozen ImageNet backbone with a new softmax head is trained first, then fine-tuned from `mixed9` at 1e-5.

## Caveats about the paper's evaluation

1. **The headline 95.18% is the CNN's training accuracy.** Table 6 reports 95.18% train and **86.43% test**. Table 7's weighted recall (0.85) agrees with roughly 85–86% test accuracy.
2. **Data leakage from the flip augmentation.** Flipping is done *before* the train/test split and before CV, so most test images have their mirror image in the training set. The global features barely change under a horizontal flip: the colour histogram is identical, and Haralick averaged over 4 directions and the Hu moments are nearly identical. In this reproduction an image's feature vector is about 14× closer to its flipped copy than to any other image. This is why "augmentation" raises RF from about 0.5 to about 0.86. With `--leak-free` the gain disappears (see the table above). The CNN is affected the same way: its test accuracy falls from 87.5% to 47.1% once the mirrored and same-lesion images are kept out of the test set.
3. **Several images per lesion.** HAM10000 often has more than one image of the same lesion (`lesion_id`). A random split puts them on both sides, which adds a smaller amount of leakage. `--leak-free` groups by `lesion_id`.
4. **Small balanced subset.** Only 700 of the 10015 images are used and the classes are balanced. So the numbers are not comparable with the full-HAM10000 results in Table 8.
