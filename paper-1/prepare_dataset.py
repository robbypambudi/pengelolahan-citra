"""Pre-processing steps 1 and 3 of the paper.

Step 1 - ordering the dataset: HAM10000 images are stored out of order, so they
are sorted into one folder per diagnosis using the `image_id` and `dx` columns
of the metadata. Because the smallest class (df) has only 115 images, 100 images
are chosen per class -> 100*7 = 700 images.

Step 3 - data augmentation: every selected image is horizontally flipped, giving
200 images per class -> 200*7 = 1400 images (Fig. 3).

Output:
    data/original/<dx>/<image_id>.jpg            (700 images)
    data/augmented/<dx>/<image_id>[_hflip].jpg   (1400 images)
    data/original_manifest.csv, data/augmented_manifest.csv
"""
import argparse
import shutil

import cv2
import pandas as pd

from common import CLASSES, DATA_DIR, IMAGES_PER_CLASS, METADATA_CSV, RAW_DIR, SEED, manifest_path


def select_subset(metadata: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Randomly pick `per_class` images of every diagnosis."""
    parts = []
    for dx in CLASSES:
        rows = metadata[metadata["dx"] == dx].sort_values("image_id")
        parts.append(rows.sample(n=per_class, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=IMAGES_PER_CLASS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    metadata = pd.read_csv(METADATA_CSV)
    print("HAM10000 class distribution:")
    print(metadata["dx"].value_counts().reindex(CLASSES).to_string(), "\n")

    subset = select_subset(metadata, args.per_class, args.seed)

    for name in ("original", "augmented"):
        shutil.rmtree(DATA_DIR / name, ignore_errors=True)
        for dx in CLASSES:
            (DATA_DIR / name / dx).mkdir(parents=True, exist_ok=True)

    original_rows, augmented_rows = [], []
    for row in subset.itertuples(index=False):
        src = RAW_DIR / f"{row.image_id}.jpg"
        image = cv2.imread(str(src))
        if image is None:
            raise IOError(f"could not read {src}")
        meta = {"image_id": row.image_id, "lesion_id": row.lesion_id, "dx": row.dx}

        # Step 1: ordered copy of the original image.
        dst = DATA_DIR / "original" / row.dx / f"{row.image_id}.jpg"
        shutil.copyfile(src, dst)
        original_rows.append({**meta, "path": str(dst.relative_to(DATA_DIR)), "flipped": False})

        # Step 3: original + horizontally flipped copy.
        dst = DATA_DIR / "augmented" / row.dx / f"{row.image_id}.jpg"
        shutil.copyfile(src, dst)
        augmented_rows.append({**meta, "path": str(dst.relative_to(DATA_DIR)), "flipped": False})
        dst = DATA_DIR / "augmented" / row.dx / f"{row.image_id}_hflip.jpg"
        cv2.imwrite(str(dst), cv2.flip(image, 1))
        augmented_rows.append({**meta, "path": str(dst.relative_to(DATA_DIR)), "flipped": True})

    pd.DataFrame(original_rows).to_csv(manifest_path("original"), index=False)
    pd.DataFrame(augmented_rows).to_csv(manifest_path("augmented"), index=False)
    print(f"original : {len(original_rows)} images ({args.per_class} per class)")
    print(f"augmented: {len(augmented_rows)} images ({2 * args.per_class} per class)")
    print(f"written to {DATA_DIR}")


if __name__ == "__main__":
    main()
