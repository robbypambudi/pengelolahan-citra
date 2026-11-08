"""Pemuat HAM10000 (rilis ISIC 2018 Task 3) sebagai DataFrame: image, path, dx, lesion_id."""

from pathlib import Path

import pandas as pd

KELAS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # urutan abjad, sama seperti Tabel 7 Shetty
_KOLOM_GT = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc"}

ROOT = Path(__file__).resolve().parents[1]
FOLDER_ISIC2018 = ROOT / "data" / "mentah" / "isic2018"


def muat_ham10000(folder=FOLDER_ISIC2018):
    folder = Path(folder)
    gt = pd.read_csv(folder / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv")
    gt["dx"] = gt[list(_KOLOM_GT)].idxmax(axis=1).map(_KOLOM_GT)
    df = gt[["image", "dx"]].copy()

    grup = folder / "ISIC2018_Task3_Training_LesionGroupings.csv"
    if grup.exists():
        df = df.merge(pd.read_csv(grup)[["image", "lesion_id"]], on="image", how="left")

    df["path"] = [folder / "ISIC2018_Task3_Training_Input" / f"{i}.jpg" for i in df["image"]]
    return df.sort_values("image").reset_index(drop=True)
