"""Replikasi pipeline ML Shetty dkk. (2022) — protokol L0 (persis paper, termasuk kebocorannya).

Alur per seed subset:
  100 citra/kelas -> resize 220 -> fitur global (532) [+ salinan flip horizontal]
  -> MinMaxScaler (fit semua data) -> split 80:20 tak berstrata
  -> 10-fold CV pada porsi latih (= angka Tabel 4) + akurasi pada 20% uji (tambahan)

Jalankan dari root proyek:  uv run python tugas/pcd-lesi-kulit/eksperimen/00-replikasi-shetty-L0/ml.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from citra import folder_run, seed_semua, simpan_metrik
from citra.data import KELAS, ROOT, muat_ham10000
from citra.fitur import fitur_global

FOLDER = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "olahan" / "fitur_shetty_220.npz"

# Angka paper (Tabel 4) untuk dibandingkan
PAPER_TABEL4 = {
    "tanpa_aug": {"LR": 48.929, "LDA": 35.179, "KNN": 41.786, "DT": 44.286, "RF": 58.571, "NB": 34.107, "SVM": 42.143},
    "flip": {"LR": 58.125, "LDA": 57.589, "KNN": 48.393, "DT": 68.660, "RF": 87.321, "NB": 36.339, "SVM": 53.125},
}


def model_tabel1():
    """Klasifier & hiperparameter sesuai Tabel 1 paper."""
    return {
        "LR": LogisticRegression(random_state=9),
        "LDA": LinearDiscriminantAnalysis(solver="svd"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DT": DecisionTreeClassifier(random_state=9),  # "Estimators = 100" bukan parameter DT -> diabaikan
        "RF": RandomForestClassifier(n_estimators=200, random_state=0),
        "NB": GaussianNB(var_smoothing=1e-9),
        "SVM": SVC(kernel="linear", C=1, random_state=0),
    }


def _fitur_satu(path, ukuran, bins):
    img = cv2.resize(cv2.imread(str(path)), (ukuran, ukuran))
    return fitur_global(img, bins), fitur_global(cv2.flip(img, 1), bins)


def fitur_semua(df, ukuran, bins):
    """Fitur asli & flip untuk seluruh 10.015 citra, di-cache agar setiap seed subset cukup mengindeks."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["image"]) == list(df["image"]):
            return z["asli"], z["flip"]
    hasil = Parallel(n_jobs=-1, verbose=5)(delayed(_fitur_satu)(p, ukuran, bins) for p in df["path"])
    asli, flip = map(np.array, zip(*hasil))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, image=df["image"].values, asli=asli, flip=flip)
    return asli, flip


def satu_seed(df, asli, flip, seed, cfg, augment):
    c = cfg["ml"]
    idx = (df.groupby("dx", group_keys=False)
             .apply(lambda g: g.sample(cfg["subset"]["per_kelas"], random_state=seed)).index.values)
    X, y = asli[idx], df.loc[idx, "dx"].values
    asal = idx.copy()  # id citra asal, untuk mengukur kebocoran pasangan flip
    if augment:
        X, y, asal = np.vstack([X, flip[idx]]), np.concatenate([y, y]), np.concatenate([idx, idx])

    if c["minmax_sebelum_split"]:
        X = MinMaxScaler().fit_transform(X)
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        X, y, asal, test_size=c["test_size"], random_state=c["split_seed"], stratify=y if c["stratify"] else None)

    kfold = KFold(n_splits=c["kfold"])  # tanpa shuffle (data sudah diacak oleh train_test_split)
    baris = {}
    for nama, model in model_tabel1().items():
        cv = cross_val_score(model, Xtr, ytr, cv=kfold, scoring="accuracy")
        uji = model.fit(Xtr, ytr).score(Xte, yte)
        baris[nama] = {"cv_latih": float(cv.mean()), "uji_20": float(uji)}
    bocor = float(np.isin(ate, atr).mean())  # porsi citra uji yang "kembarannya" ada di data latih
    return baris, {"n_latih": len(ytr), "n_uji": len(yte), "uji_punya_kembaran_di_latih": bocor}


def main():
    cfg = yaml.safe_load((FOLDER / "config.yaml").read_text())
    seed_semua(0)
    run = folder_run(FOLDER, FOLDER / "config.yaml")

    df = muat_ham10000()
    assert len(df) == 10015 and sorted(df["dx"].unique()) == KELAS, "Data HAM10000 tidak lengkap"
    asli, flip = fitur_semua(df, cfg["ml"]["ukuran"], cfg["ml"]["bins_hsv"])

    rekap, rows = {}, []
    for augment, label in ((False, "tanpa_aug"), (True, "flip")):
        for seed in cfg["subset"]["seed_list"]:
            hasil, info = satu_seed(df, asli, flip, seed, cfg, augment)
            for m, v in hasil.items():
                rows.append({"protokol": label, "seed": seed, "model": m, **v, **info})
    tab = pd.DataFrame(rows)
    tab.to_csv(run / "per_seed.csv", index=False)

    for label in ("tanpa_aug", "flip"):
        t = tab[tab.protokol == label]
        rekap[label] = {
            m: {
                "paper_tabel4": PAPER_TABEL4[label][m],
                "cv_latih_rata": round(100 * g.cv_latih.mean(), 2),
                "cv_latih_sd": round(100 * g.cv_latih.std(), 2),
                "cv_latih_min_maks": [round(100 * g.cv_latih.min(), 2), round(100 * g.cv_latih.max(), 2)],
                "uji_20_rata": round(100 * g.uji_20.mean(), 2),
            }
            for m, g in t.groupby("model", sort=False)
        }
        rekap[label + "_kebocoran_uji"] = round(float(t.uji_punya_kembaran_di_latih.mean()), 3)

    simpan_metrik(run, {"nama": cfg["nama"], "bagian": "ml", "n_seed_subset": len(cfg["subset"]["seed_list"]), "hasil": rekap})
    print(f"Hasil: {run}")
    for label in ("tanpa_aug", "flip"):
        print(f"\n== {label}  (porsi uji yang kembarannya di data latih: {rekap[label + '_kebocoran_uji']:.1%})")
        print(f"{'model':<6}{'paper':>8}{'CV latih':>16}{'rentang':>18}{'uji 20%':>10}")
        for m, v in rekap[label].items():
            lo, hi = v["cv_latih_min_maks"]
            print(f"{m:<6}{v['paper_tabel4']:>8.2f}{v['cv_latih_rata']:>9.2f} ± {v['cv_latih_sd']:<5.2f}{lo:>8.2f}–{hi:<8.2f}{v['uji_20_rata']:>9.2f}")


if __name__ == "__main__":
    main()
