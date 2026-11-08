"""Pembagian data berkelompok per lesi (tanpa kebocoran lesi antar-fold).

Versi split v0: grup = `lesion_id` HAM10000. Penggabungan duplikat berbasis perceptual hash
(Plan 1) BELUM diterapkan; bila sudah, cukup ganti kolom grup.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def buat_fold_grup(df, n_fold=5, seed=0, kolom_grup="lesion_id", kolom_label="dx"):
    """Nomor fold (0..n_fold-1) per baris, berstrata kelas dan berkelompok per grup."""
    sgkf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for i, (_, idx_uji) in enumerate(sgkf.split(df, df[kolom_label], df[kolom_grup])):
        fold[idx_uji] = i
    return fold


def sub_validasi(df, idx_latih, frac=0.1, seed=0, kolom_grup="lesion_id", kolom_label="dx"):
    """Pisahkan validasi dalam (≈frac) dari indeks latih, juga berkelompok per lesi.

    Mengembalikan (idx_latih_baru, idx_validasi) sebagai array indeks posisi df.
    """
    idx_latih = np.asarray(idx_latih)
    sub = df.iloc[idx_latih]
    n = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    lat, val = next(sgkf.split(sub, sub[kolom_label], sub[kolom_grup]))
    return idx_latih[lat], idx_latih[val]


def cek_kebocoran(df, fold, kolom_grup="lesion_id"):
    """Jumlah grup yang muncul di lebih dari satu fold (harus 0)."""
    per_grup = pd.Series(fold).groupby(df[kolom_grup].to_numpy()).nunique()
    return int((per_grup > 1).sum())
