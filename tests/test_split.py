import numpy as np
import pandas as pd

from citra.split import buat_fold_grup, cek_kebocoran, sub_validasi


def _df_sintetis(n_lesi=400, seed=0):
    rng = np.random.default_rng(seed)
    kelas = rng.choice(["nv", "mel", "bkl", "bcc"], size=n_lesi, p=[0.6, 0.2, 0.15, 0.05])
    baris = []
    for i, k in enumerate(kelas):
        for _ in range(rng.integers(1, 4)):          # 1–3 citra per lesi
            baris.append({"lesion_id": f"L{i}", "dx": k})
    return pd.DataFrame(baris)


def test_tidak_ada_lesi_di_dua_fold():
    df = _df_sintetis()
    fold = buat_fold_grup(df, n_fold=5, seed=0)
    assert set(fold) == set(range(5))
    assert cek_kebocoran(df, fold) == 0


def test_proporsi_kelas_per_fold_mirip():
    df = _df_sintetis()
    fold = buat_fold_grup(df, n_fold=5, seed=1)
    global_ = df["dx"].value_counts(normalize=True)
    for f in range(5):
        lokal = df.loc[fold == f, "dx"].value_counts(normalize=True)
        assert (lokal.reindex(global_.index, fill_value=0) - global_).abs().max() < 0.06


def test_validasi_dalam_tanpa_kebocoran():
    df = _df_sintetis()
    fold = buat_fold_grup(df, n_fold=5, seed=0)
    idx_latih = np.flatnonzero(fold != 0)
    lat, val = sub_validasi(df, idx_latih, frac=0.1, seed=0)
    assert len(set(lat) & set(val)) == 0 and set(lat) | set(val) == set(idx_latih)
    assert not set(df.iloc[lat]["lesion_id"]) & set(df.iloc[val]["lesion_id"])
    assert 0.05 < len(val) / len(idx_latih) < 0.16
