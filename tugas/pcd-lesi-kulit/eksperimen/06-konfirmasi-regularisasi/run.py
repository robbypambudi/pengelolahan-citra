"""Eksperimen 06 — konfirmasi temuan argmax eksperimen 05 pada split baru (seed ulangan 1 dan 2).

Pelatihan dan evaluasi = eksperimen 05 (buat_model + latih dengan regime), alur = ../03-derm-v2/run.py.
Setelah semua run selesai, uji konfirmasi yang dibekukan di README ditambahkan ke metrics.json
dengan kunci "konfirmasi":
  H1 (primer)   : X|argmax − A0|argmax,   X ∈ primer,   Holm atas |primer|
  H2 (sekunder) : X|argmax − A0|bias_val, X ∈ sekunder, Holm atas |sekunder|
  eksploratif   : kedua perbandingan di atas, tanpa koreksi lintas keluarga dan tanpa klaim
Didukung bila selisih rata-rata ≥ SESOI dan p_holm < alpha (corrected resampled t-test, Nadeau & Bengio, 2003).

Jalankan dari root proyek:
  uv run python tugas/pcd-lesi-kulit/eksperimen/06-konfirmasi-regularisasi/run.py [--config config-cepat.yaml] [--lanjut hasil/<waktu>]
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd

FOLDER = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("eks05", FOLDER.parent / "05-regularisasi" / "run.py")
eks05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eks05)
eks03, eks01 = eks05.eks03, eks05.eks01


def uji(df, pasangan, rasio):
    """pasangan: [(varian_a, keputusan_a, varian_b, keputusan_b)] → selisih per fold (ulangan, fold) + t-test."""
    out = {}
    for a, ka, b, kb in pasangan:
        ga = df[(df.varian == a) & (df.keputusan == ka)].set_index(["ulangan", "fold"])["bacc"]
        gb = df[(df.varian == b) & (df.keputusan == kb)].set_index(["ulangan", "fold"])["bacc"]
        d = (ga - gb).dropna()
        t, p = eks01.uji_t_terkoreksi(d.to_numpy(), rasio)
        out[f"{a}|{ka} − {b}|{kb}"] = {
            "selisih_bacc_rata": float(d.mean()), "n_fold": int(len(d)), "n_fold_positif": int((d > 0).sum()),
            "selisih_per_ulangan": {int(r): float(g.mean()) for r, g in d.groupby(level=0)},
            "selisih_per_fold": d.round(4).tolist(), "t": t, "p": p}
    return out


def keluarga(hasil, st):
    kunci = list(hasil)
    for k, ph in zip(kunci, eks01.holm([hasil[k]["p"] for k in kunci])):
        h = hasil[k]
        h["p_holm"] = float(ph)
        h["didukung"] = bool(h["selisih_bacc_rata"] >= st["sesoi_bacc"] and ph < st["alpha"])
    return hasil


def konfirmasi(run):
    m = json.loads((run / "metrics.json").read_text())
    cfg = __import__("yaml").safe_load((run / "config.yaml").read_text())
    st, kf = cfg["statistik"], cfg["statistik"]["konfirmasi"]
    acuan, rasio = st["pembanding"], m["rasio_uji_latih"]
    df = pd.read_csv(run / "ringkasan_per_fold.csv")
    h1 = keluarga(uji(df, [(x, "argmax", acuan, "argmax") for x in kf["primer"]], rasio), st)
    h2 = keluarga(uji(df, [(x, "argmax", acuan, "bias_val") for x in kf["sekunder"]], rasio), st)
    eks = uji(df, [(x, "argmax", acuan, kb) for x in kf["eksploratif"] for kb in ("argmax", "bias_val")], rasio)
    m["konfirmasi"] = {
        "aturan": "didukung bila selisih rata-rata ≥ sesoi_bacc dan p_holm < alpha; Holm per keluarga; "
                  "dibekukan di README sebelum run penuh",
        "sesoi_bacc": st["sesoi_bacc"], "alpha": st["alpha"], "seed_ulangan": cfg["split"]["seed_ulangan"],
        "H1_argmax_vs_A0_argmax": h1, "H2_argmax_vs_A0_bias_val": h2, "eksploratif": eks}
    (run / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    for nama, kel in (("H1", h1), ("H2", h2), ("eksploratif", eks)):
        print(f"\n{nama}:")
        for k, h in kel.items():
            print(f"  {k:<36}{h['selisih_bacc_rata']:>+9.4f}  {h['n_fold_positif']}/{h['n_fold']} fold +"
                  f"  p {h['p']:.3f}  p_holm {h.get('p_holm', float('nan')):.3f}"
                  + (f"  {'DIDUKUNG' if h['didukung'] else 'tidak didukung'}" if "didukung" in h else ""))


if __name__ == "__main__":
    run = eks03.main(folder=FOLDER, pembuat=eks05.buat_model, pelatih=eks05.latih)
    konfirmasi(run)
