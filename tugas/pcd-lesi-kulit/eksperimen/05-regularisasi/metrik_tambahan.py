"""Metrik tambahan dari prediksi yang sudah tersimpan (tanpa melatih ulang).

Untuk setiap varian × fold × keputusan (argmax, bias_val), dihitung dari per_run/*_preds.csv:
  akurasi, presisi makro, recall makro (= BACC), F1 makro, F1 berbobot, F1 per kelas,
  Cohen's kappa, MCC, Brier score multi-kelas, MSE probabilitas (= Brier / K), log loss.

Probabilitas +bias_val = terapkan_bias(p, bias_val) dengan bias yang tersimpan di per_run/*.json.
Sebagai cek, BACC dan F1 makro hasil hitung ulang harus sama dengan metrics.json run tersebut.

Pakai:
  uv run python tugas/pcd-lesi-kulit/eksperimen/05-regularisasi/metrik_tambahan.py <folder-run>
Keluaran: <folder-run>/metrik_tambahan.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             log_loss, matthews_corrcoef, precision_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from citra.imbalance import terapkan_bias  # noqa: E402

KELAS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
K = len(KELAS)


def metrik(y, p):
    pred = p.argmax(1)
    lbl = list(range(K))
    satu = np.eye(K)[y]
    brier = float(((p - satu) ** 2).sum(1).mean())
    return {
        "akurasi": float(accuracy_score(y, pred)),
        "presisi_makro": float(precision_score(y, pred, average="macro", labels=lbl, zero_division=0)),
        "recall_makro": float(balanced_accuracy_score(y, pred)),
        "f1_makro": float(f1_score(y, pred, average="macro", labels=lbl, zero_division=0)),
        "f1_berbobot": float(f1_score(y, pred, average="weighted", labels=lbl, zero_division=0)),
        "kappa": float(cohen_kappa_score(y, pred, labels=lbl)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "brier": brier,
        "mse_prob": brier / K,
        "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1), labels=lbl)),
        "f1_per_kelas": dict(zip(KELAS, f1_score(y, pred, average=None, labels=lbl, zero_division=0).round(4).tolist())),
    }


def main(run):
    run = Path(run)
    per = {}
    for f_json in sorted((run / "per_run").glob("*_r*_f*.json")):
        info = json.loads(f_json.read_text())
        df = pd.read_csv(f_json.with_name(f_json.stem + "_preds.csv"))
        y = df["dx"].map(KELAS.index).to_numpy()
        p = df[[f"p_{k}" for k in KELAS]].to_numpy(dtype=np.float64)
        p = p / p.sum(1, keepdims=True)          # CSV float32: jumlah ≈ 1, dinormalkan ulang
        for kep, prob in (("argmax", p), ("bias_val", terapkan_bias(p, np.asarray(info["bias_val"])))):
            h = metrik(y, prob)
            asli = info["keputusan"][kep]["uji"]
            h["selisih_cek_bacc"] = abs(h["recall_makro"] - asli["bacc"])
            h["selisih_cek_f1"] = abs(h["f1_makro"] - asli["f1_makro"])
            per.setdefault(f"{info['varian']}|{kep}", []).append(h)
    ringkasan = {}
    for kunci, daftar in per.items():
        r = {}
        for k in daftar[0]:
            if k == "f1_per_kelas":
                r[k] = {c: float(np.mean([d[k][c] for d in daftar])) for c in KELAS}
            else:
                v = np.array([d[k] for d in daftar])
                r[k] = {"rata": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        r["n_fold"] = len(daftar)
        ringkasan[kunci] = r
    cek = max(max(d["selisih_cek_bacc"], d["selisih_cek_f1"]) for ds in per.values() for d in ds)
    keluar = {
        "sumber_run": run.name,
        "catatan": "Dihitung ulang dari per_run/*_preds.csv; rata-rata ± SD antar fold uji. "
                   "brier = mean_i Σ_k (p_ik − y_ik)²; mse_prob = brier / K. "
                   "+bias_val memakai bias tersimpan (dibulatkan 2 desimal).",
        "selisih_maks_vs_metrics_json": cek,
        "ringkasan": ringkasan,
    }
    (run / "metrik_tambahan.json").write_text(json.dumps(keluar, indent=2, ensure_ascii=False))
    print(f"{run / 'metrik_tambahan.json'} · selisih maks vs metrics.json = {cek:.2e}")
    for kunci, r in ringkasan.items():
        print(f"{kunci:24s} F1 {r['f1_makro']['rata']:.4f}  presisi {r['presisi_makro']['rata']:.4f}  "
              f"MCC {r['mcc']['rata']:.4f}  Brier {r['brier']['rata']:.4f}  MSE {r['mse_prob']['rata']:.4f}  "
              f"logloss {r['log_loss']['rata']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
