"""Eksperimen 05 — cara latih, bukan arsitektur: regularisasi pada A0 (MobileNetV3-Large polos).

Arsitektur tetap A0. Yang berubah per varian hanya regime pelatihan:
  label smoothing, mixup, cutmix, stochastic depth, epoch lebih pendek, tukar latar.
Alur lain (cache, split per lesi, S6-logit-adj, keputusan argmax/+bias_val, metrik, statistik)
memakai ulang ../03-derm-v2/run.py. Fungsi latih di bawah = eksperimen 02 (`latih`) + regime.
Tanpa regime, hasilnya identik dengan A0 eksperimen 02–04 (cek konsistensi).

Jalankan dari root proyek:
  uv run python tugas/pcd-lesi-kulit/eksperimen/05-regularisasi/run.py [--config config-cepat.yaml] [--lanjut hasil/<waktu>]
"""

import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from citra.imbalance import hitung_kelas
from citra.regularisasi import cutmix_batch, mixup_batch, pasang_stochastic_depth, tukar_latar

FOLDER = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("eks03", FOLDER.parent / "03-derm-v2" / "run.py")
eks03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eks03)
eks01, DEV, K = eks03.eks01, eks03.eks01.DEV, eks03.K


def buat_model(v, cm):
    model = eks03.buat_model(v, cm)
    sd = v.get("regime", {}).get("stochastic_depth")
    return pasang_stochastic_depth(model, sd) if sd else model


def latih(model, X, M, y, idx_lat, idx_val, cfg, strategi, seed, varian):
    rg = varian.get("regime", {})
    cl, ukuran = {**cfg["latih"], **({"epoch": rg["epoch"]} if "epoch" in rg else {})}, cfg["data"]["ukuran_input"]
    if strategi.get("loss") != "logit_adjusted":
        raise ValueError("eksperimen 05 hanya untuk S6-logit-adj")
    gen = torch.Generator().manual_seed(seed)
    gen_rg = torch.Generator().manual_seed(seed + 7)     # generator terpisah: urutan batch & augmentasi tetap sama dengan A0
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.kelompok_parameter(cl["lr_dasar"], cl["lr_modul"], cl["weight_decay"]),
                            weight_decay=cl["weight_decay"])
    langkah_per_epoch = math.ceil(len(idx_lat) / cl["batch"])
    total, warmup = cl["epoch"] * langkah_per_epoch, cl["warmup_epoch"] * langkah_per_epoch
    jadwal = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    prior = hitung_kelas(y[idx_lat], K) / len(idx_lat)
    log_prior = torch.tensor(np.log(prior), dtype=torch.float32, device=DEV)
    tau, ls = strategi.get("tau", 1.0), rg.get("label_smoothing", 0.0)
    y_t = torch.from_numpy(y).to(DEV)

    def kerugian(logit, yb):
        return F.cross_entropy(logit + tau * log_prior, yb, label_smoothing=ls)

    riwayat = []
    for ep in range(cl["epoch"]):
        model.train()
        urut = idx_lat[torch.randperm(len(idx_lat), generator=gen).numpy()]
        t0, total_loss = time.time(), 0.0
        for i in range(0, len(urut), cl["batch"]):
            b = urut[i:i + cl["batch"]]
            x, m = eks01.augmentasi(*eks01.ke_tensor(X, M, b), cfg["augmentasi"], ukuran, gen)
            yb, perm, lam = y_t[b], None, 1.0
            if rg.get("tukar_latar"):
                x = tukar_latar(x, m, gen_rg, p=rg["tukar_latar"], dilasi_px=rg.get("dilasi_px", 16))
            if rg.get("mixup"):
                x, perm, lam = mixup_batch(x, gen_rg, rg["mixup"])
            elif rg.get("cutmix"):
                x, perm, lam = cutmix_batch(x, gen_rg, rg["cutmix"])
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=cl["amp"]):
                logit = model(x)
            logit = logit.float()
            loss = kerugian(logit, yb) if perm is None else lam * kerugian(logit, yb) + (1 - lam) * kerugian(logit, yb[perm])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            jadwal.step()
            total_loss += loss.item() * len(b)
        bacc_val = balanced_accuracy_score(y[idx_val], eks01.prediksi(model, X, M, idx_val, cfg).argmax(1))
        riwayat.append({"epoch": ep + 1, "loss": total_loss / len(urut), "bacc_val": float(bacc_val),
                        "detik": round(time.time() - t0, 1)})
    return model, riwayat, prior


if __name__ == "__main__":
    eks03.main(folder=FOLDER, pembuat=buat_model, pelatih=latih)
