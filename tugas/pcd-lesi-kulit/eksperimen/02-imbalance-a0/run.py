"""Eksperimen 02 — strategi penanganan ketidakseimbangan kelas pada A0 (MobileNetV3-Large polos).

Memakai cache, split per lesi, augmentasi, model, dan metrik yang SAMA dengan eksperimen 01
(fungsi diimpor dari ../01-derm-mobilenetv3/run.py). Yang berubah hanya cara menangani kelas
yang tidak seimbang:
  loss    : CE polos / CE berbobot (penuh, akar, efektif) / focal / logit-adjusted
  sampler : acak biasa / seimbang (hanya data latih fold)
  keputusan (pasca-latih, tanpa latih ulang):
    - koreksi prior τ=1 (untuk strategi tanpa koreksi apa pun saat latih)
    - bias per kelas yang dicari di VALIDASI DALAM, lalu diterapkan ke fold uji

Aturan pemilihan (ditetapkan sebelum run): strategi terbaik = BACC validasi dalam rata-rata
tertinggi (epoch akhir, keputusan argmax). Fold uji TIDAK dipakai untuk memilih.

Jalankan dari root proyek:
  uv run python tugas/pcd-lesi-kulit/eksperimen/02-imbalance-a0/run.py [--config config-cepat.yaml] [--lanjut hasil/<waktu>]
"""

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score

from citra import folder_run, seed_semua, simpan_metrik
from citra.data import KELAS
from citra.imbalance import (bobot_kelas, cari_bias_bacc, focal_loss, hitung_kelas, indeks_sampler_seimbang,
                             koreksi_prior, logit_adjusted_loss, terapkan_bias)
from citra.split import buat_fold_grup, cek_kebocoran, sub_validasi

FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / "pyproject.toml").exists())
EKS01 = FOLDER.parent / "01-derm-mobilenetv3"
sys.path.insert(0, str(EKS01))
_spec = importlib.util.spec_from_file_location("eks01", EKS01 / "run.py")
eks01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eks01)
from siapkan import muat_cache  # noqa: E402  (dari folder eksperimen 01)

DEV = eks01.DEV
K = len(KELAS)


def latih(model, X, M, y, idx_lat, idx_val, cfg, strategi, seed):
    cl, ukuran = cfg["latih"], cfg["data"]["ukuran_input"]
    gen = torch.Generator().manual_seed(seed)
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.kelompok_parameter(cl["lr_dasar"], cl["lr_modul"], cl["weight_decay"]),
                            weight_decay=cl["weight_decay"])
    langkah_per_epoch = math.ceil(len(idx_lat) / cl["batch"])
    total, warmup = cl["epoch"] * langkah_per_epoch, cl["warmup_epoch"] * langkah_per_epoch
    jadwal = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    w = torch.tensor(bobot_kelas(y[idx_lat], K, strategi.get("bobot", "tanpa"), strategi.get("beta", 0.999)),
                     dtype=torch.float32, device=DEV)
    prior = hitung_kelas(y[idx_lat], K) / len(idx_lat)
    log_prior = torch.tensor(np.log(prior), dtype=torch.float32, device=DEV)
    y_t = torch.from_numpy(y).to(DEV)
    loss_jenis = strategi.get("loss", "ce")

    riwayat = []
    for ep in range(cl["epoch"]):
        model.train()
        if strategi.get("sampler") == "seimbang":
            urut = indeks_sampler_seimbang(idx_lat, y, K, gen)
        else:
            urut = idx_lat[torch.randperm(len(idx_lat), generator=gen).numpy()]
        t0, total_loss = time.time(), 0.0
        for i in range(0, len(urut), cl["batch"]):
            b = urut[i:i + cl["batch"]]
            x, m = eks01.augmentasi(*eks01.ke_tensor(X, M, b), cfg["augmentasi"], ukuran, gen)
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=cl["amp"]):
                logit = model(x)
            logit, yb = logit.float(), y_t[b]
            if loss_jenis == "ce":
                loss = F.cross_entropy(logit, yb, weight=w)
            elif loss_jenis == "focal":
                loss = focal_loss(logit, yb, gamma=strategi.get("gamma", 2.0), weight=w)
            elif loss_jenis == "logit_adjusted":
                loss = logit_adjusted_loss(logit, yb, log_prior, tau=strategi.get("tau", 1.0))
            else:
                raise ValueError(loss_jenis)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            jadwal.step()
            total_loss += loss.item() * len(b)
        bacc_val = balanced_accuracy_score(y[idx_val], eks01.prediksi(model, X, M, idx_val, cfg).argmax(1))
        riwayat.append({"epoch": ep + 1, "loss": total_loss / len(urut), "bacc_val": float(bacc_val),
                        "detik": round(time.time() - t0, 1)})
    return model, riwayat, prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--lanjut", default=None)
    args = ap.parse_args()
    if args.lanjut:
        run = (FOLDER / args.lanjut).resolve()
        cfg = yaml.safe_load((run / "config.yaml").read_text())
    else:
        cfg_path = FOLDER / args.config
        cfg = yaml.safe_load(cfg_path.read_text())
        run = folder_run(FOLDER, cfg_path)
    seed_semua(cfg["seed"])
    (run / "per_run").mkdir(exist_ok=True)

    meta, X, M = muat_cache(cfg["data"]["ukuran_cache"])
    if cfg["data"]["maks_citra"]:
        pilih = meta.sample(n=cfg["data"]["maks_citra"], random_state=cfg["seed"]).index.sort_values()
        meta, X, M = meta.loc[pilih].reset_index(drop=True), X[pilih.to_numpy()], M[pilih.to_numpy()]
    y = meta["dx"].map({k: i for i, k in enumerate(KELAS)}).to_numpy().astype(np.int64, copy=True)

    cs = cfg["split"]
    folds = {r: buat_fold_grup(meta, cs["n_fold"], seed=r) for r in cs["seed_ulangan"]}
    assert all(cek_kebocoran(meta, f) == 0 for f in folds.values())

    t_mulai = time.time()
    for r in cs["seed_ulangan"]:
        for f in (cs["fold_dipakai"] if cs["fold_dipakai"] is not None else range(cs["n_fold"])):
            idx_uji = np.flatnonzero(folds[r] == f)
            idx_lat, idx_val = sub_validasi(meta, np.flatnonzero(folds[r] != f), cs["frac_validasi"], seed=r * 100 + f)
            for s in cfg["strategi"]:
                path = run / "per_run" / f"{s['kode']}_r{r}_f{f}.json"
                if path.exists():
                    print(f"[lewati] {path.name}")
                    continue
                seed = cfg["seed"] + 1000 * r + 10 * f        # sama dengan eksperimen 01
                seed_semua(seed)
                t0 = time.time()
                model = eks01.buat_model({"kode": "A0"}, cfg["model"])
                model, riwayat, prior = latih(model, X, M, y, idx_lat, idx_val, cfg, s, seed)
                p_val = eks01.prediksi(model, X, M, idx_val, cfg)
                p_uji = eks01.prediksi(model, X, M, idx_uji, cfg)

                keputusan = {"argmax": (p_val, p_uji)}
                if s.get("koreksi_prior"):
                    keputusan["prior"] = (koreksi_prior(p_val, prior), koreksi_prior(p_uji, prior))
                bias, _ = cari_bias_bacc(p_val, y[idx_val], K)
                keputusan["bias_val"] = (terapkan_bias(p_val, bias), terapkan_bias(p_uji, bias))

                hasil = {"strategi": s["kode"], "ulangan": r, "fold": f, "seed": seed,
                         "n_latih": int(len(idx_lat)), "n_validasi": int(len(idx_val)), "n_uji": int(len(idx_uji)),
                         "bias_val": bias.round(2).tolist(), "riwayat": riwayat,
                         "menit": round((time.time() - t0) / 60, 2), "keputusan": {}}
                for nama, (pv, pu) in keputusan.items():
                    hasil["keputusan"][nama] = {
                        "bacc_val": float(balanced_accuracy_score(y[idx_val], pv.argmax(1))),
                        "uji": eks01.hitung_metrik(y[idx_uji], pu)}
                path.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
                pd.DataFrame(p_uji, columns=[f"p_{k}" for k in KELAS]).assign(
                    image=meta["image"].to_numpy()[idx_uji], dx=meta["dx"].to_numpy()[idx_uji],
                    strategi=s["kode"], ulangan=r, fold=f).to_csv(run / "per_run" / f"{s['kode']}_r{r}_f{f}_preds.csv", index=False)
                a = hasil["keputusan"]["argmax"]
                print(f"{s['kode']:14s} r{r} f{f}: BACC val {a['bacc_val']:.4f} | BACC uji {a['uji']['bacc']:.4f} "
                      f"| +bias_val uji {hasil['keputusan']['bias_val']['uji']['bacc']:.4f} | {hasil['menit']} mnt", flush=True)
                del model
                torch.cuda.empty_cache()

    # ---- ringkasan
    per_run = [json.loads(p.read_text()) for p in sorted((run / "per_run").glob("*.json"))]
    urutan = {s["kode"]: i for i, s in enumerate(cfg["strategi"])}
    per_run.sort(key=lambda h: (urutan.get(h["strategi"], 99), h["ulangan"], h["fold"]))
    baris = []
    for h in per_run:
        for nama, k in h["keputusan"].items():
            baris.append({"strategi": h["strategi"], "keputusan": nama, "ulangan": h["ulangan"], "fold": h["fold"],
                          "bacc_val": k["bacc_val"], **{m: k["uji"][m] for m in eks01.METRIK_UTAMA}})
    df = pd.DataFrame(baris)
    df.to_csv(run / "ringkasan_per_fold.csv", index=False)
    ringkasan = {}
    for (s, kep), g in df.groupby(["strategi", "keputusan"], sort=False):
        ringkasan[f"{s}|{kep}"] = {m: {"rata": float(g[m].mean()), "sd": float(g[m].std(ddof=1)) if len(g) > 1 else float("nan")}
                                   for m in ["bacc_val", *eks01.METRIK_UTAMA]}
        ringkasan[f"{s}|{kep}"]["n_fold"] = int(len(g))

    argmax = df[df.keputusan == "argmax"]
    rata_val = argmax.groupby("strategi", sort=False)["bacc_val"].mean()
    pemenang = str(rata_val.idxmax())
    rasio = float(np.mean([h["n_uji"] / (h["n_latih"] + h["n_validasi"]) for h in per_run]))
    acuan = cfg["statistik"]["pembanding"]
    banding = {}
    if acuan in set(argmax.strategi):
        d0 = argmax[argmax.strategi == acuan].set_index(["ulangan", "fold"])["bacc"]
        for (s, kep), g in df.groupby(["strategi", "keputusan"], sort=False):
            if s == acuan and kep == "argmax":
                continue
            d = (g.set_index(["ulangan", "fold"])["bacc"] - d0).dropna().to_numpy()
            t, p = eks01.uji_t_terkoreksi(d, rasio)
            banding[f"{s}|{kep}"] = {"selisih_bacc_rata": float(d.mean()), "selisih_per_fold": d.round(4).tolist(), "t": t, "p": p}
        kunci = [k for k in banding if not math.isnan(banding[k]["p"])]
        for k, ph in zip(kunci, eks01.holm([banding[k]["p"] for k in kunci])):
            banding[k]["p_holm"] = float(ph)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        kotor = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        commit, kotor = None, None
    simpan_metrik(run, {
        "nama": cfg["nama"], "mode": cfg["mode"], "git_commit": commit, "git_ada_perubahan_belum_commit": kotor,
        "aturan_pemilihan": "BACC validasi dalam rata-rata tertinggi (keputusan argmax), ditetapkan sebelum run",
        "pemenang_menurut_validasi": pemenang, "bacc_val_rata_per_strategi": rata_val.round(4).to_dict(),
        "ringkasan": ringkasan, f"banding_vs_{acuan}|argmax": banding, "rasio_uji_latih": rasio,
        "per_run": per_run, "durasi_jam": round((time.time() - t_mulai) / 3600, 3)})

    print(f"\nHasil disimpan di {run}")
    print(f"Pemenang menurut validasi dalam: {pemenang}")
    print(f"{'strategi|keputusan':<30}{'BACC val':>10}{'BACC uji':>18}{'sens mel':>16}{'Δ vs acuan':>11}{'p_holm':>8}")
    for k, s in ringkasan.items():
        b = banding.get(k, {})
        print(f"{k:<30}{s['bacc_val']['rata']:>10.4f}{s['bacc']['rata']:>11.4f}±{s['bacc']['sd']:.4f}"
              f"{s['sens_mel']['rata']:>9.3f}±{s['sens_mel']['sd']:.3f}{b.get('selisih_bacc_rata', float('nan')):>11.4f}"
              f"{b.get('p_holm', float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
