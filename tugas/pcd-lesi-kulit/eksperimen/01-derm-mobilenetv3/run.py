"""Eksperimen 01 — RQ5: ablasi Derm-MobileNetV3 dengan CV berkelompok per lesi.

Alur: seed → folder_run → cache + split per lesi → untuk tiap (ulangan, fold, varian):
latih dengan validasi dalam (pilih epoch terbaik menurut BACC) → prediksi fold uji →
simpan per-run JSON + preds → ringkasan, uji t terkoreksi vs A0 (Nadeau & Bengio, 2003),
koreksi Holm → metrics.json.

Jalankan dari root proyek:
  uv run python tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/run.py                       # config.yaml
  uv run python tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/run.py --config config-cepat.yaml
  uv run python tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/run.py --lanjut hasil/<waktu>  # lanjutkan run terputus
"""

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

from citra import folder_run, seed_semua, simpan_metrik
from citra.data import KELAS
from citra.derm import DermMobileNetV3, buat_varian_lilhore, hitung_biaya
from citra.split import buat_fold_grup, cek_kebocoran, sub_validasi
from siapkan import muat_cache

FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / "pyproject.toml").exists())
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEL = KELAS.index("mel")
METRIK_UTAMA = ["bacc", "f1_makro", "auroc_makro", "sens_mel", "sens_mel_spes80", "akurasi", "ece"]


# ------------------------------------------------------------------ model
def buat_model(v, cm):
    if v.get("lilhore"):
        return buat_varian_lilhore(n_kelas=len(KELAS), pralatih=True)
    model = DermMobileNetV3(
        n_kelas=len(KELAS), pralatih=True, m1_cc=v.get("m1_cc", False), m1_rambut=v.get("m1_rambut", False),
        m2=v.get("m2", False), m3=v.get("m3", False), m3_mode=v.get("m3_mode", "penuh"),
        lapisan_m3=cm["lapisan_m3"], lebar_cincin_px=cm["lebar_cincin_px"], k_rambut=cm["k_rambut"],
        p_awal=cm["p_awal"], bottleneck=cm["bottleneck"])
    if v.get("praproses_tetap"):
        # A0+P: operator M1 yang sama, gerbang = 1 dan semua parameter M1 dibekukan.
        for modul in (model.cc, model.rambut):
            with torch.no_grad():
                modul.gerbang.fill_(1.0)
            for p in modul.parameters():
                p.requires_grad_(False)
    return model


# ------------------------------------------------------------------ data
def ke_tensor(X, M, idx):
    x = torch.from_numpy(np.ascontiguousarray(X[idx])).to(DEV, non_blocking=True)
    m = torch.from_numpy(np.ascontiguousarray(M[idx])).to(DEV, non_blocking=True)
    return x.permute(0, 3, 1, 2).float().div_(255), m[:, None].float()


def augmentasi(x, m, ca, ukuran, gen):
    """Affine acak (rotasi, flip, zoom, geser) yang sama untuk citra dan mask + jitter kecerahan/kontras."""
    b = x.shape[0]
    u = lambda *s: torch.rand(*s, generator=gen).to(DEV)
    sudut = u(b) * 2 * math.pi if ca["rotasi"] else torch.zeros(b, device=DEV)
    flip = torch.where(u(b) < 0.5, -1.0, 1.0) if ca["flip"] else torch.ones(b, device=DEV)
    skala = ca["skala_min"] + (1 - ca["skala_min"]) * u(b)
    tx, ty = (u(b) * 2 - 1) * ca["geser"], (u(b) * 2 - 1) * ca["geser"]
    cos, sin = torch.cos(sudut) * skala, torch.sin(sudut) * skala
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (b, 3, ukuran, ukuran), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    m = (F.grid_sample(m, grid, mode="bilinear", padding_mode="reflection", align_corners=False) > 0.5).float()
    kec = 1 + (u(b, 1, 1, 1) * 2 - 1) * ca["kecerahan"]
    kon = 1 + (u(b, 1, 1, 1) * 2 - 1) * ca["kontras"]
    rata = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - rata) * kon + rata * kec).clamp(0, 1), m


def ubah_ukuran(x, m, ukuran):
    x = F.interpolate(x, size=ukuran, mode="area")
    m = (F.interpolate(m, size=ukuran, mode="area") > 0.5).float()
    return x, m


@torch.no_grad()
def prediksi(model, X, M, idx, cfg):
    model.eval()
    ukuran, keluar = cfg["data"]["ukuran_input"], []
    for i in range(0, len(idx), cfg["latih"]["batch_eval"]):
        x, m = ubah_ukuran(*ke_tensor(X, M, idx[i:i + cfg["latih"]["batch_eval"]]), ukuran)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=cfg["latih"]["amp"]):
            logit = model(x, m) if getattr(model, "butuh_mask", False) else model(x)
        keluar.append(logit.float().softmax(1).cpu())
    return torch.cat(keluar).numpy()


# ------------------------------------------------------------------ metrik
def sens_pada_spes(y_pos, skor, spes=0.8):
    neg = skor[~y_pos]
    if y_pos.sum() == 0 or len(neg) == 0:
        return float("nan")
    ambang = np.quantile(neg, spes)
    return float((skor[y_pos] > ambang).mean())


def ece(y, prob, n_bin=15):
    conf, benar = prob.max(1), prob.argmax(1) == y
    tepi = np.linspace(0, 1, n_bin + 1)
    total = 0.0
    for a, b in zip(tepi[:-1], tepi[1:]):
        sel = (conf > a) & (conf <= b)
        if sel.any():
            total += sel.mean() * abs(benar[sel].mean() - conf[sel].mean())
    return float(total)


def hitung_metrik(y, prob):
    pred = prob.argmax(1)
    try:
        auroc = float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=list(range(len(KELAS)))))
    except ValueError:
        auroc = float("nan")
    return {
        "bacc": float(balanced_accuracy_score(y, pred)),
        "f1_makro": float(f1_score(y, pred, average="macro", labels=list(range(len(KELAS))), zero_division=0)),
        "auroc_makro": auroc,
        "sens_mel": float(recall_score(y == MEL, pred == MEL, zero_division=0)),
        "sens_mel_spes80": sens_pada_spes(y == MEL, prob[:, MEL]),
        "akurasi": float(accuracy_score(y, pred)),
        "ece": ece(y, prob),
        "recall_per_kelas": dict(zip(KELAS, recall_score(y, pred, average=None, labels=list(range(len(KELAS))),
                                                         zero_division=0).round(4).tolist())),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(KELAS)))).tolist(),
        "n": int(len(y)),
    }


# ------------------------------------------------------------------ latih
def latih(model, X, M, y, idx_lat, idx_val, cfg, seed):
    cl, ukuran = cfg["latih"], cfg["data"]["ukuran_input"]
    gen = torch.Generator().manual_seed(seed)
    model = model.to(DEV)
    if hasattr(model, "kelompok_parameter"):
        grup = model.kelompok_parameter(cl["lr_dasar"], cl["lr_modul"], cl["weight_decay"])
        for g in grup:
            g["params"] = [p for p in g["params"] if p.requires_grad]
        grup = [g for g in grup if g["params"]]
    else:
        grup = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cl["lr_dasar"]}]
    opt = torch.optim.AdamW(grup, weight_decay=cl["weight_decay"])
    langkah_per_epoch = math.ceil(len(idx_lat) / cl["batch"])
    total, warmup = cl["epoch"] * langkah_per_epoch, cl["warmup_epoch"] * langkah_per_epoch
    jadwal = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    hitung = np.bincount(y[idx_lat], minlength=len(KELAS)).astype(float)
    bobot = torch.tensor(len(idx_lat) / (len(KELAS) * np.maximum(hitung, 1)), dtype=torch.float32, device=DEV) \
        if cl["bobot_kelas"] else None
    y_t = torch.from_numpy(y).to(DEV)

    terbaik, state_terbaik, epoch_terbaik, riwayat, sabar = -1.0, None, -1, [], 0
    for ep in range(cl["epoch"]):
        model.train()
        urut = idx_lat[torch.randperm(len(idx_lat), generator=gen).numpy()]
        t0, total_loss = time.time(), 0.0
        for i in range(0, len(urut), cl["batch"]):
            b = urut[i:i + cl["batch"]]
            x, m = augmentasi(*ke_tensor(X, M, b), cfg["augmentasi"], ukuran, gen)
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=cl["amp"]):
                logit = model(x, m) if getattr(model, "butuh_mask", False) else model(x)
            loss = F.cross_entropy(logit.float(), y_t[b], weight=bobot)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            jadwal.step()
            total_loss += loss.item() * len(b)
        bacc_val = balanced_accuracy_score(y[idx_val], prediksi(model, X, M, idx_val, cfg).argmax(1))
        catat = {"epoch": ep + 1, "loss": total_loss / len(urut), "bacc_val": float(bacc_val),
                 "detik": round(time.time() - t0, 1)}
        if hasattr(model, "parameter_terpelajari"):
            catat.update(model.parameter_terpelajari())
        riwayat.append(catat)
        if bacc_val > terbaik:
            terbaik, epoch_terbaik, sabar = bacc_val, ep + 1, 0
            if cl["pilih_epoch"] == "terbaik_val":
                state_terbaik = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            sabar += 1
            if cl["pilih_epoch"] == "terbaik_val" and sabar >= cl["patience"]:
                break
    if cl["pilih_epoch"] == "terbaik_val":
        model.load_state_dict(state_terbaik)
        dipakai = epoch_terbaik
    else:                                   # "terakhir": model di akhir jadwal cosine
        dipakai = len(riwayat)
    return model, {"epoch_dipakai": dipakai, "epoch_bacc_val_tertinggi": epoch_terbaik,
                   "bacc_val_tertinggi": float(terbaik), "bacc_val_akhir": riwayat[-1]["bacc_val"],
                   "riwayat": riwayat}


# ------------------------------------------------------------------ statistik
def uji_t_terkoreksi(d, rasio_uji_latih):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) untuk selisih per-fold d."""
    d = np.asarray(d, float)
    k = len(d)
    if k < 2 or d.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t = d.mean() / math.sqrt((1 / k + rasio_uji_latih) * d.var(ddof=1))
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def holm(p):
    p = np.asarray(p, float)
    urut = np.argsort(p)
    hasil, maks = np.empty_like(p), 0.0
    for r, i in enumerate(urut):
        maks = max(maks, min(1.0, (len(p) - r) * p[i]))
        hasil[i] = maks
    return hasil


def ringkas(per_run, cfg, rasio):
    df = pd.DataFrame([{"varian": r["varian"], "ulangan": r["ulangan"], "fold": r["fold"], **{k: r["uji"][k] for k in METRIK_UTAMA}}
                       for r in per_run])
    ringkasan = {}
    for v, g in df.groupby("varian", sort=False):
        ringkasan[v] = {k: {"rata": float(g[k].mean()), "sd": float(g[k].std(ddof=1)) if len(g) > 1 else float("nan")}
                        for k in METRIK_UTAMA}
        ringkasan[v]["n_fold"] = int(len(g))
    pemb, banding = cfg["statistik"]["pembanding"], {}
    if pemb in ringkasan:
        dasar = df[df.varian == pemb].set_index(["ulangan", "fold"])
        for v in ringkasan:
            if v == pemb:
                continue
            lain = df[df.varian == v].set_index(["ulangan", "fold"])
            pasangan = dasar.index.intersection(lain.index)
            d = (lain.loc[pasangan, "bacc"] - dasar.loc[pasangan, "bacc"]).to_numpy()
            t, p = uji_t_terkoreksi(d, rasio)
            banding[v] = {"selisih_bacc_rata": float(d.mean()) if len(d) else float("nan"),
                          "selisih_per_fold": d.round(4).tolist(), "t": t, "p": p}
        kunci = [v for v in banding if not math.isnan(banding[v]["p"])]
        for v, ph in zip(kunci, holm([banding[v]["p"] for v in kunci])):
            banding[v]["p_holm"] = float(ph)
        for v in banding:
            s = banding[v]
            s["lolos_sesoi"] = bool(s["selisih_bacc_rata"] >= cfg["statistik"]["sesoi_bacc"]
                                    and s.get("p_holm", 1.0) < cfg["statistik"]["alpha"])
    return ringkasan, banding


# ------------------------------------------------------------------ contoh visual
def simpan_contoh(model, X, M, idx, cfg, path):
    """Baris: asli | keluaran M1 (bila ada) | mask lesi (hijau) + cincin (kuning)."""
    from citra.derm import mask_cincin
    x, m = ubah_ukuran(*ke_tensor(X, M, idx[:6]), cfg["data"]["ukuran_input"])
    with torch.no_grad():
        x1 = model.praproses(x) if hasattr(model, "praproses") else x
        cincin = mask_cincin(m, cfg["model"]["lebar_cincin_px"])
    baris = []
    for i in range(len(x)):
        a = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        b = (x1[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        c = a.copy()
        c[m[i, 0].cpu().numpy() > 0] = (0.5 * c[m[i, 0].cpu().numpy() > 0] + [0, 110, 0]).astype(np.uint8)
        c[cincin[i, 0].cpu().numpy() > 0] = (0.5 * c[cincin[i, 0].cpu().numpy() > 0] + [120, 120, 0]).astype(np.uint8)
        baris.append(np.hstack([a, b, c]))
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(baris), cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--lanjut", default=None, help="folder run (relatif ke folder eksperimen) yang dilanjutkan")
    args = ap.parse_args()

    if args.lanjut:
        run = (FOLDER / args.lanjut).resolve()
        cfg_path = run / "config.yaml"
    else:
        cfg_path = FOLDER / args.config
        run = None
    cfg = yaml.safe_load(cfg_path.read_text())
    seed_semua(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    if run is None:
        run = folder_run(FOLDER, cfg_path)
    (run / "per_run").mkdir(exist_ok=True)

    meta, X, M = muat_cache(cfg["data"]["ukuran_cache"])
    if cfg["data"]["sumber_mask"] != "otsu":
        raise NotImplementedError("Hanya mask Otsu yang tersedia saat ini (U-Net: rencana berikutnya).")
    if cfg["data"]["maks_citra"]:
        pilih = meta.sample(n=cfg["data"]["maks_citra"], random_state=cfg["seed"]).index.sort_values()
        meta, X, M = meta.loc[pilih].reset_index(drop=True), X[pilih.to_numpy()], M[pilih.to_numpy()]
    y = meta["dx"].map({k: i for i, k in enumerate(KELAS)}).to_numpy().astype(np.int64, copy=True)

    cs = cfg["split"]
    folds = {r: buat_fold_grup(meta, cs["n_fold"], seed=r) for r in cs["seed_ulangan"]}
    bocor = {r: cek_kebocoran(meta, f) for r, f in folds.items()}
    assert all(b == 0 for b in bocor.values()), f"Kebocoran lesi antar-fold: {bocor}"
    if cs.get("file") and not cfg["data"]["maks_citra"]:
        path_split = ROOT / cs["file"]
        path_split.parent.mkdir(parents=True, exist_ok=True)
        meta[["image", "lesion_id", "dx"]].assign(**{f"fold_r{r}": f for r, f in folds.items()}).to_csv(path_split, index=False)

    biaya = {v["kode"]: hitung_biaya(buat_model(v, cfg["model"])) for v in cfg["varian"]}

    t_mulai = time.time()
    for r in cs["seed_ulangan"]:
        for f in (cs["fold_dipakai"] if cs["fold_dipakai"] is not None else range(cs["n_fold"])):
            idx_uji = np.flatnonzero(folds[r] == f)
            idx_lat, idx_val = sub_validasi(meta, np.flatnonzero(folds[r] != f), cs["frac_validasi"], seed=r * 100 + f)
            for v in cfg["varian"]:
                path = run / "per_run" / f"{v['kode']}_r{r}_f{f}.json"
                if path.exists():
                    print(f"[lewati] {path.name}")
                    continue
                seed = cfg["seed"] + 1000 * r + 10 * f
                seed_semua(seed)
                t0 = time.time()
                model = buat_model(v, cfg["model"])
                model, info = latih(model, X, M, y, idx_lat, idx_val, cfg, seed)
                prob = prediksi(model, X, M, idx_uji, cfg)
                hasil = {"varian": v["kode"], "ulangan": r, "fold": f, "seed": seed,
                         "n_latih": int(len(idx_lat)), "n_validasi": int(len(idx_val)), "n_uji": int(len(idx_uji)),
                         "uji": hitung_metrik(y[idx_uji], prob), **info,
                         "parameter_terpelajari": model.parameter_terpelajari() if hasattr(model, "parameter_terpelajari") else {},
                         "menit": round((time.time() - t0) / 60, 2)}
                path.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
                pd.DataFrame(prob, columns=[f"p_{k}" for k in KELAS]).assign(
                    image=meta["image"].to_numpy()[idx_uji], dx=meta["dx"].to_numpy()[idx_uji], varian=v["kode"],
                    ulangan=r, fold=f).to_csv(run / "per_run" / f"{v['kode']}_r{r}_f{f}_preds.csv", index=False)
                if r == cs["seed_ulangan"][0] and f == (cs["fold_dipakai"] or [0])[0] and (v.get("m1_cc") or v.get("m3")):
                    simpan_contoh(model, X, M, idx_uji, cfg, run / "contoh" / f"{v['kode']}_contoh.jpg")
                print(f"{v['kode']:6s} r{r} f{f}: BACC uji {hasil['uji']['bacc']:.4f} | sens mel {hasil['uji']['sens_mel']:.3f} "
                      f"| epoch dipakai {info['epoch_dipakai']} | {hasil['menit']} mnt", flush=True)
                del model
                torch.cuda.empty_cache()

    per_run = [json.loads(p.read_text()) for p in sorted((run / "per_run").glob("*.json"))]
    urutan = {v["kode"]: i for i, v in enumerate(cfg["varian"])}
    per_run.sort(key=lambda r: (urutan.get(r["varian"], 99), r["ulangan"], r["fold"]))
    rasio = np.mean([r["n_uji"] / (r["n_latih"] + r["n_validasi"]) for r in per_run])
    ringkasan, banding = ringkas(per_run, cfg, rasio)
    pd.concat([pd.read_csv(p) for p in sorted((run / "per_run").glob("*_preds.csv"))]).to_csv(run / "preds.csv", index=False)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        kotor = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        commit, kotor = None, None
    simpan_metrik(run, {
        "nama": cfg["nama"], "mode": cfg["mode"], "git_commit": commit, "git_ada_perubahan_belum_commit": kotor,
        "perangkat": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
        "data": {"n": int(len(meta)), "per_kelas": meta["dx"].value_counts().to_dict(),
                 "otsu_gagal": int(meta["otsu_gagal"].sum()), "versi_split": "v0-lesion_id (tanpa dedup pHash)",
                 "kebocoran_lesi_antar_fold": bocor},
        "biaya": biaya, "ringkasan": ringkasan, "banding_vs_" + cfg["statistik"]["pembanding"]: banding,
        "rasio_uji_latih": float(rasio), "per_run": per_run,
        "durasi_jam": round((time.time() - t_mulai) / 3600, 3),
    })

    print(f"\nHasil disimpan di {run}")
    print(f"{'varian':<8}{'BACC':>16}{'sens mel':>16}{'AUROC':>16}{'Δ vs A0':>10}{'p_holm':>9}")
    for v, s in ringkasan.items():
        b = banding.get(v, {})
        print(f"{v:<8}{s['bacc']['rata']:>9.4f}±{s['bacc']['sd']:.4f}{s['sens_mel']['rata']:>9.3f}±{s['sens_mel']['sd']:.3f}"
              f"{s['auroc_makro']['rata']:>9.4f}±{s['auroc_makro']['sd']:.4f}"
              f"{b.get('selisih_bacc_rata', float('nan')):>10.4f}{b.get('p_holm', float('nan')):>9.3f}")


if __name__ == "__main__":
    main()
