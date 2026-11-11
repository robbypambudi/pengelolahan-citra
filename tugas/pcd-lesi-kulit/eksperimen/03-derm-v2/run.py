"""Eksperimen 03 — Derm-MobileNetV3 v2: letak M3, lebar cincin, dan uji jalan pintas H5d.

Semua varian dilatih dengan strategi pemenang eksperimen 02 (S6, logit-adjusted loss) dan
dievaluasi dengan dua keputusan: argmax dan +bias_val (bias per kelas dari validasi dalam).
Cache, split per lesi, augmentasi, metrik, dan seed SAMA dengan eksperimen 01/02
(fungsi diimpor dari ../01-derm-mobilenetv3/run.py dan ../02-imbalance-a0/run.py).

Varian:
  A0                MobileNetV3-Large polos (acuan; harus = S6 di eksperimen 02)
  A3-b<j>-r<w>      + M3 penuh dari blok j, cincin selebar w px
  A0-lesi / A0-latar  H5d: input ditutup (lihat citra.derm.TutupInput)

Aturan (ditetapkan sebelum run penuh): lihat README.

Jalankan dari root proyek:
  uv run python tugas/pcd-lesi-kulit/eksperimen/03-derm-v2/run.py [--config config-cepat.yaml] [--lanjut hasil/<waktu>]
"""

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

from citra import folder_run, seed_semua, simpan_metrik
from citra.data import KELAS
from citra.derm import TutupInput, hitung_biaya
from citra.imbalance import cari_bias_bacc, terapkan_bias
from citra.split import buat_fold_grup, cek_kebocoran, sub_validasi

FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / "pyproject.toml").exists())
EKS = FOLDER.parent


def _muat(nama, path):
    spec = importlib.util.spec_from_file_location(nama, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eks02 = _muat("eks02", EKS / "02-imbalance-a0" / "run.py")
eks01 = eks02.eks01
from siapkan import muat_cache  # noqa: E402  (path eksperimen 01 sudah ditambahkan oleh eks02)

K = len(KELAS)
KEPUTUSAN = ("argmax", "bias_val")


def buat_model(v, cm):
    cm = {**cm, **{k: v[k] for k in ("lapisan_m3", "lebar_cincin_px") if k in v}}
    model = eks01.buat_model(v, cm)
    if v.get("tutup_input"):
        model = TutupInput(model, v["tutup_input"], dilasi_px=cm["dilasi_tutup_px"])
    return model


def simpan_contoh_tutup(model, X, M, idx, cfg, path):
    x, m = eks01.ubah_ukuran(*eks01.ke_tensor(X, M, idx[:6]), cfg["data"]["ukuran_input"])
    with torch.no_grad():
        h = model.tutup(x, m)
    baris = [np.hstack([(a.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for a in (x[i], h[i])])
             for i in range(len(x))]
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(baris), cv2.COLOR_RGB2BGR))


def banding(df, pasangan, rasio):
    """Selisih BACC per fold (a − b) dengan corrected resampled t-test; pasangan: [(a, b, keputusan)]."""
    out = {}
    for a, b, kep in pasangan:
        ga = df[(df.varian == a) & (df.keputusan == kep)].set_index(["ulangan", "fold"])["bacc"]
        gb = df[(df.varian == b) & (df.keputusan == kep)].set_index(["ulangan", "fold"])["bacc"]
        d = (ga - gb).dropna().to_numpy()
        if len(d) == 0:
            continue
        t, p = eks01.uji_t_terkoreksi(d, rasio)
        out[f"{a} − {b} | {kep}"] = {"selisih_bacc_rata": float(d.mean()), "selisih_per_fold": d.round(4).tolist(),
                                     "n_fold_positif": int((d > 0).sum()), "t": t, "p": p}
    kunci = [k for k in out if not math.isnan(out[k]["p"])]
    for k, ph in zip(kunci, eks01.holm([out[k]["p"] for k in kunci])):
        out[k]["p_holm"] = float(ph)
    return out


def main(folder=None, pembuat=None, pelatih=None):
    """folder/pembuat/pelatih dipakai ulang oleh eksperimen berikutnya (04, 05); bawaan = eksperimen 03.

    pelatih(model, X, M, y, idx_lat, idx_val, cfg, strategi, seed, varian) → (model, riwayat, prior)
    """
    folder, pembuat = folder or FOLDER, pembuat or buat_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--lanjut", default=None)
    args = ap.parse_args()
    if args.lanjut:
        run = (folder / args.lanjut).resolve()
        cfg = yaml.safe_load((run / "config.yaml").read_text())
    else:
        cfg_path = folder / args.config
        cfg = yaml.safe_load(cfg_path.read_text())
        run = folder_run(folder, cfg_path)
    seed_semua(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    (run / "per_run").mkdir(exist_ok=True)
    (run / "contoh").mkdir(exist_ok=True)

    meta, X, M = muat_cache(cfg["data"]["ukuran_cache"])
    if cfg["data"]["maks_citra"]:
        pilih = meta.sample(n=cfg["data"]["maks_citra"], random_state=cfg["seed"]).index.sort_values()
        meta, X, M = meta.loc[pilih].reset_index(drop=True), X[pilih.to_numpy()], M[pilih.to_numpy()]
    y = meta["dx"].map({k: i for i, k in enumerate(KELAS)}).to_numpy().astype(np.int64, copy=True)
    otsu_ok = ~meta["otsu_gagal"].to_numpy().astype(bool)

    cs, strategi = cfg["split"], cfg["strategi"]
    folds = {r: buat_fold_grup(meta, cs["n_fold"], seed=r) for r in cs["seed_ulangan"]}
    assert all(cek_kebocoran(meta, f) == 0 for f in folds.values())

    biaya = {v["kode"]: hitung_biaya(pembuat(v, cfg["model"])) for v in cfg["varian"]}

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
                seed = cfg["seed"] + 1000 * r + 10 * f        # sama dengan eksperimen 01/02
                seed_semua(seed)
                t0 = time.time()
                model = pembuat(v, cfg["model"])
                if pelatih is None:
                    model, riwayat, _ = eks02.latih(model, X, M, y, idx_lat, idx_val, cfg, strategi, seed)
                else:
                    model, riwayat, _ = pelatih(model, X, M, y, idx_lat, idx_val, cfg, strategi, seed, v)
                p_val = eks01.prediksi(model, X, M, idx_val, cfg)
                p_uji = eks01.prediksi(model, X, M, idx_uji, cfg)
                bias, _ = cari_bias_bacc(p_val, y[idx_val], K)
                keputusan = {"argmax": (p_val, p_uji), "bias_val": (terapkan_bias(p_val, bias), terapkan_bias(p_uji, bias))}

                ok = otsu_ok[idx_uji]
                hasil = {"varian": v["kode"], "ulangan": r, "fold": f, "seed": seed,
                         "n_latih": int(len(idx_lat)), "n_validasi": int(len(idx_val)), "n_uji": int(len(idx_uji)),
                         "n_uji_otsu_ok": int(ok.sum()), "bias_val": bias.round(2).tolist(), "riwayat": riwayat,
                         "parameter_terpelajari": model.parameter_terpelajari(),
                         "menit": round((time.time() - t0) / 60, 2), "keputusan": {}}
                for nama, (pv, pu) in keputusan.items():
                    hasil["keputusan"][nama] = {
                        "bacc_val": float(balanced_accuracy_score(y[idx_val], pv.argmax(1))),
                        "uji": eks01.hitung_metrik(y[idx_uji], pu),
                        "bacc_uji_otsu_ok": float(balanced_accuracy_score(y[idx_uji][ok], pu[ok].argmax(1)))}
                path.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
                pd.DataFrame(p_uji, columns=[f"p_{k}" for k in KELAS]).assign(
                    image=meta["image"].to_numpy()[idx_uji], dx=meta["dx"].to_numpy()[idx_uji],
                    varian=v["kode"], ulangan=r, fold=f).to_csv(run / "per_run" / f"{v['kode']}_r{r}_f{f}_preds.csv", index=False)
                if f == 0 and r == cs["seed_ulangan"][0]:
                    if isinstance(model, TutupInput):
                        simpan_contoh_tutup(model, X, M, idx_uji, cfg, run / "contoh" / f"{v['kode']}_input.jpg")
                    elif model.butuh_mask:
                        eks01.simpan_contoh(model, X, M, idx_uji, {**cfg, "model": {**cfg["model"], **v}},
                                            run / "contoh" / f"{v['kode']}_contoh.jpg")
                a, b = hasil["keputusan"]["argmax"], hasil["keputusan"]["bias_val"]
                print(f"{v['kode']:12s} r{r} f{f}: val {a['bacc_val']:.4f} | uji {a['uji']['bacc']:.4f} "
                      f"| +bias_val {b['uji']['bacc']:.4f} | {hasil['menit']} mnt", flush=True)
                del model
                torch.cuda.empty_cache()

    # ---- ringkasan
    per_run = [json.loads(p.read_text()) for p in sorted((run / "per_run").glob("*.json"))]
    urutan = {v["kode"]: i for i, v in enumerate(cfg["varian"])}
    per_run.sort(key=lambda h: (urutan.get(h["varian"], 99), h["ulangan"], h["fold"]))
    baris = [{"varian": h["varian"], "keputusan": kep, "ulangan": h["ulangan"], "fold": h["fold"],
              "bacc_val": k["bacc_val"], "bacc_uji_otsu_ok": k["bacc_uji_otsu_ok"],
              **{m: k["uji"][m] for m in eks01.METRIK_UTAMA}}
             for h in per_run for kep, k in h["keputusan"].items()]
    df = pd.DataFrame(baris)
    df.to_csv(run / "ringkasan_per_fold.csv", index=False)
    ringkasan = {}
    for (v, kep), g in df.groupby(["varian", "keputusan"], sort=False):
        ringkasan[f"{v}|{kep}"] = {m: {"rata": float(g[m].mean()), "sd": float(g[m].std(ddof=1)) if len(g) > 1 else float("nan")}
                                   for m in ["bacc_val", "bacc_uji_otsu_ok", *eks01.METRIK_UTAMA]}
        ringkasan[f"{v}|{kep}"]["n_fold"] = int(len(g))

    rasio = float(np.mean([h["n_uji"] / (h["n_latih"] + h["n_validasi"]) for h in per_run]))
    acuan, st = cfg["statistik"]["pembanding"], cfg["statistik"]
    a3 = st.get("kandidat") or [v["kode"] for v in cfg["varian"] if v.get("m3")]
    argmax = df[df.keputusan == "argmax"]
    val_a3 = argmax[argmax.varian.isin(a3)].groupby("varian", sort=False)["bacc_val"].mean()
    terpilih = str(val_a3.idxmax()) if len(val_a3) else None

    uji_a3 = banding(df, [(v, acuan, kep) for v in a3 for kep in KEPUTUSAN], rasio)
    uji_h5d = banding(df, [(a, b, kep) for a, b in st.get("h5d", []) for kep in KEPUTUSAN], rasio)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        kotor = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        if kotor:   # simpan perubahan yang belum di-commit agar run tetap bisa direproduksi
            (run / "perubahan_belum_commit.patch").write_text(
                subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True)
                + "\n# berkas belum dilacak:\n" + subprocess.check_output(
                    ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True))
    except Exception:
        commit, kotor = None, None
    simpan_metrik(run, {
        "nama": cfg["nama"], "mode": cfg["mode"], "git_commit": commit, "git_ada_perubahan_belum_commit": kotor,
        "strategi_latih": strategi,
        "aturan_pemilihan": "kandidat dengan BACC validasi dalam rata-rata tertinggi (argmax), ditetapkan sebelum run",
        "kandidat": a3,
        "a3_terpilih_menurut_validasi": terpilih, "bacc_val_rata_a3": val_a3.round(4).to_dict(),
        "biaya": biaya, "ringkasan": ringkasan,
        f"banding_a3_vs_{acuan}": uji_a3, "banding_h5d": uji_h5d,
        "rasio_uji_latih": rasio, "per_run": per_run, "durasi_jam": round((time.time() - t_mulai) / 3600, 3)})

    print(f"\nHasil disimpan di {run}")
    print(f"A3 terpilih menurut validasi dalam: {terpilih}")
    print(f"{'varian|keputusan':<24}{'BACC val':>10}{'BACC uji':>18}{'sens mel':>9}{'param+':>9}")
    for k, s in ringkasan.items():
        print(f"{k:<24}{s['bacc_val']['rata']:>10.4f}{s['bacc']['rata']:>11.4f}±{s['bacc']['sd']:.4f}"
              f"{s['sens_mel']['rata']:>9.3f}{biaya[k.split('|')[0]]['parameter_tambahan']:>9,}")
    for nama, u in (("A3 vs acuan", uji_a3), ("H5d", uji_h5d)):
        print(f"\n{nama}:")
        for k, b in u.items():
            print(f"  {k:<36}{b['selisih_bacc_rata']:>+9.4f}  {b['n_fold_positif']}/{len(b['selisih_per_fold'])} fold +"
                  f"  p_holm {b.get('p_holm', float('nan')):.3f}")


if __name__ == "__main__":
    main()
