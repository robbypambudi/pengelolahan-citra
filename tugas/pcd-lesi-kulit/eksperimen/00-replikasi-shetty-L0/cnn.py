"""Replikasi CNN kustom Shetty dkk. (2022) — protokol L0 (persis paper).

Paper memakai Keras; di sini PyTorch dengan default Keras ditiru (glorot_uniform, BN momentum 0,99 / eps 1e-3).
Alur per seed: 100 citra/kelas + flip -> 1.400 citra 96x96 -> split 80:20 tak berstrata (uji = 280)
-> latih 150 epoch dengan augmentasi acak -> laporkan akurasi latih & uji epoch terakhir (Tabel 6) + laporan per kelas (Tabel 7).

Jalankan dari root proyek:  uv run python tugas/pcd-lesi-kulit/eksperimen/00-replikasi-shetty-L0/cnn.py
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn

from citra import folder_run, metrik_klasifikasi, seed_semua, simpan_metrik
from citra.data import KELAS, muat_ham10000

FOLDER = Path(__file__).resolve().parent
PAPER = {"latih": 95.18, "uji": 86.43, "uji_dari_tabel7": 85.03}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class CNNShetty(nn.Module):
    """Tabel 2 paper, urutan lapisan apa adanya (BN tidak ada setelah Conv64 pertama)."""

    def __init__(self, n_kelas=7):
        super().__init__()
        bn = lambda c: nn.BatchNorm2d(c, momentum=0.01, eps=1e-3)
        self.fitur = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), bn(32),
            nn.MaxPool2d(3), nn.Dropout(0.25),                       # 96 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), bn(64),
            nn.MaxPool2d(2), nn.Dropout(0.25),                       # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), bn(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), bn(128),
            nn.MaxPool2d(2), nn.Dropout(0.25),                       # 16 -> 8
        )
        self.kepala = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 8 * 8, 1024), nn.ReLU(),
            nn.BatchNorm1d(1024, momentum=0.01, eps=1e-3), nn.Dropout(0.5),
            nn.Linear(1024, n_kelas),                                # softmax ada di CrossEntropyLoss
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.kepala(self.fitur(x))


def augmentasi(img, a, rng):
    """Transformasi acak gaya Keras ImageDataGenerator, isi tepi 'nearest' (BORDER_REPLICATE)."""
    h, w = img.shape[:2]
    sudut = rng.uniform(-a["rotasi_derajat"], a["rotasi_derajat"])
    skala = rng.uniform(1 - a["zoom"], 1 + a["zoom"])
    shear = math.radians(rng.uniform(-a["shear_derajat"], a["shear_derajat"]))
    tx, ty = rng.uniform(-a["geser"], a["geser"]) * w, rng.uniform(-a["geser"], a["geser"]) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), sudut, 1 / skala)
    M[0, 1] += -math.sin(shear)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if a["flip_h"] and rng.random() < 0.5:
        out = out[:, ::-1]
    return out


def muat_citra(paths, ukuran):
    return np.stack([cv2.resize(cv2.imread(str(p)), (ukuran, ukuran)) for p in paths])


def ke_tensor(batch):
    return torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float().div(255)


def satu_seed(df, seed, cfg):
    c, a = cfg["cnn"], cfg["cnn"]["augment"]
    seed_semua(seed)
    rng = np.random.default_rng(seed)
    idx = (df.groupby("dx", group_keys=False)
             .apply(lambda g: g.sample(cfg["subset"]["per_kelas"], random_state=seed)).index.values)
    img = muat_citra(df.loc[idx, "path"], c["ukuran"])
    X = np.concatenate([img, img[:, :, ::-1]])                     # 200/kelas: asli + flip horizontal
    y = np.concatenate([df.loc[idx, "dx"].map(KELAS.index).values] * 2)
    asal = np.concatenate([idx, idx])
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        X, y, asal, test_size=c["test_size"], random_state=42, stratify=y if c["stratify"] else None)

    model = CNNShetty().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr_awal"])
    gamma = (c["lr_akhir"] / c["lr_awal"]) ** (1 / (c["epoch"] - 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    loss_fn = nn.CrossEntropyLoss()
    Xte_t, yte_t = ke_tensor(Xte).to(DEV), torch.from_numpy(yte).to(DEV)

    riwayat = []
    for ep in range(c["epoch"]):
        model.train()
        urut = rng.permutation(len(Xtr))
        benar = total_loss = 0
        for i in range(0, len(urut), c["batch"]):
            b = urut[i:i + c["batch"]]
            xb = ke_tensor(np.stack([augmentasi(Xtr[j], a, rng) for j in b])).to(DEV)
            yb = torch.from_numpy(ytr[b]).to(DEV)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            benar += (out.argmax(1) == yb).sum().item()
            total_loss += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            out_te = model(Xte_t)
            val_loss = loss_fn(out_te, yte_t).item()
            val_acc = (out_te.argmax(1) == yte_t).float().mean().item()
        riwayat.append({"epoch": ep + 1, "train_acc": benar / len(Xtr), "train_loss": total_loss / len(Xtr),
                        "val_acc": val_acc, "val_loss": val_loss})

    pred = out_te.argmax(1).cpu().numpy()
    laporan = classification_report(yte, pred, target_names=KELAS, output_dict=True, zero_division=0)
    return {
        "akhir": riwayat[-1],
        "metrik_uji": metrik_klasifikasi(yte, pred),
        "laporan_per_kelas": laporan,
        "support_uji": np.bincount(yte, minlength=7).tolist(),
        "uji_punya_kembaran_di_latih": float(np.isin(ate, atr).mean()),
    }, riwayat


def main():
    cfg = yaml.safe_load((FOLDER / "config.yaml").read_text())
    run = folder_run(FOLDER, FOLDER / "config.yaml")
    df = muat_ham10000()

    per_seed = {}
    for seed in cfg["cnn"]["seed_list"]:
        hasil, riwayat = satu_seed(df, seed, cfg)
        per_seed[seed] = hasil
        pd.DataFrame(riwayat).to_csv(run / f"riwayat_seed{seed}.csv", index=False)
        ak = hasil["akhir"]
        print(f"seed {seed}: latih {100*ak['train_acc']:.2f}%  uji {100*ak['val_acc']:.2f}%  "
              f"BACC uji {100*hasil['metrik_uji']['recall']:.2f}%  kembaran-bocor {hasil['uji_punya_kembaran_di_latih']:.1%}")

    lat = [100 * h["akhir"]["train_acc"] for h in per_seed.values()]
    uji = [100 * h["akhir"]["val_acc"] for h in per_seed.values()]
    rekap = {"paper": PAPER,
             "latih_rata_sd": [round(np.mean(lat), 2), round(np.std(lat, ddof=1), 2)],
             "uji_rata_sd": [round(np.mean(uji), 2), round(np.std(uji, ddof=1), 2)]}
    simpan_metrik(run, {"nama": cfg["nama"], "bagian": "cnn", "rekap": rekap, "per_seed": per_seed})
    print(f"\nHasil: {run}\nPaper: latih {PAPER['latih']} / uji {PAPER['uji']}  |  "
          f"Replikasi: latih {rekap['latih_rata_sd'][0]} ± {rekap['latih_rata_sd'][1]} / "
          f"uji {rekap['uji_rata_sd'][0]} ± {rekap['uji_rata_sd'][1]}")


if __name__ == "__main__":
    main()
