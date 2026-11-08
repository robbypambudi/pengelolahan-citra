"""Strategi penanganan ketidakseimbangan kelas: bobot, loss, sampler, dan koreksi keputusan.

Semua statistik kelas (jumlah, prior) WAJIB dihitung dari data latih fold saja.

Rujukan:
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on
  effective number of samples. CVPR.  → bobot "efektif"
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object
  detection. ICCV.  → focal_loss
- Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2021). Long-tail
  learning via logit adjustment. ICLR.  → logit_adjusted_loss, koreksi_prior
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score


def hitung_kelas(y, n_kelas):
    return np.bincount(np.asarray(y), minlength=n_kelas).astype(float)


def bobot_kelas(y, n_kelas, jenis="penuh", beta=0.999):
    """Bobot per kelas, dinormalisasi agar rata-rata tertimbang (oleh jumlah sampel) = 1.

    penuh   : 1 / n_k                         (sama dengan n/(K·n_k) setelah normalisasi)
    akar    : 1 / sqrt(n_k)                   (lebih lunak)
    efektif : (1 − β) / (1 − β^n_k)           (Cui dkk., 2019)
    tanpa   : 1
    """
    n = np.maximum(hitung_kelas(y, n_kelas), 1)
    if jenis == "penuh":
        w = 1 / n
    elif jenis == "akar":
        w = 1 / np.sqrt(n)
    elif jenis == "efektif":
        w = (1 - beta) / (1 - np.power(beta, n))
    elif jenis == "tanpa":
        w = np.ones_like(n)
    else:
        raise ValueError(f"jenis bobot tidak dikenal: {jenis}")
    return w * n.sum() / (w * n).sum()


def focal_loss(logit, y, gamma=2.0, weight=None):
    """Focal loss multikelas: −α_y (1 − p_y)^γ log p_y, dirata-rata per sampel."""
    logp = F.log_softmax(logit, dim=1)
    logp_y = logp.gather(1, y[:, None]).squeeze(1)
    loss = -((1 - logp_y.exp()) ** gamma) * logp_y
    if weight is not None:
        loss = loss * weight[y]
    return loss.mean()


def logit_adjusted_loss(logit, y, log_prior, tau=1.0):
    """Cross-entropy pada logit + τ·log π (Menon dkk., 2021). Prediksi tetap memakai logit asli."""
    return F.cross_entropy(logit + tau * log_prior, y)


def indeks_sampler_seimbang(idx, y, n_kelas, gen):
    """Ambil len(idx) indeks dengan pengembalian, peluang ∝ 1/n_k → setiap kelas rata-rata sama banyak."""
    idx = np.asarray(idx)
    n = np.maximum(hitung_kelas(y[idx], n_kelas), 1)
    p = 1 / n[y[idx]]
    p = torch.tensor(p / p.sum(), dtype=torch.float64)
    pilih = torch.multinomial(p, len(idx), replacement=True, generator=gen).numpy()
    return idx[pilih]


def koreksi_prior(prob, prior, tau=1.0):
    """Koreksi post-hoc (Menon dkk., 2021): p_k / π_k^τ lalu dinormalisasi ulang."""
    q = prob / np.power(np.asarray(prior), tau)[None, :]
    return q / q.sum(axis=1, keepdims=True)


def cari_bias_bacc(prob, y, n_kelas, rentang=3.0, langkah=0.1, putaran=3):
    """Bias aditif per kelas pada log-probabilitas yang memaksimalkan BACC (coordinate ascent).

    HANYA boleh dipanggil dengan data validasi dalam, lalu diterapkan ke data uji
    lewat terapkan_bias.
    """
    logp = np.log(np.clip(prob, 1e-12, 1))
    b = np.zeros(n_kelas)
    grid = np.arange(-rentang, rentang + 1e-9, langkah)
    terbaik = balanced_accuracy_score(y, (logp + b).argmax(1))
    for _ in range(putaran):
        for k in range(n_kelas):
            for g in grid:
                c = b.copy()
                c[k] = g
                s = balanced_accuracy_score(y, (logp + c).argmax(1))
                if s > terbaik + 1e-12:
                    terbaik, b = s, c
    return b, float(terbaik)


def terapkan_bias(prob, b):
    logp = np.log(np.clip(prob, 1e-12, 1)) + b[None, :]
    q = np.exp(logp - logp.max(axis=1, keepdims=True))
    return q / q.sum(axis=1, keepdims=True)
