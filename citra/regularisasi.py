"""Regularisasi pelatihan (eksperimen 05): mixup, cutmix, stochastic depth, tukar latar.

Semua fungsi batch bekerja di GPU setelah augmentasi geometri, dengan generator eksplisit
agar run bisa direproduksi.

Rujukan:
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk
  minimization. ICLR.  → mixup_batch
- Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix: Regularization
  strategy to train strong classifiers with localizable features. ICCV.  → cutmix_batch
- Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep networks with
  stochastic depth. ECCV.  → pasang_stochastic_depth
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the
  inception architecture for computer vision. CVPR.  → label smoothing (lewat F.cross_entropy)
- tukar_latar: augmentasi kami sendiri, dimotivasi temuan H5d (eksperimen 03) dan
  Bissoto dkk. (2019) tentang jalan pintas lewat latar.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.ops import StochasticDepth


def _beta(alpha, gen):
    return float(np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen))).beta(alpha, alpha))


def _perm(b, gen, device):
    return torch.randperm(b, generator=gen).to(device)


def mixup_batch(x, gen, alpha=0.2):
    """Kembalikan (x_campur, perm, λ). Loss = λ·L(y) + (1 − λ)·L(y[perm])."""
    lam = _beta(alpha, gen)
    perm = _perm(x.shape[0], gen, x.device)
    return lam * x + (1 - lam) * x[perm], perm, lam


def cutmix_batch(x, gen, alpha=1.0):
    """Tempel kotak acak dari x[perm]. λ = porsi piksel asli (dikoreksi setelah kotak dipotong tepi)."""
    lam = _beta(alpha, gen)
    perm = _perm(x.shape[0], gen, x.device)
    _, _, h, w = x.shape
    r = np.sqrt(1 - lam)
    ch, cw = int(h * r), int(w * r)
    cy, cx = (int(v) for v in torch.randint(0, h, (1,), generator=gen).tolist() + torch.randint(0, w, (1,), generator=gen).tolist())
    y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
    return x, perm, lam


def pasang_stochastic_depth(model, p_maks=0.2):
    """Bungkus cabang residual setiap InvertedResidual (yang punya skip) dengan StochasticDepth.

    Peluang drop naik linear terhadap kedalaman sampai p_maks (Huang dkk., 2016). Saat eval
    tidak berpengaruh. Bobot pralatih tetap (hanya membungkus).
    """
    blok = [b for b in model.features if isinstance(b, InvertedResidual) and b.use_res_connect]
    for i, b in enumerate(blok):
        b.block = nn.Sequential(b.block, StochasticDepth(p_maks * (i + 1) / len(blok), "row"))
    return model


def _blur_ternormalisasi(x, w, k=31, ulang=2):
    """Konvolusi ternormalisasi: rata-rata x hanya dari piksel ber-bobot w (inpainting kasar)."""
    a, b = x * w, w
    for _ in range(ulang):
        a = F.avg_pool2d(a, k, stride=1, padding=k // 2, count_include_pad=False)
        b = F.avg_pool2d(b, k, stride=1, padding=k // 2, count_include_pad=False)
    return a / b.clamp_min(1e-3), b


def tukar_latar(x, m, gen, p=0.5, dilasi_px=16, tepi_px=9):
    """Tempel lesi (mask didilasi + tepi lembut) ke kulit citra lain yang lesinya sudah dihapus.

    Label tetap label lesi. Tujuannya memutus hubungan latar ↔ kelas (H5d).
    1. Daerah lesi donor latar (mask + dilasi) diisi rata-rata ternormalisasi kulit di sekitarnya.
    2. Lesi citra i (mask + dilasi, tepi diperhalus) ditempel di atas latar bersih itu.
    Hanya diterapkan pada porsi p dari batch.
    """
    b = x.shape[0]
    pakai = (torch.rand(b, generator=gen) < p).to(x.device)
    if not pakai.any():
        return x
    k = 2 * dilasi_px + 1
    md = F.max_pool2d(m, k, stride=1, padding=k // 2)
    isi, cakupan = _blur_ternormalisasi(x, 1 - md)
    rata_kulit = (x * (1 - md)).sum((2, 3), keepdim=True) / (1 - md).sum((2, 3), keepdim=True).clamp_min(1)
    isi = torch.where(cakupan > 1e-3, isi, rata_kulit)            # lesi sangat besar → warna kulit rata-rata
    bersih = x * (1 - md) + isi * md
    alpha = F.avg_pool2d(md, tepi_px, stride=1, padding=tepi_px // 2, count_include_pad=False)
    perm = _perm(b, gen, x.device)
    hasil = alpha * x + (1 - alpha) * bersih[perm]
    return torch.where(pakai[:, None, None, None], hasil, x)
