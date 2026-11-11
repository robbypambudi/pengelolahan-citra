"""Modul attention untuk MobileNetV3-Large (eksperimen 04): CBAM dan Coordinate Attention.

Rujukan:
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention
  module. ECCV.  → AttnKanal, AttnSpasial, CBAM
- Hou, Q., Zhou, D., & Feng, J. (2021). Coordinate attention for efficient mobile network
  design. CVPR.  → CoordAtt

Pelajaran dari eksperimen 01–03, diterapkan di sini:
1. Setiap modul dibungkus `Bergerbang`: keluaran = x + g·(attn(x) − x), dengan g = 0 saat awal.
   Model identik dengan A0 pralatih di langkah ke-0 (diuji di tests/test_attention.py).
2. SE pralatih TIDAK diganti. Varian AL (SE→ECA) kehilangan bobot SE pralatih dan kalah.
   Di dalam blok, SE dipertahankan sebagai attention kanal, lalu ditambah attention spasial
   sesudahnya. Urutan kanal → spasial sama dengan CBAM, tetapi bagian kanalnya memakai SE pralatih.
3. Parameter baru didaftarkan di `model._id_modul_tambahan`, agar masuk grup optimizer
   modul (lr_modul; gerbang tanpa weight decay) walaupun letaknya di dalam `features`.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.ops.misc import SqueezeExcitation


class AttnKanal(nn.Module):
    """CBAM channel attention: MLP bersama pada avg- dan max-pool, lalu sigmoid."""

    def __init__(self, c, r=16):
        super().__init__()
        h = max(8, c // r)
        self.mlp = nn.Sequential(nn.Conv2d(c, h, 1), nn.ReLU(inplace=True), nn.Conv2d(h, c, 1))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(a)


class AttnSpasial(nn.Module):
    """CBAM spatial attention: konvolusi k×k pada [rata-rata kanal, maks kanal], lalu sigmoid."""

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(s))


class CBAM(nn.Module):
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.kanal, self.spasial = AttnKanal(c, r), AttnSpasial(k)

    def forward(self, x):
        return self.spasial(self.kanal(x))


class CoordAtt(nn.Module):
    """Coordinate attention: pooling per baris dan per kolom → 1×1 conv bersama → dua peta attention."""

    def __init__(self, c, r=32):
        super().__init__()
        h = max(8, c // r)
        self.reduksi = nn.Sequential(nn.Conv2d(c, h, 1), nn.BatchNorm2d(h), nn.Hardswish())
        self.ke_h, self.ke_w = nn.Conv2d(h, c, 1), nn.Conv2d(h, c, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        ph = x.mean(3, keepdim=True)                       # (B, C, H, 1)
        pw = x.mean(2, keepdim=True).transpose(2, 3)       # (B, C, W, 1)
        y = self.reduksi(torch.cat([ph, pw], 2))
        yh, yw = torch.split(y, [H, W], 2)
        return x * torch.sigmoid(self.ke_h(yh)) * torch.sigmoid(self.ke_w(yw.transpose(2, 3)))


class Bergerbang(nn.Module):
    """x + g·(inti(x) − x); g = 0 saat awal → identitas."""

    def __init__(self, inti):
        super().__init__()
        self.inti = inti
        self.gerbang = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x + self.gerbang * (self.inti(x) - x)


def _daftarkan(model, modul):
    model._id_modul_tambahan = getattr(model, "_id_modul_tambahan", set()) | {id(p) for p in modul.parameters()}


def pasang_attention(model, jenis, letak):
    """Pasang attention pada DermMobileNetV3 (in-place) dan kembalikan model.

    jenis : "cbam" | "ca" (coordinate attention)
    letak : "akhir" → satu modul pada peta 960 kanal (blok 16), sebelum GAP
            "blok"  → di setiap blok ber-SE (blok 4–6, 11–15) sesudah SE:
                      cbam → attention spasial saja (SE pralatih berperan sebagai bagian kanal CBAM)
                      ca   → coordinate attention
    """
    if letak == "akhir":
        c = model.classifier[0].in_features
        modul = Bergerbang(CBAM(c) if jenis == "cbam" else CoordAtt(c))
        model.attn_akhir = modul
        _daftarkan(model, modul)
    elif letak == "blok":
        n = 0
        for blok in model.features:
            if not isinstance(blok, InvertedResidual):
                continue
            for i, lapis in enumerate(blok.block):
                if isinstance(lapis, SqueezeExcitation):
                    c = lapis.fc1.in_channels
                    modul = Bergerbang(AttnSpasial() if jenis == "cbam" else CoordAtt(c))
                    blok.block[i] = nn.Sequential(lapis, modul)
                    _daftarkan(model, modul)
                    n += 1
        if n == 0:
            raise ValueError("tidak ada blok ber-SE")
    else:
        raise ValueError(f"letak tidak dikenal: {letak}")
    return model


@torch.no_grad()
def gerbang_attention(model):
    g = [float(m.gerbang) for m in model.modules() if isinstance(m, Bergerbang)]
    if not g:
        return {}
    return {"attn_gerbang_rata": sum(g) / len(g), "attn_gerbang_min": min(g), "attn_gerbang_maks": max(g)}
