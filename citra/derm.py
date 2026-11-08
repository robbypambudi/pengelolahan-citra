"""Derm-MobileNetV3: MobileNetV3-Large + prior pengolahan citra yang bisa dilatih (RQ5).

Desain: `riset/PROPOSAL.md` (RQ5, H5a–H5d) dan `riset/LITERATUR-DERM-MOBILENETV3.md`.

Prinsip utama: blok MobileNetV3-Large asli tidak diubah. Setiap modul ditambahkan sehingga
pada langkah ke-0 keluaran Derm-MobileNetV3 identik dengan MobileNetV3-Large polos (A0):
M1 lewat gerbang skalar = 0, M2/M3 lewat Linear terakhir = 0. Ini diuji di tests/test_derm.py.

Modul:
  M1a  ShadesOfGrayTerlatih  — color constancy Shades of Gray, eksponen p dipelajari
                               (estimasi iluminan = GeM pooling per kanal RGB).
  M1b  PelemahRambut         — black-hat morfologi + soft-mask rambut yang dipelajari;
                               piksel rambut diganti closing per kanal.
  M2   TeksturOrde2          — (opsional) covariance pooling dari blok tengah.
  M3   PoolingLesiSekitar    — pooling dalam mask lesi, di cincin kulit sekitar, dan selisihnya.

Konvensi input: x float di [0, 1], RGB, (B, 3, H, W). Normalisasi ImageNet dilakukan
di dalam model, setelah M1. mask: (B, 1, H, W) float {0,1} pada resolusi input.

Membutuhkan extra `dl`:  uv sync --extra dl
"""

import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _softplus_inv(y):
    return math.log(math.expm1(y))


def _dilasi(x, k):
    return F.max_pool2d(x, k, stride=1, padding=k // 2)


def _erosi(x, k):
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)


def closing(x, k):
    """Closing morfologi dengan elemen struktur persegi k×k (k ganjil), per kanal."""
    return _erosi(_dilasi(x, k), k)


# --------------------------------------------------------------------------- M1a
class ShadesOfGrayTerlatih(nn.Module):
    """Shades of Gray dengan eksponen Minkowski p yang dipelajari.

    e_c = (mean_xy x_c^p)^(1/p)       (p=1 Gray World, p→∞ max-RGB)
    x'_c = x_c * ē / e_c,  ē = rata-rata e_c antar kanal (kecerahan dipertahankan)
    keluaran = x + g * (x' − x), gerbang g diinisialisasi 0 (identitas).

    p = 1 + softplus(θ) ≥ 1, diinisialisasi p_awal (6, nilai yang lazim di dermoskopi;
    Barata dkk., 2015).
    """

    def __init__(self, p_awal=6.0, eps=1e-6):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(_softplus_inv(p_awal - 1.0)))
        self.gerbang = nn.Parameter(torch.zeros(()))
        self.eps = eps

    @property
    def p(self):
        return 1.0 + F.softplus(self.theta)

    def koreksi(self, x):
        p = self.p
        e = x.clamp_min(self.eps).pow(p).mean(dim=(2, 3), keepdim=True).pow(1.0 / p)
        e_rata = e.mean(dim=1, keepdim=True)
        return (x * e_rata / e.clamp_min(self.eps)).clamp(0.0, 1.0)

    def forward(self, x):
        return x + self.gerbang * (self.koreksi(x) - x)


# --------------------------------------------------------------------------- M1b
class PelemahRambut(nn.Module):
    """Pelemah rambut gaya DullRazor yang bisa diturunkan.

    abu = rata-rata kanal; bh = closing_k(abu) − abu  (black-hat: struktur gelap tipis)
    m   = sigmoid(a · (bh − t))                       (soft-mask rambut; a, t dipelajari)
    x'  = (1 − g·m) · x + g·m · closing_k(x)          (gerbang g diinisialisasi 0)

    Closing per kanal berfungsi sebagai inpainting murah: struktur gelap yang lebih
    tipis dari k piksel diganti warna sekitarnya.
    """

    def __init__(self, k=11, a_awal=100.0, t_awal=0.08):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("k harus ganjil")
        self.k = k
        self.log_a = nn.Parameter(torch.tensor(math.log(a_awal)))
        self.t = nn.Parameter(torch.tensor(t_awal))
        self.gerbang = nn.Parameter(torch.zeros(()))

    def soft_mask(self, x):
        abu = x.mean(dim=1, keepdim=True)
        bh = closing(abu, self.k) - abu
        return torch.sigmoid(self.log_a.exp() * (bh - self.t))

    def forward(self, x):
        m = self.gerbang * self.soft_mask(x)
        return (1.0 - m) * x + m * closing(x, self.k)


# --------------------------------------------------------------------------- M2
class TeksturOrde2(nn.Module):
    """Covariance pooling ringan: 1×1 conv → kovarians d×d → segitiga atas → √bertanda → L2."""

    def __init__(self, c_in, d=32):
        super().__init__()
        self.reduksi = nn.Sequential(nn.Conv2d(c_in, d, 1, bias=False), nn.BatchNorm2d(d))
        self.idx = torch.triu_indices(d, d)
        self.dim_keluar = self.idx.shape[1]

    def forward(self, f):
        x = self.reduksi(f).flatten(2)                       # (B, d, N)
        x = x - x.mean(dim=2, keepdim=True)
        cov = x @ x.transpose(1, 2) / max(x.shape[2] - 1, 1)  # (B, d, d)
        v = cov[:, self.idx[0], self.idx[1]]
        v = v.sign() * v.abs().add(1e-8).sqrt()
        return F.normalize(v, dim=1)


# --------------------------------------------------------------------------- M3
def mask_cincin(mask, lebar_px):
    """Cincin kulit sekitar = dilasi(mask, lebar) − mask, pada resolusi input."""
    return (_dilasi(mask, 2 * lebar_px + 1) - mask).clamp(0.0, 1.0)


class PoolingLesiSekitar(nn.Module):
    """Pooling fitur berpanduan mask: dalam lesi, cincin kulit sekitar, dan selisihnya.

    Mask lesi dan cincin dihitung pada resolusi input lalu diturunkan ke resolusi peta fitur
    dengan interpolasi area, sehingga menjadi bobot lunak (fraksi luas per sel fitur).

    mode: "penuh" → [f_lesi, f_cincin, f_lesi − f_cincin]   (A3)
          "lesi"  → [f_lesi]                                 (A3-L)
          "cincin"→ [f_cincin]                               (A3-C)
    """

    MODE = {"penuh": 3, "lesi": 1, "cincin": 1}

    def __init__(self, c_in, lebar_cincin_px=16, mode="penuh", eps=1e-6):
        super().__init__()
        if mode not in self.MODE:
            raise ValueError(f"mode harus salah satu dari {list(self.MODE)}")
        self.lebar = lebar_cincin_px
        self.mode = mode
        self.eps = eps
        self.dim_keluar = c_in * self.MODE[mode]

    def _pool(self, f, w):
        return (f * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + self.eps)

    def bobot(self, mask, ukuran):
        cincin = mask_cincin(mask, self.lebar)
        w_lesi = F.adaptive_avg_pool2d(mask, ukuran)
        w_cincin = F.adaptive_avg_pool2d(cincin, ukuran)
        return w_lesi, w_cincin

    def forward(self, f, mask):
        w_lesi, w_cincin = self.bobot(mask, f.shape[-2:])
        f_lesi, f_cincin = self._pool(f, w_lesi), self._pool(f, w_cincin)
        if self.mode == "lesi":
            return f_lesi
        if self.mode == "cincin":
            return f_cincin
        return torch.cat([f_lesi, f_cincin, f_lesi - f_cincin], dim=1)


class _CabangBergerbang(nn.Module):
    """LayerNorm → Linear(bottleneck) → Hardswish → Linear ke dimensi GAP, dikali gerbang skalar.

    Inisialisasi: Linear terakhir = 0, gerbang = 1. Keluaran awal tetap 0 (model identik
    dengan A0), tetapi bobot Linear terakhir langsung mendapat gradien.

    Catatan versi: rancangan awal memakai gerbang = 0 dengan Linear acak (run
    hasil/20261108-220753, dihentikan). Cabang tetap belajar karena AdamW menormalkan
    skala gradien, tetapi lambat: kontribusi M3 ≈ 5% → 19% norma GAP dalam 4 epoch,
    vs ≈ 48% dengan inisialisasi ini (uji 3.000 citra, 8 Nov 2026).

    Bottleneck menjaga tambahan parameter tetap kecil (M3 penuh: 2.880 → 128 → 960).
    """

    def __init__(self, dim_in, dim_keluar, bottleneck=128):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.proyeksi = nn.Sequential(
            nn.Linear(dim_in, bottleneck), nn.Hardswish(), nn.Linear(bottleneck, dim_keluar))
        nn.init.zeros_(self.proyeksi[-1].weight)
        nn.init.zeros_(self.proyeksi[-1].bias)
        self.gerbang = nn.Parameter(torch.ones(()))

    def forward(self, v):
        return self.gerbang * self.proyeksi(self.norm(v))


# --------------------------------------------------------------------------- model
class DermMobileNetV3(nn.Module):
    """MobileNetV3-Large + M1/M2/M3 opsional. Semua False = A0 (MobileNetV3-Large polos).

    Parameter
    ---------
    n_kelas         jumlah kelas (7 untuk HAM10000)
    pralatih        muat bobot ImageNet torchvision (IMAGENET1K_V1); butuh internet saat pertama
    m1_cc, m1_rambut  aktifkan M1a / M1b
    m2              aktifkan M2 (opsional, bukan klaim kebaruan)
    m3              aktifkan M3; forward lalu wajib menerima mask
    m3_mode         "penuh" | "lesi" | "cincin"
    lapisan_m2      indeks blok `features` sumber M2 (12 → 112 kanal, 14×14 pada input 224)
    lapisan_m3      indeks blok sumber M3 (16 → 960 kanal, 7×7; 12 → 14×14 lebih halus)
    lebar_cincin_px lebar cincin kulit sekitar pada resolusi input
    bottleneck      dimensi tengah cabang M2/M3 sebelum dijumlahkan ke GAP
    """

    def __init__(self, n_kelas=7, pralatih=True, m1_cc=False, m1_rambut=False, m2=False, m3=False,
                 m3_mode="penuh", lapisan_m2=12, lapisan_m3=16, lebar_cincin_px=16,
                 k_rambut=11, p_awal=6.0, d_m2=32, bottleneck=128, dilated=False):
        super().__init__()
        bobot = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pralatih else None
        dasar = torchvision.models.mobilenet_v3_large(weights=bobot, dilated=dilated)
        self.features = dasar.features
        self.avgpool = dasar.avgpool
        self.classifier = dasar.classifier
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, n_kelas)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        c_gap = self.classifier[0].in_features  # 960
        self.rambut = PelemahRambut(k=k_rambut) if m1_rambut else None
        self.cc = ShadesOfGrayTerlatih(p_awal=p_awal) if m1_cc else None

        self.lapisan_m2, self.lapisan_m3 = lapisan_m2, lapisan_m3
        self.m2 = self.cabang_m2 = None
        if m2:
            self.m2 = TeksturOrde2(self._kanal(lapisan_m2), d=d_m2)
            self.cabang_m2 = _CabangBergerbang(self.m2.dim_keluar, c_gap, bottleneck)
        self.m3 = self.cabang_m3 = None
        if m3:
            self.m3 = PoolingLesiSekitar(self._kanal(lapisan_m3), lebar_cincin_px, m3_mode)
            self.cabang_m3 = _CabangBergerbang(self.m3.dim_keluar, c_gap, bottleneck)

    def _kanal(self, i):
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            for j, blok in enumerate(self.features):
                x = blok(x)
                if j == i:
                    return x.shape[1]
        raise IndexError(i)

    @property
    def butuh_mask(self):
        return self.m3 is not None

    def praproses(self, x):
        if self.rambut is not None:
            x = self.rambut(x)
        if self.cc is not None:
            x = self.cc(x)
        return x

    def forward(self, x, mask=None):
        if self.butuh_mask and mask is None:
            raise ValueError("M3 aktif: forward(x, mask) wajib menerima mask lesi")
        with torch.autocast(device_type=x.device.type, enabled=False):   # M1 (pow, closing) di float32
            h = (self.praproses(x.float()) - self.mean) / self.std
        f_m2 = f_m3 = None
        for j, blok in enumerate(self.features):
            h = blok(h)
            if self.m2 is not None and j == self.lapisan_m2:
                f_m2 = h
            if self.m3 is not None and j == self.lapisan_m3:
                f_m3 = h
        z = torch.flatten(self.avgpool(h), 1)
        if self.m2 is not None:
            z = z + self.cabang_m2(self.m2(f_m2))
        if self.m3 is not None:
            z = z + self.cabang_m3(self.m3(f_m3, mask))
        return self.classifier(z)

    def kelompok_parameter(self, lr_dasar, lr_modul, weight_decay=1e-4):
        """Grup optimizer: backbone + classifier (lr_dasar) vs modul baru (lr_modul).

        Gerbang dan parameter skalar M1 (p, a, t) tidak diberi weight decay,
        agar tidak ditarik kembali ke nol secara artifisial.
        """
        dasar, modul, tanpa_wd = [], [], []
        for nama, p in self.named_parameters():
            if nama.startswith(("features.", "classifier.")):
                dasar.append(p)
            elif p.ndim == 0:
                tanpa_wd.append(p)
            else:
                modul.append(p)
        grup = [{"params": dasar, "lr": lr_dasar, "weight_decay": weight_decay},
                {"params": modul, "lr": lr_modul, "weight_decay": weight_decay},
                {"params": tanpa_wd, "lr": lr_modul, "weight_decay": 0.0}]
        return [g for g in grup if g["params"]]

    @torch.no_grad()
    def parameter_terpelajari(self):
        """Nilai p, gerbang, dan ambang yang dipelajari, untuk dicatat ke metrics.json."""
        out = {}
        if self.cc is not None:
            out["m1_cc_p"] = float(self.cc.p)
            out["m1_cc_gerbang"] = float(self.cc.gerbang)
        if self.rambut is not None:
            out["m1_rambut_a"] = float(self.rambut.log_a.exp())
            out["m1_rambut_t"] = float(self.rambut.t)
            out["m1_rambut_gerbang"] = float(self.rambut.gerbang)
        for nama, cabang in (("m2", self.cabang_m2), ("m3", self.cabang_m3)):
            if cabang is not None:
                out[f"{nama}_gerbang"] = float(cabang.gerbang)
                out[f"{nama}_norma_bobot_akhir"] = float(cabang.proyeksi[-1].weight.norm())
        return out


# --------------------------------------------------------------------------- AL
class ECA(nn.Module):
    """Efficient Channel Attention (Wang dkk., 2020): GAP → conv1d k adaptif → sigmoid."""

    def __init__(self, c, gamma=2, b=1):
        super().__init__()
        k = int(abs(math.log2(c) / gamma + b / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)   # (B, 1, C)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


def buat_varian_lilhore(n_kelas=7, pralatih=True):
    """AL: pendekatan modifikasi blok Lilhore dkk. (2024) untuk pembanding ablasi.

    Diimplementasikan: SE → ECA, dilated conv di tahap akhir (opsi `dilated` torchvision).
    Dihilangkan (tidak dirinci di paper): cross-layer connection, versi "lebih dalam/lebar",
    bias loss. Catat di tabel kesetiaan.
    """
    from torchvision.ops.misc import SqueezeExcitation

    model = DermMobileNetV3(n_kelas=n_kelas, pralatih=pralatih, dilated=True)
    for nama, modul in list(model.features.named_modules()):
        for anak_nama, anak in list(modul.named_children()):
            if isinstance(anak, SqueezeExcitation):
                setattr(modul, anak_nama, ECA(anak.fc1.in_channels))
    return model


# --------------------------------------------------------------------------- biaya
def hitung_biaya(model, ukuran=224, perangkat="cpu"):
    """Jumlah parameter (total, tambahan dari modul baru) dan FLOPs satu citra."""
    from torch.utils.flop_counter import FlopCounterMode

    model = model.to(perangkat).eval()
    x = torch.rand(1, 3, ukuran, ukuran, device=perangkat)
    mask = torch.zeros(1, 1, ukuran, ukuran, device=perangkat)
    mask[..., ukuran // 4: 3 * ukuran // 4, ukuran // 4: 3 * ukuran // 4] = 1
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x, mask) if getattr(model, "butuh_mask", False) else model(x)
    total = sum(p.numel() for p in model.parameters())
    dasar = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("features.", "classifier.")))
    return {"parameter": total, "parameter_tambahan": total - dasar, "flops": fc.get_total_flops()}
