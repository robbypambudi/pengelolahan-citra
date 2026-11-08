"""Tes Derm-MobileNetV3 (citra/derm.py). Tanpa bobot pralatih agar tidak butuh internet."""

import pytest

torch = pytest.importorskip("torch")

from citra.derm import (  # noqa: E402
    DermMobileNetV3, PelemahRambut, PoolingLesiSekitar, ShadesOfGrayTerlatih,
    buat_varian_lilhore, hitung_biaya, mask_cincin,
)


def _input(b=2, s=224, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(b, 3, s, s, generator=g)
    mask = torch.zeros(b, 1, s, s)
    mask[..., 60:160, 70:150] = 1
    return x, mask


# ---------------------------------------------------------------- identitas saat inisialisasi
@pytest.mark.parametrize("modul", [ShadesOfGrayTerlatih(), PelemahRambut()])
def test_m1_identitas_saat_init(modul):
    x, _ = _input()
    torch.testing.assert_close(modul(x), x)


@pytest.mark.parametrize("cfg", [
    dict(m1_cc=True, m1_rambut=True),
    dict(m3=True),
    dict(m3=True, m3_mode="cincin", lapisan_m3=12),
    dict(m2=True),
    dict(m1_cc=True, m1_rambut=True, m2=True, m3=True),
])
def test_model_identik_dengan_a0_saat_init(cfg):
    """Syarat desain: modul baru tidak mengubah keluaran sebelum dilatih."""
    torch.manual_seed(0)
    derm = DermMobileNetV3(pralatih=False, **cfg).eval()
    a0 = DermMobileNetV3(pralatih=False).eval()
    hilang, _ = a0.load_state_dict(derm.state_dict(), strict=False)
    assert not hilang
    x, mask = _input()
    with torch.no_grad():
        torch.testing.assert_close(derm(x, mask), a0(x))


def test_gradien_mengalir_ke_semua_gerbang():
    torch.manual_seed(0)
    model = DermMobileNetV3(pralatih=False, m1_cc=True, m1_rambut=True, m2=True, m3=True).train()
    x, mask = _input()
    model(x, mask).logsumexp(dim=1).sum().backward()
    for nama in ["cc.gerbang", "rambut.gerbang",
                 "cabang_m2.proyeksi.2.weight", "cabang_m3.proyeksi.2.weight"]:
        grad = model.get_parameter(nama).grad
        assert grad is not None and grad.abs().sum() > 0, nama


def test_m3_wajib_mask():
    model = DermMobileNetV3(pralatih=False, m3=True).eval()
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 224, 224))


# ---------------------------------------------------------------- M1
def test_shades_of_gray_menghapus_cast_warna():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64) * 0.5
    x[:, 0] *= 1.6   # cast merah
    sog = ShadesOfGrayTerlatih(p_awal=6.0)
    with torch.no_grad():
        sog.gerbang.fill_(1.0)
        y = sog(x)
    p = float(sog.p.detach())
    e = y.pow(p).mean(dim=(2, 3)).pow(1 / p)[0]
    assert (e.max() - e.min()) / e.mean() < 1e-3
    assert abs(p - 6.0) < 1e-5


def test_pelemah_rambut_menghapus_garis_gelap_tipis():
    x = torch.full((1, 3, 64, 64), 0.8)
    x[:, :, 31:33, :] = 0.1          # "rambut" horizontal setebal 2 px
    rambut = PelemahRambut(k=7)
    m = rambut.soft_mask(x)
    assert m[0, 0, 32, 32] > 0.99 and m[0, 0, 10, 32] < 0.01
    with torch.no_grad():
        rambut.gerbang.fill_(1.0)
        y = rambut(x)
    assert torch.allclose(y[:, :, 32, 20:44], torch.full_like(y[:, :, 32, 20:44], 0.8), atol=1e-3)


# ---------------------------------------------------------------- M3
def test_cincin_tidak_beririsan_dengan_lesi():
    _, mask = _input(b=1)
    cincin = mask_cincin(mask, 8)
    assert float((cincin * mask).sum()) == 0
    assert float(cincin.sum()) > 0
    assert float(cincin[0, 0, 60 - 8, 100]) == 1 and float(cincin[0, 0, 60 - 9, 100]) == 0


def test_pooling_lesi_sekitar_memisahkan_nilai():
    """Peta fitur bernilai 5 di dalam lesi dan 1 di luar: f_lesi=5, f_cincin=1, selisih=4."""
    s = 56
    mask = torch.zeros(1, 1, s, s)
    mask[..., 16:40, 16:40] = 1
    f = 1 + 4 * mask.expand(1, 4, s, s)
    pool = PoolingLesiSekitar(c_in=4, lebar_cincin_px=6)
    v = pool(f, mask)[0]
    torch.testing.assert_close(v[:4], torch.full((4,), 5.0))
    torch.testing.assert_close(v[4:8], torch.full((4,), 1.0))
    torch.testing.assert_close(v[8:], torch.full((4,), 4.0))


def test_pooling_mask_kosong_tidak_nan():
    pool = PoolingLesiSekitar(c_in=8)
    v = pool(torch.rand(2, 8, 7, 7), torch.zeros(2, 1, 224, 224))
    assert torch.isfinite(v).all()


@pytest.mark.parametrize("mode,kali", [("penuh", 3), ("lesi", 1), ("cincin", 1)])
def test_dimensi_mode_m3(mode, kali):
    model = DermMobileNetV3(pralatih=False, m3=True, m3_mode=mode).eval()
    assert model.m3.dim_keluar == 960 * kali
    x, mask = _input(b=1)
    assert model(x, mask).shape == (1, 7)


# ---------------------------------------------------------------- AL & biaya
def test_varian_lilhore_tanpa_se():
    from torchvision.ops.misc import SqueezeExcitation

    model = buat_varian_lilhore(pralatih=False).eval()
    assert not any(isinstance(m, SqueezeExcitation) for m in model.modules())
    assert model(torch.rand(1, 3, 224, 224)).shape == (1, 7)


def test_biaya_modul_tambahan():
    a0 = hitung_biaya(DermMobileNetV3(pralatih=False))
    derm = hitung_biaya(DermMobileNetV3(pralatih=False, m1_cc=True, m1_rambut=True, m3=True))
    assert a0["parameter_tambahan"] == 0
    assert derm["parameter_tambahan"] > 0
    assert derm["parameter"] == a0["parameter"] + derm["parameter_tambahan"]


def test_kelompok_parameter_mencakup_semua():
    model = DermMobileNetV3(pralatih=False, m1_cc=True, m1_rambut=True, m2=True, m3=True)
    grup = model.kelompok_parameter(1e-4, 1e-3)
    ids = [id(p) for g in grup for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    gerbang = {id(model.cc.gerbang), id(model.cabang_m3.gerbang)}
    assert all(g["weight_decay"] == 0 for g in grup if gerbang & {id(p) for p in g["params"]})
