"""Tes citra/attention.py: identitas saat awal, bentuk keluaran, dan grup optimizer."""

import pytest

torch = pytest.importorskip("torch")

from citra.attention import Bergerbang, CBAM, CoordAtt, gerbang_attention, pasang_attention  # noqa: E402
from citra.derm import DermMobileNetV3, hitung_biaya  # noqa: E402

VARIAN = [("cbam", "akhir"), ("cbam", "blok"), ("ca", "akhir"), ("ca", "blok")]


def _pasangan(jenis, letak):
    torch.manual_seed(0)
    a0 = DermMobileNetV3(pralatih=False).eval()
    torch.manual_seed(0)
    m = pasang_attention(DermMobileNetV3(pralatih=False), jenis, letak).eval()
    return a0, m


@pytest.mark.parametrize("jenis,letak", VARIAN)
def test_identitas_saat_init(jenis, letak):
    a0, m = _pasangan(jenis, letak)
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(m(x), a0(x))


@pytest.mark.parametrize("jenis,letak", VARIAN)
def test_gerbang_mendapat_gradien_dan_grup_benar(jenis, letak):
    _, m = _pasangan(jenis, letak)
    m.train()
    m(torch.rand(2, 3, 224, 224)).sum().backward()
    gerbang = [g for g in m.modules() if isinstance(g, Bergerbang)]
    assert gerbang and all(g.gerbang.grad is not None and g.gerbang.grad.abs() > 0 for g in gerbang)
    grup = m.kelompok_parameter(1e-4, 1e-3)
    id_dasar = {id(p) for p in grup[0]["params"]}
    assert not id_dasar & m._id_modul_tambahan              # modul baru tidak ikut lr dasar
    id_tanpa_wd = {id(p) for g in grup if g["weight_decay"] == 0 for p in g["params"]}
    assert all(id(g.gerbang) in id_tanpa_wd for g in gerbang)
    assert sum(len(g["params"]) for g in grup) == len(list(m.parameters()))
    assert "attn_gerbang_rata" in gerbang_attention(m)
    assert 0 < hitung_biaya(m)["parameter_tambahan"] < 1_000_000


@pytest.mark.parametrize("modul", [CBAM(64), CoordAtt(64)])
def test_bentuk_keluaran(modul):
    x = torch.rand(2, 64, 14, 10)
    assert modul(x).shape == x.shape
