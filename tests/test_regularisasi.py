"""Tes citra/regularisasi.py."""

import pytest

torch = pytest.importorskip("torch")

from citra.derm import DermMobileNetV3  # noqa: E402
from citra.regularisasi import cutmix_batch, mixup_batch, pasang_stochastic_depth, tukar_latar  # noqa: E402


def _batch(b=4, s=64):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(b, 3, s, s, generator=g)
    m = torch.zeros(b, 1, s, s)
    m[..., 24:40, 24:40] = 1
    return x, m


def test_mixup_kombinasi_konveks():
    x, _ = _batch()
    xm, perm, lam = mixup_batch(x, torch.Generator().manual_seed(1))
    assert 0 <= lam <= 1
    torch.testing.assert_close(xm, lam * x + (1 - lam) * x[perm])


def test_cutmix_lambda_sesuai_porsi_piksel():
    x, _ = _batch()
    xc, perm, lam = cutmix_batch(x, torch.Generator().manual_seed(2))
    sama = (xc == x).all(1).float().mean((1, 2))           # porsi piksel asli per citra
    beda = perm != torch.arange(len(x))
    assert torch.allclose(sama[beda], torch.full_like(sama[beda], lam), atol=1e-6)


def test_stochastic_depth_identitas_saat_eval():
    torch.manual_seed(0)
    a = DermMobileNetV3(pralatih=False).eval()
    torch.manual_seed(0)
    b = pasang_stochastic_depth(DermMobileNetV3(pralatih=False), 0.2).eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x))
    b.train()
    with torch.no_grad():
        assert not torch.allclose(b(x), b(x))               # acak saat latih


def test_tukar_latar_lesi_utuh_latar_berganti():
    x, m = _batch()
    out = tukar_latar(x, m, torch.Generator().manual_seed(0), p=1.0, dilasi_px=2, tepi_px=1)
    torch.testing.assert_close(out[..., 30, 30], x[..., 30, 30])           # pusat lesi tetap
    assert not torch.allclose(out[..., 2, 2], x[..., 2, 2])                 # latar diganti
    assert torch.equal(tukar_latar(x, m, torch.Generator().manual_seed(0), p=0.0), x)
