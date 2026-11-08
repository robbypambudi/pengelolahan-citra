import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from citra.imbalance import (bobot_kelas, cari_bias_bacc, indeks_sampler_seimbang, koreksi_prior,
                             terapkan_bias)

torch = pytest.importorskip("torch")
from citra.imbalance import focal_loss, logit_adjusted_loss  # noqa: E402

Y = np.array([0] * 900 + [1] * 90 + [2] * 10)


@pytest.mark.parametrize("jenis", ["penuh", "akar", "efektif", "tanpa"])
def test_bobot_ternormalisasi_dan_monoton(jenis):
    w = bobot_kelas(Y, 3, jenis)
    n = np.bincount(Y)
    assert np.isclose((w * n).sum() / n.sum(), 1.0)
    if jenis != "tanpa":
        assert w[0] < w[1] < w[2]


def test_bobot_penuh_sama_dengan_rumus_eksperimen_01():
    n = np.bincount(Y)
    np.testing.assert_allclose(bobot_kelas(Y, 3, "penuh"), len(Y) / (3 * n))


def test_akar_lebih_lunak_dari_penuh():
    p, a = bobot_kelas(Y, 3, "penuh"), bobot_kelas(Y, 3, "akar")
    assert a[2] / a[0] < p[2] / p[0]


def test_focal_gamma0_sama_dengan_cross_entropy():
    torch.manual_seed(0)
    logit, y = torch.randn(16, 3), torch.randint(0, 3, (16,))
    torch.testing.assert_close(focal_loss(logit, y, gamma=0.0), torch.nn.functional.cross_entropy(logit, y))


def test_logit_adjusted_tau0_sama_dengan_cross_entropy():
    torch.manual_seed(0)
    logit, y = torch.randn(16, 3), torch.randint(0, 3, (16,))
    lp = torch.log(torch.tensor([0.9, 0.09, 0.01]))
    torch.testing.assert_close(logit_adjusted_loss(logit, y, lp, tau=0.0), torch.nn.functional.cross_entropy(logit, y))


def test_sampler_seimbang():
    gen = torch.Generator().manual_seed(0)
    idx = indeks_sampler_seimbang(np.arange(len(Y)), Y, 3, gen)
    frek = np.bincount(Y[idx], minlength=3) / len(idx)
    assert len(idx) == len(Y) and np.all(np.abs(frek - 1 / 3) < 0.05)


def test_koreksi_prior_menaikkan_kelas_minoritas():
    prob = np.array([[0.6, 0.3, 0.1]])
    q = koreksi_prior(prob, [0.9, 0.09, 0.01])
    assert q.argmax() == 2 and np.isclose(q.sum(), 1)


def test_bias_bacc_tidak_menurunkan_bacc_validasi():
    rng = np.random.default_rng(0)
    y = Y.copy()
    prob = rng.dirichlet(np.ones(3), len(y)) * 0.5
    prob[np.arange(len(y)), y] += 0.3
    prob[:, 0] += 0.4                            # bias ke kelas mayoritas
    prob /= prob.sum(1, keepdims=True)
    awal = balanced_accuracy_score(y, prob.argmax(1))
    b, skor = cari_bias_bacc(prob, y, 3)
    assert skor >= awal
    assert np.isclose(balanced_accuracy_score(y, terapkan_bias(prob, b).argmax(1)), skor)
