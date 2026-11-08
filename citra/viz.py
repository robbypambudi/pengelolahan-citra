"""Visualisasi cepat untuk laporan."""

import matplotlib

matplotlib.use("Agg")  # aman tanpa display (WSL/headless)
import matplotlib.pyplot as plt
import numpy as np


def tampil_grid(citra, judul=None, kolom=4, simpan_ke=None, ukuran=3.0):
    """Susun beberapa citra dalam grid; simpan ke file bila simpan_ke diberikan."""
    n = len(citra)
    baris = int(np.ceil(n / kolom))
    kolom = min(kolom, n)
    fig, axes = plt.subplots(baris, kolom, figsize=(ukuran * kolom, ukuran * baris), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            img = citra[i]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
            if judul:
                ax.set_title(judul[i], fontsize=10)
    fig.tight_layout()
    if simpan_ke:
        fig.savefig(simpan_ke, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def histogram(img, simpan_ke=None):
    """Histogram intensitas (grayscale) atau per kanal RGB."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if img.ndim == 2:
        ax.hist(img.ravel(), bins=256, range=(0, 256), color="#233D80")
    else:
        for k, warna in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
            ax.hist(img[..., k].ravel(), bins=256, range=(0, 256), color=warna, alpha=0.5)
    ax.set_xlabel("Intensitas")
    ax.set_ylabel("Jumlah piksel")
    fig.tight_layout()
    if simpan_ke:
        fig.savefig(simpan_ke, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
