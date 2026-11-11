"""Eksperimen 04 — MobileNetV3-Large + CBAM / Coordinate Attention.

Memakai ulang alur eksperimen 03 (cache, split per lesi, S6-logit-adj, keputusan argmax dan
+bias_val, metrik, statistik). Hanya pembuat model yang berbeda: attention dipasang lewat
citra.attention.pasang_attention (identitas saat awal, SE pralatih dipertahankan).

Jalankan dari root proyek:
  uv run python tugas/pcd-lesi-kulit/eksperimen/04-attention/run.py [--config config-cepat.yaml] [--lanjut hasil/<waktu>]
"""

import importlib.util
from pathlib import Path

from citra.attention import pasang_attention

FOLDER = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("eks03", FOLDER.parent / "03-derm-v2" / "run.py")
eks03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eks03)


def buat_model(v, cm):
    model = eks03.buat_model(v, cm)
    if v.get("attention"):
        model = pasang_attention(model, v["attention"], v["letak"])
    return model


if __name__ == "__main__":
    eks03.main(folder=FOLDER, pembuat=buat_model)
