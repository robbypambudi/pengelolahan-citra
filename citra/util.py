"""Reproducibility: seed, folder run, dan pencatatan metrik."""

import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


def seed_semua(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def folder_run(folder_eksperimen, config_path=None):
    """Buat hasil/<YYYYmmdd-HHMMSS>/ dan salin config ke dalamnya."""
    run = Path(folder_eksperimen) / "hasil" / datetime.now().strftime("%Y%m%d-%H%M%S")
    (run / "contoh").mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy(config_path, run / "config.yaml")
    return run


def simpan_metrik(run, metrik):
    path = Path(run) / "metrics.json"
    path.write_text(json.dumps(metrik, indent=2, ensure_ascii=False))
    return path
