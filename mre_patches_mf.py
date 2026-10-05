"""
v3: multi-frequency patches.
A patch = the same 7x7 voxels seen at 40, 60 and 80 Hz:
  [re40, im40, re60, im60, re80, im80, mask]  -> 6 x 49 + 49 = 343 numbers
Each frequency gets its own random phase and its own scaling (peak = 1), so the
strong 40 Hz wave cannot drown out the weaker 80 Hz one.
"""
from pathlib import Path
import numpy as np
import mre_patches as P

FREQS = [40, 60, 80]
N_IN_MF = 2 * len(FREQS) * P.W * P.W + P.W * P.W     # 343


def freq_dir(base, f):
    """60 Hz lives in the original folder; 40 / 80 Hz in the v3 folders."""
    b = Path(base) / "mre_project" / "stage_b"
    return b / "variant_A" if f == 60 else b / f"variant_A_f{f}"


def load_brain(base, sim_id):
    """Clean waves at all frequencies + truth (truth and mask are the same at every frequency)."""
    d = {f: np.load(freq_dir(base, f) / f"sim_{sim_id:05d}.npz") for f in FREQS}
    d60 = d[60]
    return dict(u={f: d[f]["u3"].astype(complex) for f in FREQS}, mask=d60["mask3"],
                G3=d60["G3"].astype(float), xi3=d60["xi3"].astype(float),
                brainfrac3=d60["brainfrac3"])


def extract_mf(us, mask, cen, phases):
    """us: dict freq -> (noisy) wave; phases: dict freq -> array of per-patch phases."""
    parts = []
    for f in FREQS:
        X = P.extract(us[f], mask, cen, phase=phases[f])     # (N, 147): re, im, mask; scaled per frequency
        parts.append(X[:, :2 * P.W * P.W])
    parts.append(X[:, 2 * P.W * P.W:])                         # mask channel once
    return np.concatenate(parts, axis=1).astype(np.float32)
