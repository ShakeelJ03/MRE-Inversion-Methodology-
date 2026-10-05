"""
v3, STEP 2 - Multi-frequency patches + train the multi-frequency network (ILI-MF)
================================================================================
Same recipe as v1 (stage_c1 + stage_c2), only the input changes:
  v1: 7x7 patch at 60 Hz                 -> 147 numbers
  v3: same 7x7 patch at 40, 60 and 80 Hz -> 343 numbers
Same brains, same split, same noise model (one SNR per brain copy, applied to every
frequency), same network size, same early stopping, ensemble of 5.
So any difference from v1 comes from the extra frequencies.

Usage:
  python stage_c3_patches_train_mf.py patches    make the patches (a few minutes)
  python stage_c3_patches_train_mf.py quick      speed test (1 net, 50k patches, 3 epochs)
  python stage_c3_patches_train_mf.py train      train 5 networks (restartable)
Output: mre_project/stage_c_mf/patches_{train,val}.npz, models/ILI_MF_seed*.pkl,
        training_history.json
"""
import csv, json, pickle, sys, time
from pathlib import Path
import numpy as np
import mre_patches as P
import mre_patches_mf as MF
from stage_c2_train import train_one

NOISE_COPIES = 2
SNR_RANGE = (5.0, 50.0)
MAX_PATCHES = {"train": 500_000, "val": 80_000}
N_SEEDS = 5
MAX_EPOCHS = 40
SEED = 2026

BASE = Path(__file__).resolve().parent
OUT = BASE / "mre_project" / "stage_c_mf"
M_DIR = OUT / "models"
M_DIR.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


def split_ids():
    with open(BASE / "mre_project" / "stage_b" / "catalog_b.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {s: sorted(int(r["sim_id"]) for r in rows if r["split"] == s) for s in ("train", "val")}


def build(split, ids, rng):
    per_copy = int(np.ceil(MAX_PATCHES[split] / (len(ids) * NOISE_COPIES)))
    X, Y = [], []
    for k, i in enumerate(ids):
        b = MF.load_brain(BASE, i)
        cen_all = P.centres(b["mask"])
        for _ in range(NOISE_COPIES):
            snr = float(np.exp(rng.uniform(*np.log(SNR_RANGE))))
            un = {f: P.add_noise(b["u"][f], b["mask"], snr, rng) for f in MF.FREQS}
            cen = cen_all[rng.permutation(len(cen_all))[:per_copy]]
            ph = {f: rng.uniform(0, 2 * np.pi, len(cen)) for f in MF.FREQS}
            X.append(MF.extract_mf(un, b["mask"], cen, ph))
            Y.append(P.targets(b["G3"], b["xi3"], cen))
        if (k + 1) % 50 == 0 or k + 1 == len(ids):
            log(f"  {split}: {k + 1}/{len(ids)} brains")
    return np.concatenate(X), np.concatenate(Y)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "patches"
    if mode == "patches":
        rng = np.random.default_rng(SEED)
        ids = split_ids()
        for split in ("train", "val"):
            X, Y = build(split, ids[split], rng)
            np.savez(OUT / f"patches_{split}.npz", X=X.astype(np.float16), Y=Y)
            log(f"saved {split}: {X.shape[0]} patches x {X.shape[1]} inputs")
        sys.exit()

    tr, va = np.load(OUT / "patches_train.npz"), np.load(OUT / "patches_val.npz")
    Xtr, Ytr = tr["X"].astype(np.float32), tr["Y"]
    Xva, Yva = va["X"].astype(np.float32), va["Y"]
    log(f"train {Xtr.shape}, val {Xva.shape}")
    if mode == "quick":
        train_one("ILI_MF", 0, Xtr[:50_000], Ytr[:50_000], Xva[:10_000], Yva[:10_000],
                  Xva[:10_000], Yva[:10_000], 3)
        per_epoch = (time.time() - START) / 3 * len(Xtr) / 50_000
        log(f"estimate ~{per_epoch/60:.1f} min per full epoch, ~{per_epoch*22*N_SEEDS/3600:.1f} h for 5 nets")
        sys.exit()

    hist_path = OUT / "training_history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    for seed in range(N_SEEDS):
        out = M_DIR / f"ILI_MF_seed{seed}.pkl"
        if out.exists():
            log(f"seed {seed} already trained, skipping"); continue
        net, h, extra = train_one("ILI_MF", seed, Xtr, Ytr, Xva, Yva, Xva, Yva, MAX_EPOCHS)
        pickle.dump(net, open(out, "wb"))
        hist[f"ILI_MF_seed{seed}"] = dict(epochs=h, **extra)
        hist_path.write_text(json.dumps(hist, indent=1))
    print("\nSummary: median G' error on validation patches (v1 single-frequency ILI was 7.0-7.2 %)")
    for k, h in hist.items():
        print(f"  {k}: {h['realistic_val_median_err_pct']:.1f} %  ({len(h['epochs'])} epochs)")
    log("Done")
