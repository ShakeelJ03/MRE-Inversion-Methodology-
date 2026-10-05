"""
STAGE C, step 1 - Turn the simulated brains into training examples (patches)
===========================================================================
For every TRAIN and VAL brain from Stage B:
  1. add scanner noise at a random SNR between 5 and 50 (log-uniform),
     NOISE_COPIES times per brain with different noise -> free extra data
  2. cut a 7x7 patch around every brain voxel (real, imag, mask = 147 numbers)
  3. random global phase per patch (the scanner's phase is arbitrary)
  4. answer = log(true G') and damping at the centre voxel

Two training sets are made:
  ILI : patches from the realistic brains (stiffness varies inside a patch)  -> main model
  HLI : matched UNIFORM patches (Murphy 2018)  -> comparison model
        same stiffness/damping distribution, same masks, same noise levels

Test brains are NOT turned into patches here: they are evaluated whole, at
fixed SNRs, in the evaluation step.

Output: mre_project/stage_c/patches_train.npz, patches_val.npz, qc_patches.png
"""
import csv, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mre_patches as P

VARIANT = "A"
NOISE_COPIES = 2
SNR_RANGE = (5.0, 50.0)
MAX_PATCHES = {"train": 500_000, "val": 80_000}   # caps keep RAM and training time sane
SEED = 2026

BASE = Path(__file__).resolve().parent
B_DIR = BASE / "mre_project" / "stage_b" / f"variant_{VARIANT}"
OUT = BASE / "mre_project" / "stage_c"
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({time.time() - START:.0f} s)", flush=True)


def load_split_ids():
    with open(BASE / "mre_project" / "stage_b" / "catalog_b.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {s: sorted(int(r["sim_id"]) for r in rows if r["split"] == s)
            for s in ("train", "val", "test")}


def build(split, ids, rng):
    """Realistic (ILI) patches for one split, subsampled per brain to the cap."""
    per_copy = int(np.ceil(MAX_PATCHES[split] / (len(ids) * NOISE_COPIES)))
    X, Y, SIM, SNR = [], [], [], []
    for k, i in enumerate(ids):
        d = np.load(B_DIR / f"sim_{i:05d}.npz")
        u3, mask3 = d["u3"].astype(complex), d["mask3"]
        cen_all = P.centres(mask3)
        for _ in range(NOISE_COPIES):
            snr = float(np.exp(rng.uniform(*np.log(SNR_RANGE))))
            un = P.add_noise(u3, mask3, snr, rng)
            cen = cen_all[rng.permutation(len(cen_all))[:per_copy]]
            X.append(P.extract(un, mask3, cen, phase=rng.uniform(0, 2 * np.pi, len(cen))))
            Y.append(P.targets(d["G3"], d["xi3"], cen))
            SIM.append(np.full(len(cen), i, np.int32))
            SNR.append(np.full(len(cen), snr, np.float32))
        if (k + 1) % 50 == 0 or k + 1 == len(ids):
            log(f"  {split}: {k + 1}/{len(ids)} brains")
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(SIM), np.concatenate(SNR))


def build_uniform(Y_real, X_real, snr_real, rng, chunk=20_000):
    """HLI patches matched one-to-one in distribution to the realistic set."""
    n = len(Y_real)
    G, xi = P.decode(Y_real[rng.permutation(n)])                  # same G', xi distribution
    masks = X_real[rng.permutation(n), 2 * P.W * P.W:]            # same masks
    snr = snr_real[rng.permutation(n)]                            # same noise levels
    Xh = np.empty((n, P.N_IN), np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Xh[s:e] = P.uniform_patches(G[s:e], xi[s:e], masks[s:e], snr[s:e], rng)
    Yh = np.stack([np.log(G / 1000), 5 * xi], axis=1).astype(np.float32)
    return Xh, Yh


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    ids = load_split_ids()
    log(f"Brains: train {len(ids['train'])}, val {len(ids['val'])}, test {len(ids['test'])} (test untouched)")

    for split in ("train", "val"):
        log(f"Making realistic (ILI) {split} patches...")
        X, Y, SIM, SNR = build(split, ids[split], rng)
        log(f"Making matched uniform (HLI) {split} patches...")
        Xh, Yh = build_uniform(Y, X, SNR, rng)
        np.savez(OUT / f"patches_{split}.npz",
                 X_ili=X.astype(np.float16), Y_ili=Y, sim_ili=SIM, snr_ili=SNR,
                 X_hli=Xh.astype(np.float16), Y_hli=Yh)
        G, xi = P.decode(Y)
        log(f"  saved {split}: {len(X)} patches | G' {G.min()/1000:.2f}-{G.max()/1000:.2f} kPa "
            f"(median {np.median(G)/1000:.2f}) | xi median {np.median(xi):.3f} | "
            f"SNR {SNR.min():.1f}-{SNR.max():.1f} | mean masked fraction "
            f"{1 - X[:, 2*P.W*P.W:].mean():.2f}")
        if split == "train":
            X_show, Y_show, Xh_show, Yh_show = X[:2000], Y[:2000], Xh[:2000], Yh[:2000]

    # QC figure: what the network actually sees
    order = np.argsort(Y_show[:, 0])
    pick = order[np.linspace(0, len(order) - 1, 6).astype(int)]   # soft -> stiff
    fig, ax = plt.subplots(3, 6, figsize=(13, 7))
    for j, i in enumerate(pick):
        re = X_show[i, :49].reshape(7, 7)
        m = X_show[i, 98:].reshape(7, 7)
        ax[0, j].imshow(np.where(m > 0, re, np.nan), cmap="RdBu_r", vmin=-1, vmax=1)
        ax[0, j].set_title(f"realistic\nG' {P.decode(Y_show[i])[0]/1000:.2f} kPa", fontsize=9)
        ax[1, j].imshow(m, cmap="gray", vmin=0, vmax=1)
        ax[1, j].set_title("mask (black = CSF/outside)", fontsize=8)
        k = np.argmin(np.abs(Yh_show[:, 0] - Y_show[i, 0]))       # uniform patch, similar G'
        ax[2, j].imshow(np.where(Xh_show[k, 98:].reshape(7, 7) > 0,
                                 Xh_show[k, :49].reshape(7, 7), np.nan),
                        cmap="RdBu_r", vmin=-1, vmax=1)
        ax[2, j].set_title(f"uniform (HLI)\nG' {P.decode(Yh_show[k])[0]/1000:.2f} kPa", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("What the network sees: 7x7 patches (real part), soft (left) to stiff (right)")
    fig.tight_layout()
    fig.savefig(OUT / "qc_patches.png", dpi=130)
    log(f"Done. Saved in {OUT}")
