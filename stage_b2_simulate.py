"""
STAGE B, step B3 - Shake every fake brain and save what the scanner would see
============================================================================
For each Stage A brain (variant A):
  1. split each 1 mm pixel into 2x2 (0.5 mm grid, chosen by the C1 checks)
  2. shake the skull with a random smooth drive, solve the wave equation
  3. average to 3 mm voxels (like the scanner), keep the true stiffness per voxel
  4. save the CLEAN (noise-free) 3 mm data

Noise is NOT added here. It is added later, when patches are made for training,
so the same brain can be reused at many noise levels (SNR 5-50) without
re-simulating, and the test set can be evaluated at fixed SNRs.

Usage:   python stage_b2_simulate.py          (all brains)
         python stage_b2_simulate.py 20       (first 20 only, quick test)
Safe to stop and restart: brains already done are skipped.

Output:  mre_project/stage_b/variant_A/sim_00000.npz ...
         mre_project/stage_b/catalog_b.csv
         mre_project/stage_b/qc_waves.png
"""
import csv, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from mre_solver import simulate_brain

VARIANT = "A"
REFINE = 2                         # 0.5 mm solve grid (C1 check 2)
SEED = 2026
N_WORKERS = max(1, (os.cpu_count() or 2) // 2)   # parallel brains; lower this if RAM is tight

BASE = Path(__file__).resolve().parent
IN_DIR = BASE / "mre_project" / "stage_a" / f"variant_{VARIANT}"
OUT_DIR = BASE / "mre_project" / "stage_b" / f"variant_{VARIANT}"
CATALOG_A = BASE / "mre_project" / "stage_a" / "catalog.csv"


def load_splits():
    """sim_id -> train / val / test, from the Stage A catalog."""
    with open(CATALOG_A, newline="") as f:
        return {int(r["sim_id"]): r["split"] for r in csv.DictReader(f)
                if r["variant"] == VARIANT}


def run_one(job):
    sim_id, split = job
    out = OUT_DIR / f"sim_{sim_id:05d}.npz"
    if out.exists():                                  # resume support
        d = np.load(out)
        return dict(sim_id=sim_id, split=split, skipped=True,
                    seconds=0.0, residual=float(d["residual"]),
                    amp_median=float(np.median(np.abs(d["u3"][d["mask3"]]))),
                    amp_p5=float(np.percentile(np.abs(d["u3"][d["mask3"]]), 5)),
                    n_brain_voxels=int(d["mask3"].sum()))
    t0 = time.time()
    a = np.load(IN_DIR / f"sim_{sim_id:05d}.npz")
    rng = np.random.default_rng([SEED, sim_id])       # reproducible drive per brain
    s = simulate_brain(a["tissue"], a["Gp"].astype(float), a["xi"].astype(float),
                       rng, refine=REFINE)
    save = dict(u3=s["u3"].astype(np.complex64),      # wave at 3 mm (clean)
                G3=s["G3"].astype(np.float32),        # true G' per voxel (Pa)
                xi3=s["xi3"].astype(np.float32),      # true damping per voxel
                mask3=s["mask3"],                     # voxel is >50 % brain
                brainfrac3=s["brainfrac3"].astype(np.float32),
                csffrac3=s["csffrac3"].astype(np.float32),
                residual=s["residual"])
    if split == "test":                               # keep 1 mm fields for figures
        save.update(u_1mm=s["u_1mm"].astype(np.complex64), tissue=a["tissue"],
                    Gp=a["Gp"], xi=a["xi"], inclusions=a["inclusions"])
    np.savez_compressed(out, **save)
    amp = np.abs(s["u3"][s["mask3"]])
    return dict(sim_id=sim_id, split=split, skipped=False, seconds=time.time() - t0,
                residual=s["residual"], amp_median=float(np.median(amp)),
                amp_p5=float(np.percentile(amp, 5)), n_brain_voxels=int(s["mask3"].sum()))


def qc_figure(rows):
    import matplotlib.pyplot as plt
    test = ([r for r in rows if r["split"] == "test"] or rows)[:6]   # test brains if any
    fig, ax = plt.subplots(3, len(test), figsize=(3.2 * len(test), 9.5), squeeze=False)
    for j, r in enumerate(test):
        d = np.load(OUT_DIR / f"sim_{r['sim_id']:05d}.npz")
        m = d["mask3"]
        lim = np.percentile(np.abs(d["u3"][m]), 98)
        ax[0, j].imshow(np.where(m, d["G3"] / 1000, np.nan), cmap="magma", vmin=0, vmax=5,
                        interpolation="nearest")
        ax[0, j].set_title(f"#{r['sim_id']} {r['split']}  true G' (kPa)", fontsize=9)
        ax[1, j].imshow(np.where(m, d["u3"].real, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim,
                        interpolation="nearest")
        ax[1, j].set_title("wave, real part (3 mm)", fontsize=9)
        ax[2, j].imshow(np.where(m, np.abs(d["u3"]), np.nan), cmap="viridis",
                        norm=plt.matplotlib.colors.LogNorm(vmin=1e-2, vmax=1.5),
                        interpolation="nearest")
        ax[2, j].set_title(f"amplitude (median {r['amp_median']:.2f})", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("Stage B quality check: true stiffness, wave, amplitude (log, 0.01-1.5)")
    fig.savefig(OUT_DIR.parent / "qc_waves.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":                 # needed on Windows for multiprocessing
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    jobs = sorted(splits.items())
    if len(sys.argv) > 1:
        jobs = jobs[:int(sys.argv[1])]
    print(f"Simulating {len(jobs)} brains on {N_WORKERS} worker(s), 0.5 mm grid...", flush=True)
    t0, rows = time.time(), []
    with Pool(N_WORKERS) as pool:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(r)
            if k % 10 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  {k}/{len(jobs)} done | {el/60:5.1f} min elapsed | "
                      f"~{el/k*(len(jobs)-k)/60:5.1f} min left", flush=True)
    rows.sort(key=lambda r: r["sim_id"])
    with open(OUT_DIR.parent / "catalog_b.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    res = np.array([r["residual"] for r in rows])
    med = np.array([r["amp_median"] for r in rows])
    p5 = np.array([r["amp_p5"] for r in rows])
    nv = np.array([r["n_brain_voxels"] for r in rows])
    print("\nSummary")
    print(f"  max solver residual        {res.max():.1e}   (should be < 1e-10)")
    print(f"  median amplitude in brain  {np.median(med):.3f}  (range {med.min():.3f}-{med.max():.3f})")
    print(f"  weakest 5 % of voxels      {np.median(p5):.4f} typical, {p5.min():.4f} worst brain")
    print(f"  brain voxels per slice     {int(np.median(nv))} typical ({nv.min()}-{nv.max()})")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {sum(r['split'] == s for r in rows)} brains")
    low = [r["sim_id"] for r in rows if r["amp_p5"] < 0.01]
    if low:
        print(f"  WARNING: {len(low)} brains have very weak centres (5th pct < 0.01): {low[:10]}")
    qc_figure(rows)
    print(f"\nSaved to {OUT_DIR.parent}  ({(time.time()-t0)/60:.1f} min)")