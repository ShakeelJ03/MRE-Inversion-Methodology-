"""
v2, STEP 1 - Run the trained networks on the VALIDATION brains
=============================================================
Why: to improve the trust map honestly we must develop and tune on brains
that are NOT the test set. The 75 validation brains were only used to decide
when training stops, so they are a fair "practice ground".
The test brains stay untouched until the single final evaluation (step 5).

Same networks, same phase cycling, same noise model as stage_d1_evaluate.py.
Output: mre_project/stage_f/predictions_val/sim_XXXXX.npz  (same format as the test predictions)
"""
import csv, time
from pathlib import Path
import numpy as np
import mre_patches as P
from stage_d1_evaluate import load_nets, predict

SNRS = [5, 10, 20, 50]
BASE = Path(__file__).resolve().parent
B_DIR = BASE / "mre_project" / "stage_b" / "variant_A"
OUT = BASE / "mre_project" / "stage_f" / "predictions_val"
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    t0 = time.time()
    nets = load_nets("ILI")
    print(f"Loaded {len(nets)} ILI networks", flush=True)
    with open(BASE / "mre_project" / "stage_b" / "catalog_b.csv", newline="") as f:
        ids = sorted(int(r["sim_id"]) for r in csv.DictReader(f) if r["split"] == "val")
    for n, sid in enumerate(ids):
        d = np.load(B_DIR / f"sim_{sid:05d}.npz")
        u3, mask = d["u3"].astype(complex), d["mask3"]
        save = dict(G3=d["G3"], xi3=d["xi3"], mask3=mask)
        for snr in SNRS:
            rng = np.random.default_rng([sid, snr, 7])        # own noise seeds, never reused for test
            un = P.add_noise(u3, mask, snr, rng)
            lg, xi = predict(nets, un, mask)
            save[f"u_noisy_snr{snr}"] = un.astype(np.complex64)
            save[f"logG_ILI_snr{snr}"] = lg
            save[f"xi_ILI_snr{snr}"] = xi
        np.savez_compressed(OUT / f"sim_{sid:05d}.npz", **save)
        if (n + 1) % 15 == 0 or n + 1 == len(ids):
            print(f"  {n + 1}/{len(ids)} validation brains ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"Done. Saved in {OUT}")
