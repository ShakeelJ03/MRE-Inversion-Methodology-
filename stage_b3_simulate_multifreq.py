"""
v3, STEP 1 - Shake the SAME 500 brains at two more frequencies (40 and 80 Hz)
============================================================================
Why: every v2 result says the physics check is starved of information (it gets
better and better as noise drops). One frequency = one wave equation per voxel.
Three frequencies = three independent equations for the SAME stiffness.

60 Hz already exists (stage_b/variant_A). This adds 40 and 80 Hz:
  - same brains, same stiffness, same damping, same skull-shaking pattern
    (same random seed), only the frequency changes
  - stiffness and damping are kept frequency-independent (a simplification;
    real tissue stiffens slightly with frequency -> roadmap: power-law model)

Step 0 inside: a mini C1 check at each new frequency (wavelength vs theory).
If any error is above 2 % the script stops.

Usage:  python stage_b3_simulate_multifreq.py        (safe to stop and restart)
Output: mre_project/stage_b/variant_A_f40/, variant_A_f80/ (same file format as variant_A)
"""
import csv, os, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from mre_solver import simulate_brain, solve_helmholtz, wavenumber, complex_modulus

NEW_FREQS = [40, 80]
SEED = 2026                                  # SAME seed as stage_b2 -> same skull drive pattern
N_WORKERS = max(1, (os.cpu_count() or 2) // 2)
BASE = Path(__file__).resolve().parent
IN_DIR = BASE / "mre_project" / "stage_a" / "variant_A"
B_DIR = BASE / "mre_project" / "stage_b"


def plane_wave_error(Gp, xi, freq, h_mm=0.5):
    """Mini C1: simulated vs theoretical wavelength in a uniform strip with an absorbing end."""
    L, S, W = 150.0, 150.0, 6.0
    nx, ny = int(round((L + S) / h_mm)) + 1, int(round(W / h_mm))
    x = np.arange(nx) * h_mm * 1e-3
    s = np.clip((x * 1e3 - L) / S, 0, 1)
    Gc = np.tile(complex_modulus(Gp, xi + 0.8 * s ** 3), (ny, 1))
    u_bc = np.zeros((ny, nx), complex); u_bc[:, 0] = 1
    dom = np.ones((ny, nx), bool); dom[:, 0] = dom[:, -1] = False
    u, _ = solve_helmholtz(Gc, dom, u_bc, h_mm, neumann_edges=True, freq=freq)
    line = u[ny // 2]
    sel = (x > 0.02) & (x < 0.13)
    k_num = abs(np.polyfit(x[sel], np.unwrap(np.angle(line[sel])), 1)[0])
    return 100 * (k_num / wavenumber(Gp, xi, freq).real - 1) * -1     # wavelength error %


def run_one(job):
    sim_id, freq = job
    out = B_DIR / f"variant_A_f{freq}" / f"sim_{sim_id:05d}.npz"
    if out.exists():
        d = np.load(out)
        amp = np.abs(d["u3"][d["mask3"]])
        return dict(sim_id=sim_id, freq=freq, residual=float(d["residual"]),
                    amp_median=float(np.median(amp)), amp_p5=float(np.percentile(amp, 5)))
    a = np.load(IN_DIR / f"sim_{sim_id:05d}.npz")
    rng = np.random.default_rng([SEED, sim_id])      # identical drive to the 60 Hz run
    s = simulate_brain(a["tissue"], a["Gp"].astype(float), a["xi"].astype(float), rng,
                       refine=2, freq=freq)
    np.savez_compressed(out, u3=s["u3"].astype(np.complex64), G3=s["G3"].astype(np.float32),
                        xi3=s["xi3"].astype(np.float32), mask3=s["mask3"],
                        brainfrac3=s["brainfrac3"].astype(np.float32),
                        csffrac3=s["csffrac3"].astype(np.float32), residual=s["residual"])
    amp = np.abs(s["u3"][s["mask3"]])
    return dict(sim_id=sim_id, freq=freq, residual=s["residual"],
                amp_median=float(np.median(amp)), amp_p5=float(np.percentile(amp, 5)))


if __name__ == "__main__":
    print("Step 0: mini C1 check (wavelength error %, 0.5 mm grid, xi = 0.2)")
    worst = 0
    for f in NEW_FREQS:
        errs = [plane_wave_error(G, 0.2, f) for G in (1000.0, 2500.0, 5000.0)]
        worst = max(worst, max(abs(e) for e in errs))
        lam = [2 * np.pi / wavenumber(G, 0.2, f).real * 1e3 for G in (1000.0, 2500.0, 5000.0)]
        print(f"  {f} Hz: wavelengths {lam[0]:.1f} / {lam[1]:.1f} / {lam[2]:.1f} mm (1 / 2.5 / 5 kPa), "
              f"errors {errs[0]:+.3f} / {errs[1]:+.3f} / {errs[2]:+.3f} %")
    if worst > 2:
        raise SystemExit("Wavelength error above 2 % - stop and check before simulating.")

    with open(B_DIR / "catalog_b.csv", newline="") as fh:
        ids = sorted(int(r["sim_id"]) for r in csv.DictReader(fh))
    for f in NEW_FREQS:
        (B_DIR / f"variant_A_f{f}").mkdir(parents=True, exist_ok=True)
    jobs = [(i, f) for f in NEW_FREQS for i in ids]
    print(f"\nSimulating {len(ids)} brains x {len(NEW_FREQS)} frequencies on {N_WORKERS} workers...", flush=True)
    t0, rows = time.time(), []
    with Pool(N_WORKERS) as pool:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(r)
            if k % 50 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  {k}/{len(jobs)} | {el/60:4.1f} min | ~{el/k*(len(jobs)-k)/60:4.1f} min left", flush=True)

    print("\nSummary (compare: 60 Hz had median amplitude 0.24, weakest 5 % ~0.06)")
    for f in NEW_FREQS:
        rr = [r for r in rows if r["freq"] == f]
        print(f"  {f} Hz: max residual {max(r['residual'] for r in rr):.1e} | "
              f"median amplitude {np.median([r['amp_median'] for r in rr]):.3f} | "
              f"weakest 5 % typical {np.median([r['amp_p5'] for r in rr]):.4f}, "
              f"worst brain {min(r['amp_p5'] for r in rr):.4f}")
    print("Done.")
