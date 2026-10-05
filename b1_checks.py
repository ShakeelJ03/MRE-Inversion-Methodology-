"""
STAGE B, step B2 - Checkpoint C1: does the simulator obey the physics?
=====================================================================
Run this BEFORE simulating the whole database.

Check 1  Plane wave in a uniform strip: simulated wavelength vs theory
         (with and without damping), on 1, 0.5 and 0.25 mm grids.
Check 2  Grid convergence on a real Stage A brain: 1 mm and 0.5 mm vs a
         0.25 mm reference.  Decides which grid Stage B uses.
Check 3  Direct inversion (DI) on a uniform medium: exact on the fine grid,
         biased on the 3 mm acquisition grid.
Check 4  Every solve has a tiny residual (the linear solver did its job).

Output: mre_project/stage_b/C1_checks.png, C1_brain_example.png, C1_results.json
"""
import json, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mre_solver import (solve_helmholtz, wavenumber, complex_modulus, simulate_brain,
                        RHO, OMEGA, FREQ)

BASE = Path(__file__).resolve().parent
STAGE_A = BASE / "mre_project" / "stage_a" / "variant_A"
OUT = BASE / "mre_project" / "stage_b"
OUT.mkdir(parents=True, exist_ok=True)
results = {}


# ---------------------------------------------------------------------------
# Check 1: plane wave in a uniform strip
# ---------------------------------------------------------------------------
def plane_wave_strip(Gp, xi, h_mm, length_mm=150.0, sponge_mm=150.0, width_mm=6.0):
    """Strip shaken at x = 0. The far end is a 'sponge' (damping rises smoothly)
    so the wave dies out without bouncing back. Returns x (m), u along the middle."""
    nx = int(round((length_mm + sponge_mm) / h_mm)) + 1
    ny = int(round(width_mm / h_mm))
    x = np.arange(nx) * h_mm * 1e-3
    s = np.clip((x * 1e3 - length_mm) / sponge_mm, 0, 1)
    Gc = np.tile(complex_modulus(Gp, xi + 0.8 * s ** 3), (ny, 1))
    u_bc = np.zeros((ny, nx), complex)
    u_bc[:, 0] = 1.0                                   # the shaker
    domain = np.ones((ny, nx), bool)
    domain[:, 0] = domain[:, -1] = False
    u, res = solve_helmholtz(Gc, domain, u_bc, h_mm, neumann_edges=True)
    u[:, 0] = 1.0
    keep = x * 1e3 <= length_mm
    return x[keep], u[ny // 2][keep], res


def measured_wavelength(x, u):
    """Wavelength from how fast the phase turns, between 20 and 130 mm."""
    sel = (x > 0.02) & (x < 0.13)
    slope = np.polyfit(x[sel], np.unwrap(np.angle(u[sel])), 1)[0]
    return 2 * np.pi / abs(slope)


rows = []
print("Check 1: wavelength vs theory")
print("  G'(kPa)  xi   grid(mm)  pts/wavelength  theory(mm)  simulated(mm)  error(%)  no-damping formula error(%)")
for Gp in (1000.0, 2500.0, 5000.0):
    for xi in (0.0, 0.2):
        lam_true = 2 * np.pi / wavenumber(Gp, xi).real
        lam_naive = np.sqrt(Gp / RHO) / FREQ           # ignores damping
        for h in (1.0, 0.5, 0.25):
            x, u, res = plane_wave_strip(Gp, xi, h)
            lam = measured_wavelength(x, u)
            r = dict(G_kPa=Gp / 1000, xi=xi, h_mm=h, pts_per_wavelength=lam_true * 1e3 / h,
                     theory_mm=lam_true * 1e3, simulated_mm=lam * 1e3,
                     error_pct=100 * (lam - lam_true) / lam_true,
                     naive_formula_error_pct=100 * (lam_naive - lam_true) / lam_true,
                     residual=res)
            rows.append(r)
            print(f"  {Gp/1000:6.1f}  {xi:4.1f}  {h:6.2f}  {r['pts_per_wavelength']:12.1f}  "
                  f"{r['theory_mm']:10.2f}  {r['simulated_mm']:12.2f}  {r['error_pct']:+8.3f}  "
                  f"{r['naive_formula_error_pct']:+10.1f}")
results["check1_wavelength"] = rows

# ---------------------------------------------------------------------------
# Check 2: grid convergence on a real brain
# ---------------------------------------------------------------------------
print("\nCheck 2: grid convergence on brain sim_00000 (this takes ~1 min)")
d = np.load(STAGE_A / "sim_00000.npz")
tissue, Gp, xi = d["tissue"], d["Gp"].astype(float), d["xi"].astype(float)
sims = {}
for refine in (1, 2, 4):
    t0 = time.time()
    sims[refine] = simulate_brain(tissue, Gp, xi, np.random.default_rng(7), refine=refine)
    print(f"  grid {1/refine:.2f} mm: {time.time()-t0:5.1f} s, residual {sims[refine]['residual']:.1e}")
m = sims[1]["mask3"]
ref = sims[4]["u3"][m]
conv = {f"{1/r:g} mm": float(100 * np.linalg.norm(sims[r]["u3"][m] - ref) / np.linalg.norm(ref))
        for r in (1, 2)}
for k, v in conv.items():
    print(f"  {k} grid differs from the 0.25 mm reference by {v:.2f} % (at 3 mm)")
results["check2_brain_convergence_pct"] = conv

# ---------------------------------------------------------------------------
# Check 3: direct inversion, fine grid vs 3 mm
# ---------------------------------------------------------------------------
def di_1d(u, h):
    """DI along a line: G* = -rho w^2 u / (second derivative of u)."""
    lap = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2
    return (-RHO * OMEGA ** 2 * u[1:-1] / lap).real


di_rows = []
print("\nCheck 3: direct inversion on a uniform medium (no noise)")
for Gk in np.arange(1.0, 5.01, 0.5):
    x, u, _ = plane_wave_strip(Gk * 1000, 0.0, 1.0)
    g_fine = di_1d(u, 1e-3)
    n3 = (len(u) // 3) * 3
    g_3mm = di_1d(u[:n3].reshape(-1, 3).mean(axis=1), 3e-3)   # average to 3 mm first
    mid = lambda a: np.median(a[len(a) // 4: 3 * len(a) // 4])
    di_rows.append(dict(G_kPa=float(Gk), fine_error_pct=float(100 * (mid(g_fine) / (Gk * 1000) - 1)),
                        mm3_error_pct=float(100 * (mid(g_3mm) / (Gk * 1000) - 1))))
    print(f"  G' = {Gk:.1f} kPa: DI error on 1 mm grid {di_rows[-1]['fine_error_pct']:+.4f} %,"
          f" on 3 mm grid {di_rows[-1]['mm3_error_pct']:+.2f} %")
results["check3_direct_inversion"] = di_rows
(OUT / "C1_results.json").write_text(json.dumps(results, indent=1))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
x, u, _ = plane_wave_strip(2500.0, 0.2, 0.5)
k = wavenumber(2500.0, 0.2)
ax[0].plot(x * 1e3, np.real(np.exp(-1j * k * x)), color="0.6", lw=4, label="theory")
ax[0].plot(x * 1e3, u.real, "--", color="C0", lw=1.5, label="simulator")
ax[0].set(xlabel="distance (mm)", ylabel="displacement (a.u.)",
          title="A  damped plane wave, 2.5 kPa, xi = 0.2")
ax[0].legend(frameon=False)
for G, c in ((1.0, "C0"), (2.5, "C2"), (5.0, "C3")):
    rr = [r for r in rows if r["G_kPa"] == G and r["xi"] == 0.2]
    ax[1].loglog([r["pts_per_wavelength"] for r in rr], [abs(r["error_pct"]) for r in rr],
                 "o-", color=c, label=f"{G:g} kPa")
ax[1].axhline(2, color="0.5", ls=":")
ax[1].text(16, 1.3, "2 % target", color="0.4")
ax[1].set(xlabel="grid points per wavelength", ylabel="wavelength error (%)",
          title="B  grid convergence (error / 4 per halving)")
ax[1].legend(frameon=False)
Gs = [r["G_kPa"] for r in di_rows]
ax[2].plot(Gs, [r["fine_error_pct"] for r in di_rows], "o-", color="C2", label="1 mm grid")
ax[2].plot(Gs, [r["mm3_error_pct"] for r in di_rows], "o-", color="C3", label="3 mm grid")
ax[2].axhline(0, color="0.5", lw=0.8)
ax[2].set(xlabel="true stiffness (kPa)", ylabel="DI error (%)",
          title="C  DI exact on fine grid, overestimates at 3 mm")
ax[2].legend(frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "C1_checks.png", dpi=150)

s = sims[2]
dom = tissue > 0
fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
ax[0].imshow(np.where(dom, Gp / 1000, np.nan), cmap="magma", vmin=0, vmax=5)
ax[0].set_title("true G' (kPa)")
lim = np.percentile(np.abs(s["u_1mm"][dom]), 98)
ax[1].imshow(np.where(dom, s["u_1mm"].real, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim)
ax[1].set_title("wave, real part (1 mm)")
im = ax[2].imshow(np.where(dom, np.abs(s["u_1mm"]), np.nan), cmap="viridis",
                  norm=plt.matplotlib.colors.LogNorm(vmin=lim / 100, vmax=lim * 1.5))
ax[2].set_title("wave amplitude (log scale)")
fig.colorbar(im, ax=ax[2], fraction=0.04)
ax[3].imshow(np.where(s["mask3"], s["u3"].real, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim,
             interpolation="nearest")
ax[3].set_title("what the scanner sees (3 mm, brain mask)")
for a in ax:
    a.axis("off")
fig.tight_layout()
fig.savefig(OUT / "C1_brain_example.png", dpi=150)
centre = np.abs(s["u3"][s["mask3"]])
print(f"\nAmplitude in brain: median {np.median(centre):.3f}, 5th percentile "
      f"{np.percentile(centre, 5):.4f} (skull drive is ~1)")
print(f"Saved figures and results in {OUT}")