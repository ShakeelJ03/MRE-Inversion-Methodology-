"""
E6b - Weak-form SINDy: the same system identification as E6, without derivatives of noisy data
=============================================================================================
Why: E6 (strong form) failed below SNR ~50 because it needs the Laplacian (2nd derivative)
of the noisy waves. Weak SINDy (Messenger & Bortz 2021) multiplies the equation by a smooth
test function phi and integrates; integrating by parts TWICE moves every derivative onto phi:

   div(G grad u) + (rho w^2 + c) u = 0
   ->  integral of u * div(G grad phi)  +  (rho w^2 + c) * integral of u * phi  = 0

With G linear in the window, G = G0 + gx x + gy y:
   integral u div(G grad phi) = G0 * S[u lap(phi)] + gx * S[u (x lap(phi) + phi_x)] + gy * S[u (y lap(phi) + phi_y)]
So the only thing done to the noisy data is a weighted SUM (which averages noise away).
phi = (1 - r^2/R^2)^3 is smooth and vanishes with its first derivatives at the edge.

Everything else is IDENTICAL to E6 so the comparison is fair:
same candidate models (M0, M1, M3, M1+3), same 7x7-voxel footprint (9 test functions of
radius 2.5 voxels centred on the 3x3 voxels around each voxel), same thresholds
(heterogeneity >10 % over 6 mm, extra term >5 %, |coef| > 2 SE), same noise correction
(method of moments), same grading (H1, H2, H4), same test brains and noisy data.
Grid bias: phi's derivatives use the grid's own finite differences (so summation by parts
is exact on the grid); the small remaining bias is computed exactly for a plane wave at each
frequency and stiffness and divided out (same spirit as the 3 mm correction in E5/E6).

Usage:  python stage_e6b_weak_sindy.py          (all 75 test brains)
        python stage_e6b_weak_sindy.py 10       (quick check)
Output: mre_project/stage_e6b/results.json, fig_strong_vs_weak.png
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from mre_solver import RHO
import mre_patches_mf as MF
from stage_e6_model_selection import (FREQS, CONDS, H, HALF, R2_MIN, HET_TRUE, MIXED_TRUE, MODELS,
                                      TAU_HET, TAU_EXTRA, Z_MIN, LABEL, N_NOISE)

warnings.filterwarnings("ignore")
R_TEST = 2.0                 # test-function radius (voxels): phi nonzero on the 3x3 core, so the
                             # discrete kernels reach +-2 and the footprint stays 7x7 (same as E6)
OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]   # 9 test functions per window
MIN_TF = 5                   # need at least 5 valid test functions

BASE = Path(__file__).resolve().parent
PRED = BASE / "mre_project" / "stage_g" / "predictions_test"
OUT = BASE / "mre_project" / "stage_e6b"
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


def test_kernels():
    """Discrete-consistent kernels: phi is sampled on the grid and its derivatives are taken
    with the SAME finite differences as the grid (5-point Laplacian, central differences).
    Then summation by parts holds exactly on the grid:  sum(u * L phi) = sum(phi * L u).
    (Analytic derivatives of phi break this on a coarse 3 mm grid: tested, factor up to 9.)"""
    n = int(np.ceil(R_TEST)) + 1
    jy, jx = np.mgrid[-n:n + 1, -n:n + 1].astype(float)
    s = (jx ** 2 + jy ** 2) / R_TEST ** 2
    phi = np.where(s < 1, (1 - s) ** 3, 0.0)
    p = np.pad(phi, 1)
    lap = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * phi) / H ** 2
    dx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * H)
    dy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * H)
    x, y = jx * H, jy * H
    # summation by parts for the first-derivative terms flips the sign: sum(u Dx phi) = -sum(phi Dx u)
    k = dict(phi=phi, lap=lap, dx=-dx, dy=-dy, xlap=x * lap, ylap=y * lap)
    keep = np.any([np.abs(v) > 0 for v in k.values()], axis=0)
    rows, cols = np.any(keep, 1), np.any(keep, 0)
    k = {name: v[rows][:, cols] for name, v in k.items()}
    k["support"] = (np.abs(k["lap"]) + k["phi"] > 0).astype(float)
    return k


K = test_kernels()


def _weak_factor_table():
    """G_estimated / G_true for a plane wave in uniform tissue with these discrete kernels
    (averaged over directions). Exact for uniform media; used to remove the grid bias."""
    from mre_solver import RHO as rho
    n = K["phi"].shape[0] // 2
    jy, jx = np.mgrid[-n:n + 1, -n:n + 1] * H
    Gs = np.linspace(400, 9000, 120)
    tab = {}
    for f in FREQS:
        vals = []
        for G in Gs:
            k = 2 * np.pi * f * np.sqrt(rho / G)
            fac = []
            for th in np.linspace(0, np.pi, 12, endpoint=False):
                e = np.exp(1j * k * (np.cos(th) * jx + np.sin(th) * jy))
                fac.append(k ** 2 / (-(e * K["lap"]).sum() / (e * K["phi"]).sum()).real)
            vals.append(np.mean(fac))
        tab[f] = np.array(vals)
    return Gs, tab


GS_TAB, FAC_TAB = _weak_factor_table()


def weak_bias_factor(G, f):
    return np.interp(np.clip(G, GS_TAB[0], GS_TAB[-1]), GS_TAB, FAC_TAB[f])


def corr(a, k):
    if np.iscomplexobj(a):
        return (ndimage.correlate(a.real, k, mode="constant") + 1j * ndimage.correlate(a.imag, k, mode="constant")) * H ** 2
    return ndimage.correlate(a, k, mode="constant") * H ** 2


def weak_fields(u):
    """All weighted sums of u needed, at every possible test-function centre."""
    return {k: corr(u, K[k]) for k in ("phi", "lap", "dx", "dy", "xlap", "ylap")}


def fit_brain_weak(us, mask, sigmas=None, rng=None):
    noisy = sigmas is not None
    valid_tf = (ndimage.correlate(mask.astype(float), K["support"], mode="constant")
                >= K["support"].sum() - 1e-6)                 # test function fully inside brain
    F = {f: weak_fields(us[f]) for f in FREQS}
    Fn = []
    if noisy:
        for _ in range(N_NOISE):
            Fn.append({f: weak_fields(sigmas[f] * (rng.standard_normal(mask.shape) + 1j * rng.standard_normal(mask.shape))
                                      / np.sqrt(2) * mask) for f in FREQS})
    scale = {f: 1.0 / ((2 * np.pi * f) ** 2 * RHO * np.median(np.abs(us[f][mask]))) for f in FREQS}
    w2m = np.mean([(2 * np.pi * f) ** 2 for f in FREQS])
    ny, nx = mask.shape
    out = {k: np.full(mask.shape, np.nan) for k in ["choice", "r2", "h", "e", "G_phys"]}

    def rows(Ff, f, ys, xs, oy, ox, corr=1.0):
        A0 = Ff["lap"][ys, xs] * corr
        Ax = ox * H * A0 + (Ff["xlap"][ys, xs] + Ff["dx"][ys, xs]) * corr
        Ay = oy * H * A0 + (Ff["ylap"][ys, xs] + Ff["dy"][ys, xs]) * corr
        U = Ff["phi"][ys, xs]
        s = scale[f]
        cols = {"M0": np.stack([A0], 1) * s, "M1": np.stack([A0, Ax, Ay], 1) * s,
                "M3": np.stack([A0, U], 1) * s, "M13": np.stack([A0, Ax, Ay, U], 1) * s}
        return cols, -RHO * (2 * np.pi * f) ** 2 * U * s

    for y0, x0 in zip(*np.nonzero(mask)):
        pts = [(y0 + oy, x0 + ox, oy, ox) for oy, ox in OFFSETS
               if 0 <= y0 + oy < ny and 0 <= x0 + ox < nx and valid_tf[y0 + oy, x0 + ox]]
        if len(pts) < MIN_TF:
            continue
        ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
        oy = np.array([p[2] for p in pts]); ox = np.array([p[3] for p in pts])
        X = {m: [] for m in MODELS}; Y = []
        Xn = [{m: [] for m in MODELS} for _ in range(len(Fn))]; Yn = [[] for _ in range(len(Fn))]
        for f in FREQS:
            # preliminary single-frequency stiffness (noise-debiased) -> grid-bias correction
            A0 = F[f]["lap"][ys, xs]; b0 = -RHO * (2 * np.pi * f) ** 2 * F[f]["phi"][ys, xs]
            num, den = np.vdot(A0, b0).real, np.vdot(A0, A0).real
            for k in range(len(Fn)):
                An_ = Fn[k][f]["lap"][ys, xs]; bn_ = -RHO * (2 * np.pi * f) ** 2 * Fn[k][f]["phi"][ys, xs]
                num -= np.vdot(An_, bn_).real / len(Fn); den -= np.vdot(An_, An_).real / len(Fn)
            g0 = num / max(den, 0.05 * np.vdot(A0, A0).real, 1e-30)
            corr = weak_bias_factor(abs(g0), f)             # grid over-reads G by this factor -> scale regressor up
            c, t = rows(F[f], f, ys, xs, oy, ox, corr)
            for m in MODELS:
                X[m].append(c[m])
            Y.append(t)
            for k in range(len(Fn)):
                c, t = rows(Fn[k][f], f, ys, xs, oy, ox, corr)
                for m in MODELS:
                    Xn[k][m].append(c[m])
                Yn[k].append(t)
        y = np.concatenate(Y)
        yn = [np.concatenate(v) for v in Yn]
        tot = np.sum(np.abs(y) ** 2) - (np.mean([np.sum(np.abs(v) ** 2) for v in yn]) if noisy else 0)

        def fit(m):
            A_ = np.concatenate(X[m]); AhA = A_.conj().T @ A_; Ahy = A_.conj().T @ y
            for k in range(len(Fn)):
                B_ = np.concatenate(Xn[k][m])
                AhA = AhA - B_.conj().T @ B_ / len(Fn)
                Ahy = Ahy - B_.conj().T @ yn[k] / len(Fn)
            beta = np.linalg.lstsq(AhA, Ahy, rcond=None)[0]
            rss = np.sum(np.abs(y - A_ @ beta) ** 2)
            excess = max(rss - np.mean([np.sum(np.abs(yn[k] - np.concatenate(Xn[k][m]) @ beta) ** 2)
                                        for k in range(len(Fn))]), 0.0) if noisy else rss
            se = np.sqrt(np.abs(np.diag((rss / max(len(y) - beta.size, 1)) * np.linalg.pinv(AhA)))) + 1e-30
            return beta, excess, se

        def effects(m, beta, se):
            G0 = abs(beta[0]) + 1e-12
            h = zh = e = ze = 0.0
            if m in ("M1", "M13"):
                h = np.hypot(abs(beta[1]), abs(beta[2])) * HALF * H / G0
                zh = max(abs(beta[1]) / se[1], abs(beta[2]) / se[2])
            if m in ("M3", "M13"):
                e = abs(beta[-1]) / (RHO * w2m); ze = abs(beta[-1]) / se[-1]
            return h, zh, e, ze

        bf, _, sf = fit("M13")
        h_full, _, e_full, _ = effects("M13", bf, sf)
        het_on, ext_on = True, True
        for _ in range(3):
            beta, excess, se = fit(LABEL[(het_on, ext_on)])
            h, zh, e, ze = effects(LABEL[(het_on, ext_on)], beta, se)
            nh = het_on and h >= TAU_HET and zh >= Z_MIN
            ne = ext_on and e >= TAU_EXTRA and ze >= Z_MIN
            if (nh, ne) == (het_on, ext_on):
                break
            het_on, ext_on = nh, ne
        m = LABEL[(het_on, ext_on)]
        beta, excess, se = fit(m)
        out["choice"][y0, x0] = MODELS.index(m)
        out["r2"][y0, x0] = 1 - excess / max(tot, 1e-300)
        out["h"][y0, x0] = h_full
        out["e"][y0, x0] = e_full
        out["G_phys"][y0, x0] = np.real(beta[0])
    return out


if __name__ == "__main__":
    files = sorted(PRED.glob("sim_*.npz"))
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    log(f"Weak SINDy on {len(files)} test brains, conditions {CONDS}")
    keys = ["choice", "r2", "h", "e", "G_phys", "het", "mixed", "G_true", "net_err"]
    pool = {c: {k: [] for k in keys} for c in CONDS}
    for n, f in enumerate(files):
        sid = int(f.stem.split("_")[1])
        d = np.load(f)
        b = MF.load_brain(BASE, sid)
        mask, G3 = b["mask"], b["G3"]
        nonbrain = 1 - np.load(BASE / "mre_project" / "stage_b" / "variant_A" / f"sim_{sid:05d}.npz")["brainfrac3"]
        ny, nx = mask.shape
        relstd = np.full(mask.shape, np.nan); mixed = np.full(mask.shape, np.nan)
        for y0, x0 in zip(*np.nonzero(mask)):
            ys = slice(max(y0 - HALF, 0), min(y0 + HALF + 1, ny)); xs = slice(max(x0 - HALF, 0), min(x0 + HALF + 1, nx))
            g = G3[ys, xs][mask[ys, xs]]
            relstd[y0, x0] = g.std() / g.mean(); mixed[y0, x0] = np.mean(nonbrain[ys, xs])
        net_err = np.abs(1000 * np.exp(np.nanmean(d["logG_MF_snr20"], axis=0)) - G3) / G3
        for c in CONDS:
            if c == "clean":
                r = fit_brain_weak({fr: b["u"][fr] for fr in FREQS}, mask)
            else:
                us = {fr: d[f"u_f{fr}_snr{c}"].astype(complex) for fr in FREQS}
                sig = {fr: np.median(np.abs(b["u"][fr][mask])) / c for fr in FREQS}
                r = fit_brain_weak(us, mask, sig, np.random.default_rng([sid, c, 77]))
            sel = mask & np.isfinite(r["choice"])
            for k in ["choice", "r2", "h", "e", "G_phys"]:
                pool[c][k].append(r[k][sel])
            pool[c]["het"].append((relstd > HET_TRUE)[sel]); pool[c]["mixed"].append((mixed > MIXED_TRUE)[sel])
            pool[c]["G_true"].append(G3[sel]); pool[c]["net_err"].append(net_err[sel])
        if (n + 1) % 10 == 0 or n + 1 == len(files):
            log(f"  {n + 1}/{len(files)} brains")

    P = {c: {k: np.concatenate(v) for k, v in pool[c].items()} for c in CONDS}
    strong_path = BASE / "mre_project" / "stage_e6" / "results.json"
    strong = json.loads(strong_path.read_text()) if strong_path.exists() else {}
    res = {}
    print("\n=== Strong form (E6) vs WEAK form (E6b) ===")
    print(f"  {'cond':>6} | {'H1 AUC strong':>13} {'weak':>6} | {'M1-fam het/homog (weak)':>24} | "
          f"{'H2 AUC strong':>13} {'weak':>6} | {'physics G err strong':>20} {'weak':>6} | coverage")
    for c in CONDS:
        p = P[c]
        het, mix = p["het"].astype(bool), p["mixed"].astype(bool)
        a1 = roc_auc_score(het, p["h"]) if 0 < het.mean() < 1 else float("nan")
        a2 = roc_auc_score(mix, p["e"]) if 0 < mix.mean() < 1 else float("nan")
        m1 = np.isin(p["choice"], [1, 3])
        gerr = float(100 * np.nanmedian(np.abs(p["G_phys"] - p["G_true"]) / p["G_true"]))
        res[str(c)] = dict(H1_auc=float(a1), H2_auc=float(a2), M1fam_het=float(100 * m1[het].mean()),
                           M1fam_homog=float(100 * m1[~het].mean()), phys_G_median_err_pct=gerr,
                           selection_pct={mm: float(100 * np.mean(p["choice"] == i)) for i, mm in enumerate(MODELS)},
                           no_fit_pct=float(100 * np.mean(p["r2"] < R2_MIN)), n=int(len(p["choice"])))
        s = strong.get(str(c), {})
        print(f"  {str(c):>6} | {s.get('H1_auc', float('nan')):13.2f} {a1:6.2f} | "
              f"{res[str(c)]['M1fam_het']:11.1f}% / {res[str(c)]['M1fam_homog']:5.1f}%     | "
              f"{s.get('H2_auc', float('nan')):13.2f} {a2:6.2f} | "
              f"{s.get('phys_G_median_err_pct', float('nan')):19.1f}% {gerr:5.1f}% | {len(p['choice'])}")

    print("\n=== H4 with the weak form: network error (SNR 20) by what the physics says (SNR 20 data) ===")
    p = P[20]; fits = p["r2"] >= R2_MIN
    groups = {"M0 fits (homogeneous)": (p["choice"] == 0) & fits, "M1 fits (heterogeneous)": (p["choice"] == 1) & fits,
              "M3 / M1+3 (extra term)": np.isin(p["choice"], [2, 3]) & fits, "no model fits": ~fits}
    res["H4"] = {}
    for name, m in groups.items():
        if m.sum():
            res["H4"][name] = dict(pct=float(100 * m.mean()), median_err=float(100 * np.median(p["net_err"][m])),
                                   bad_pct=float(100 * np.mean(p["net_err"][m] > 0.2)))
            print(f"  {name:26s} {100*m.mean():5.1f} % | network median error {100*np.median(p['net_err'][m]):4.1f} % "
                  f"| >20 % wrong {100*np.mean(p['net_err'][m] > 0.2):4.1f} %")
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xc = [str(c) for c in CONDS]
    for key, title, j in (("H1_auc", "A  Finds true heterogeneity (H1)", 0),
                          ("phys_G_median_err_pct", "B  Physics-only stiffness error", 1)):
        if strong:
            ax[j].plot(xc, [strong[str(c)][key] for c in CONDS], "o--", color="#888780", lw=2,
                       label="strong form (E6)")
        ax[j].plot(xc, [res[str(c)][key] for c in CONDS], "o-", color="#1D9E75", lw=2.5,
                   label="weak form (E6b)")
        ax[j].set(xlabel="data (noise level)", title=title)
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[0].axhline(0.5, color="0.7", lw=0.8); ax[0].set(ylabel="AUC", ylim=(0.4, 1))
    ax[1].set(ylabel="median |error| (%)", yscale="log")
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_strong_vs_weak.png", dpi=150)
    log(f"Done. Results in {OUT}")
