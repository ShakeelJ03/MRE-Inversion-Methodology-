"""
E6 - System identification: WHICH wave equation does the data follow, and where?
================================================================================
Framework: SINDy / PDE-FIND (Brunton 2016; Rudy 2017) - choose the governing
equation from a small library of candidates. Model choice by the Bayesian
Information Criterion (as in Mangan et al. 2017 for SINDy model selection).

Candidate ladder (anti-plane waves, every frequency f, G* and c frequency-independent):
  M0   homogeneous      G* lap(u) + rho w^2 u = 0                 (assumption of DI / Helmholtz inversion)
  M1   heterogeneous    div(G* grad u) + rho w^2 u = 0, G* linear in the window
                        = G0 lap(u) + gx (x lap(u) + u_x) + gy (y lap(u) + u_y)
  M3   missing physics  G* lap(u) + c u + rho w^2 u = 0          (needs >1 frequency: rho w^2 varies, c does not)
  M13  both
(M2 anisotropy is not tested: the simulator is isotropic.)

Fit: least squares in a 5x5-voxel (15 mm) window, all 3 frequencies stacked, each
frequency normalised so none dominates. The 3 mm discrete Laplacian is corrected per
frequency with the bias formula validated against the C1 table.
Select: SINDy's sequentially thresholded least squares (STLSQ; Brunton 2016): fit the
full model, drop any term whose PHYSICAL effect is negligible, refit, repeat:
   heterogeneity kept only if G changes > TAU_HET (10 %) across the half-window (6 mm)
   extra term kept only if |c| > TAU_EXTRA (5 %) of the inertia term rho w^2
(Plain BIC was tried first: on noise-free data the leftover misfit is the systematic
 3 mm discretisation error, not random noise, so BIC always rewarded the most complex
 model. Physical thresholds avoid that.)
If even the selected model explains < 90 % (R^2 < 0.9): "nothing fits".
Continuous evidence for grading: h = estimated heterogeneity, e = estimated extra term
(from the full model), independent of the thresholds.

Because the brains are synthetic, the selection can be GRADED against the truth:
  H1  M1 chosen where stiffness truly varies in the window   (true rel. std > 10 %)
  H2  M3 / "nothing fits" chosen at mixed voxels near CSF       (partial-volume fraction)
  H3  with more noise, simpler models win more often
  H4  where no model fits, the network's errors are larger      (link to the trust map)
Conditions: clean (noise-free), SNR 50, 20, 10 - test brains, same noisy data as E2/E5.

Usage:  python stage_e6_model_selection.py          (all 75 test brains)
        python stage_e6_model_selection.py 10       (first 10 brains, quick check)
Output: mre_project/stage_e6/results.json, fig_e6_maps.png, fig_e6_summary.png
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_auc_score
from mre_solver import RHO
import mre_patches_mf as MF

warnings.filterwarnings("ignore")
FREQS = MF.FREQS
CONDS = ["clean", 50, 20, 10]
H = 3e-3
HALF = 2                      # 5x5 window
MIN_ROWS = 12                 # valid voxels needed in a window
R2_MIN = 0.90
HET_TRUE = 0.10               # truly heterogeneous window: rel. std of G' > 10 %
MIXED_TRUE = 0.05             # mixed window: mean non-brain fraction > 5 %
MODELS = ["M0", "M1", "M3", "M13"]
TAU_HET = 0.10                # relative change of G across the half-window
TAU_EXTRA = 0.05              # |c| relative to rho * mean(w^2)

BASE = Path(__file__).resolve().parent
PRED = BASE / "mre_project" / "stage_g" / "predictions_test"
OUT = BASE / "mre_project" / "stage_e6"
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


def bias_3mm(G_pa, freq):
    """3 mm discrete-Laplacian over-read of stiffness (validated vs C1 table at 60 Hz)."""
    k = 2 * np.pi * freq * np.sqrt(RHO / np.clip(G_pa, 500, None))
    kh = np.clip(k * H, 1e-3, 2.5)
    return 0.89 * (kh ** 2 / (2 * (1 - np.cos(kh))) - 1)


def derivs(u, mask):
    up, mp = np.pad(u, 1), np.pad(mask, 1)
    lap = (up[2:, 1:-1] + up[:-2, 1:-1] + up[1:-1, 2:] + up[1:-1, :-2] - 4 * u) / H ** 2
    ux = (up[1:-1, 2:] - up[1:-1, :-2]) / (2 * H)
    uy = (up[2:, 1:-1] - up[:-2, 1:-1]) / (2 * H)
    ok = mask & mp[2:, 1:-1] & mp[:-2, 1:-1] & mp[1:-1, 2:] & mp[1:-1, :-2]
    return lap, ux, uy, ok


N_NOISE = 6                   # noise draws for the errors-in-variables correction
Z_MIN = 2.0                   # a term is kept only if |coefficient| > 2 standard errors
LABEL = {(True, True): "M13", (True, False): "M1", (False, True): "M3", (False, False): "M0"}


def fit_brain(us, mask, sigmas=None, rng=None):
    """Model selection at every brain voxel. us: dict freq -> wave field.
    sigmas: dict freq -> noise std (None for clean data).

    Noise correction (same idea as the E5 physics check): noise in the measured
    curvature biases least squares (errors-in-variables). We push pure noise of the
    known level through the SAME equations and subtract its average contribution
    from the normal equations (method of moments):
        (X^H X - E[Xn^H Xn]) beta = X^H y - E[Xn^H yn]
    The fit quality R^2 counts only misfit beyond what noise alone would produce."""
    D = {f: derivs(us[f], mask) for f in FREQS}
    ok = D[FREQS[0]][3]
    noisy = sigmas is not None
    if noisy:
        Dn = []
        for _ in range(N_NOISE):
            nf = {f: sigmas[f] * (rng.standard_normal(mask.shape) + 1j * rng.standard_normal(mask.shape))
                  / np.sqrt(2) * mask for f in FREQS}
            Dn.append({f: derivs(nf[f], mask) + (nf[f],) for f in FREQS})
    scale = {f: 1.0 / ((2 * np.pi * f) ** 2 * RHO * np.median(np.abs(us[f][mask]))) for f in FREQS}
    w2m = np.mean([(2 * np.pi * f) ** 2 for f in FREQS])
    ny, nx = mask.shape
    out = {k: np.full(mask.shape, np.nan) for k in ["choice", "r2", "dbic01", "dbic03", "G_phys"]}

    def columns(lap, ux, uy, U, dx, dy, corr, s):
        Lc = lap * corr
        return {"M0": np.stack([Lc], 1) * s,
                "M1": np.stack([Lc, dx * Lc + ux, dy * Lc + uy], 1) * s,
                "M3": np.stack([Lc, U], 1) * s,
                "M13": np.stack([Lc, dx * Lc + ux, dy * Lc + uy, U], 1) * s}

    for y0, x0 in zip(*np.nonzero(mask)):
        ys = slice(max(y0 - HALF, 0), min(y0 + HALF + 1, ny))
        xs = slice(max(x0 - HALF, 0), min(x0 + HALF + 1, nx))
        w_ok = ok[ys, xs]
        if w_ok.sum() < MIN_ROWS:
            continue
        yy, xx = np.nonzero(w_ok)
        dy = (yy + ys.start - y0) * H
        dx = (xx + xs.start - x0) * H
        pick = lambda a: a[ys, xs][w_ok]
        X = {m: [] for m in MODELS}; Y = []
        Xn = [{m: [] for m in MODELS} for _ in range(N_NOISE if noisy else 0)]
        Yn = [[] for _ in range(N_NOISE if noisy else 0)]
        for f in FREQS:
            w2 = (2 * np.pi * f) ** 2
            lap, ux, uy = pick(D[f][0]), pick(D[f][1]), pick(D[f][2])
            U = pick(us[f]); b = -RHO * w2 * U
            num, den = np.vdot(lap, b).real, np.vdot(lap, lap).real
            if noisy:                                   # debiased single-frequency estimate
                for k in range(N_NOISE):
                    ln, nn = pick(Dn[k][f][0]), pick(Dn[k][f][4])
                    num -= np.vdot(ln, -RHO * w2 * nn).real / N_NOISE
                    den -= np.vdot(ln, ln).real / N_NOISE
            g0 = num / max(den, 0.05 * np.vdot(lap, lap).real, 1e-30)
            corr = 1 + bias_3mm(abs(g0), f)
            s = scale[f]
            for m, c in columns(lap, ux, uy, U, dx, dy, corr, s).items():
                X[m].append(c)
            Y.append(b * s)
            for k in range(N_NOISE if noisy else 0):
                ln, uxn, uyn, nn = (pick(Dn[k][f][0]), pick(Dn[k][f][1]), pick(Dn[k][f][2]),
                                    pick(Dn[k][f][4]))
                for m, c in columns(ln, uxn, uyn, nn, dx, dy, corr, s).items():
                    Xn[k][m].append(c)
                Yn[k].append(-RHO * w2 * nn * s)
        y = np.concatenate(Y)
        yn = [np.concatenate(v) for v in Yn]
        tot = np.sum(np.abs(y) ** 2) - (np.mean([np.sum(np.abs(v) ** 2) for v in yn]) if noisy else 0)

        def fit(m):
            A_ = np.concatenate(X[m]); AhA = A_.conj().T @ A_; Ahy = A_.conj().T @ y
            if noisy:
                for k in range(N_NOISE):
                    B_ = np.concatenate(Xn[k][m])
                    AhA = AhA - B_.conj().T @ B_ / N_NOISE
                    Ahy = Ahy - B_.conj().T @ yn[k] / N_NOISE
            beta = np.linalg.lstsq(AhA, Ahy, rcond=None)[0]
            rss = np.sum(np.abs(y - A_ @ beta) ** 2)
            if noisy:                                  # misfit that noise alone would give
                rss_n = np.mean([np.sum(np.abs(yn[k] - np.concatenate(Xn[k][m]) @ beta) ** 2)
                                 for k in range(N_NOISE)])
                excess = max(rss - rss_n, 0.0)
            else:
                excess = rss
            dof = max(len(y) - beta.size, 1)
            cov = (rss / dof) * np.linalg.pinv(AhA)
            se = np.sqrt(np.abs(np.diag(cov))) + 1e-30
            return beta, excess, se

        def effects(m, beta, se):
            G0 = abs(beta[0]) + 1e-12
            h = zh = e = ze = 0.0
            if m in ("M1", "M13"):
                h = np.hypot(abs(beta[1]), abs(beta[2])) * HALF * H / G0
                zh = max(abs(beta[1]) / se[1], abs(beta[2]) / se[2])
            if m in ("M3", "M13"):
                e = abs(beta[-1]) / (RHO * w2m)
                ze = abs(beta[-1]) / se[-1]
            return h, zh, e, ze

        bf, _, sef = fit("M13")
        h_full, _, e_full, _ = effects("M13", bf, sef)
        het_on, ext_on = True, True                    # SINDy-style: start full, prune
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
        out["dbic01"][y0, x0] = h_full
        out["dbic03"][y0, x0] = e_full
        out["G_phys"][y0, x0] = np.real(beta[0])
    return out


if __name__ == "__main__":
    files = sorted(PRED.glob("sim_*.npz"))
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    log(f"E6 on {len(files)} test brains, conditions {CONDS}")
    pool = {c: {k: [] for k in ["choice", "r2", "dbic01", "dbic03", "G_phys", "het", "mixed",
                                 "relstd", "G_true", "net_err", "fits"]} for c in CONDS}
    maps = []
    for n, f in enumerate(files):
        sid = int(f.stem.split("_")[1])
        d = np.load(f)
        b = MF.load_brain(BASE, sid)
        mask, G3 = b["mask"], b["G3"]
        nonbrain = 1 - np.load(BASE / "mre_project" / "stage_b" / "variant_A" / f"sim_{sid:05d}.npz")["brainfrac3"]
        # ground truth per window
        ny, nx = mask.shape
        relstd = np.full(mask.shape, np.nan); mixed = np.full(mask.shape, np.nan)
        for y0, x0 in zip(*np.nonzero(mask)):
            ys = slice(max(y0 - HALF, 0), min(y0 + HALF + 1, ny)); xs = slice(max(x0 - HALF, 0), min(x0 + HALF + 1, nx))
            g = G3[ys, xs][mask[ys, xs]]
            relstd[y0, x0] = g.std() / g.mean()
            mixed[y0, x0] = np.mean(nonbrain[ys, xs])
        G_net = 1000 * np.exp(np.nanmean(d["logG_MF_snr20"], axis=0))
        net_err = np.abs(G_net - G3) / G3
        for c in CONDS:
            if c == "clean":
                r = fit_brain({fr: b["u"][fr] for fr in FREQS}, mask)
            else:
                us = {fr: d[f"u_f{fr}_snr{c}"].astype(complex) for fr in FREQS}
                sig = {fr: np.median(np.abs(b["u"][fr][mask])) / c for fr in FREQS}
                r = fit_brain(us, mask, sig, np.random.default_rng([sid, c, 66]))
            sel = mask & np.isfinite(r["choice"])
            for k in ["choice", "r2", "dbic01", "dbic03", "G_phys"]:
                pool[c][k].append(r[k][sel])
            pool[c]["relstd"].append(relstd[sel]); pool[c]["het"].append((relstd > HET_TRUE)[sel])
            pool[c]["mixed"].append((mixed > MIXED_TRUE)[sel]); pool[c]["G_true"].append(G3[sel])
            pool[c]["net_err"].append(net_err[sel]); pool[c]["fits"].append((r["r2"] >= R2_MIN)[sel])
            if len(maps) < 3 * len(CONDS) and n < 3:
                maps.append(dict(sid=sid, cond=c, mask=mask, G3=G3, relstd=relstd, mixed=mixed, **r))
        if (n + 1) % 10 == 0 or n + 1 == len(files):
            log(f"  {n + 1}/{len(files)} brains")

    P = {c: {k: np.concatenate(v) for k, v in pool[c].items()} for c in CONDS}
    res = {}
    print("\n=== Which model wins? (% of checked voxels; 'no fit' = best model R^2 < 0.9) ===")
    print(f"  {'cond':>6}  {'M0 homog':>9} {'M1 heter':>9} {'M3 extra':>9} {'M1+3':>7}  {'no fit':>7}  coverage")
    for c in CONDS:
        p = P[c]
        frac = {m: float(100 * np.mean(p["choice"] == i)) for i, m in enumerate(MODELS)}
        nofit = float(100 * np.mean(~p["fits"].astype(bool)))
        res[str(c)] = dict(selection_pct=frac, no_fit_pct=nofit, n=int(len(p["choice"])))
        print(f"  {str(c):>6}  {frac['M0']:8.1f}% {frac['M1']:8.1f}% {frac['M3']:8.1f}% {frac['M13']:6.1f}%  "
              f"{nofit:6.1f}%  {len(p['choice'])} voxels")

    print("\n=== Graded against the truth ===")
    print("  H1: does the estimated heterogeneity h find windows where stiffness truly varies?")
    print("  H2: does the estimated extra term e find mixed windows near CSF?")
    print(f"  {'cond':>6}  {'H1 AUC':>7}  {'M1-family when het / homog':>27}  {'H2 AUC':>7}  {'no-fit when mixed / pure':>25}")
    for c in CONDS:
        p = P[c]
        het, mix = p["het"].astype(bool), p["mixed"].astype(bool)
        m1fam = np.isin(p["choice"], [1, 3])
        a1 = roc_auc_score(het, p["dbic01"]) if 0 < het.mean() < 1 else float("nan")
        a2 = roc_auc_score(mix, p["dbic03"]) if 0 < mix.mean() < 1 else float("nan")
        nofit = ~p["fits"].astype(bool)
        res[str(c)].update(H1_auc=float(a1), M1fam_het=float(100 * m1fam[het].mean()),
                           M1fam_homog=float(100 * m1fam[~het].mean()), H2_auc=float(a2),
                           nofit_mixed=float(100 * nofit[mix].mean()), nofit_pure=float(100 * nofit[~mix].mean()),
                           het_pct=float(100 * het.mean()), mixed_pct=float(100 * mix.mean()))
        print(f"  {str(c):>6}  {a1:7.2f}  {res[str(c)]['M1fam_het']:12.1f}% / {res[str(c)]['M1fam_homog']:5.1f}%      "
              f"{a2:7.2f}  {res[str(c)]['nofit_mixed']:12.1f}% / {res[str(c)]['nofit_pure']:5.1f}%")

    print("\n  Physics-only stiffness from the selected model (median |error| vs truth, same voxels):")
    for c in CONDS:
        p = P[c]
        e = np.abs(p["G_phys"] - p["G_true"]) / p["G_true"]
        res[str(c)]["phys_G_median_err_pct"] = float(100 * np.nanmedian(e))
        print(f"    {str(c):>6}: {100*np.nanmedian(e):5.1f} %   (network ILI-MF at SNR 20 on these voxels: "
              f"{100*np.median(p['net_err']):.1f} %)")

    print("\n=== H4: network error (ILI-MF, SNR 20) grouped by what the physics says (SNR 20 data) ===")
    p = P[20]
    groups = {"M0 fits (homogeneous)": (p["choice"] == 0) & p["fits"].astype(bool),
              "M1 fits (heterogeneous)": np.isin(p["choice"], [1]) & p["fits"].astype(bool),
              "M3 / M1+3 (extra term)": np.isin(p["choice"], [2, 3]) & p["fits"].astype(bool),
              "no model fits": ~p["fits"].astype(bool)}
    res["H4"] = {}
    for name, m in groups.items():
        if m.sum() > 0:
            res["H4"][name] = dict(pct=float(100 * m.mean()), median_err=float(100 * np.median(p["net_err"][m])),
                                   bad_pct=float(100 * np.mean(p["net_err"][m] > 0.2)))
            print(f"  {name:26s} {100*m.mean():5.1f} % of voxels | network median error "
                  f"{100*np.median(p['net_err'][m]):4.1f} % | >20 % wrong: {100*np.mean(p['net_err'][m] > 0.2):4.1f} %")
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    # figures
    cmap = ListedColormap(["#9FE1CB", "#378ADD", "#D85A30", "#7F77DD"])
    show_conds = ["clean", 20]
    brains = sorted({m["sid"] for m in maps})[:3]
    fig, ax = plt.subplots(len(brains), 5, figsize=(17, 3.4 * len(brains)), squeeze=False)
    for i, sid in enumerate(brains):
        mm = {m["cond"]: m for m in maps if m["sid"] == sid}
        m0 = mm["clean"]; msk = m0["mask"]
        nm = lambda a: np.where(msk, a, np.nan)
        ax[i, 0].imshow(nm(m0["G3"] / 1000), cmap="magma", vmin=0, vmax=5, interpolation="nearest")
        ax[i, 0].set_title(f"#{sid} true G' (kPa)", fontsize=9)
        ax[i, 1].imshow(nm(100 * m0["relstd"]), cmap="Greys", vmin=0, vmax=30, interpolation="nearest")
        ax[i, 1].contour(nm(m0["mixed"]) > MIXED_TRUE, [0.5], colors="#D85A30", linewidths=0.7)
        ax[i, 1].set_title("TRUE variation in window (%)\norange = mixed/CSF", fontsize=9)
        for j, c in enumerate(show_conds):
            m = mm[c]
            ch = np.where(m["r2"] >= R2_MIN, m["choice"], np.nan)
            ax[i, 2 + j].imshow(nm(np.where(np.isfinite(m["choice"]), 0.5, np.nan)), cmap="Greys", vmin=0, vmax=1,
                                interpolation="nearest")
            ax[i, 2 + j].imshow(nm(ch), cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
            ax[i, 2 + j].set_title(f"selected model ({c})\ngrey = no model fits", fontsize=9)
        ax[i, 4].imshow(nm(mm["clean"]["r2"]), cmap="viridis", vmin=0.5, vmax=1, interpolation="nearest")
        ax[i, 4].set_title("best-model R^2 (clean)", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("E6: light green M0 homogeneous | blue M1 heterogeneous | orange M3 extra term | purple M1+3",
                 fontsize=11)
    fig.savefig(OUT / "fig_e6_maps.png", dpi=130, bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(len(CONDS)); bottom = np.zeros(len(CONDS))
    for i, (m, col) in enumerate(zip(MODELS, ["#9FE1CB", "#378ADD", "#D85A30", "#7F77DD"])):
        v = [res[str(c)]["selection_pct"][m] for c in CONDS]
        ax[0].bar(xs, v, bottom=bottom, color=col, label=m); bottom += v
    ax[0].set_xticks(xs); ax[0].set_xticklabels([str(c) for c in CONDS])
    ax[0].set(xlabel="data (noise level)", ylabel="% of voxels", title="A  Which model wins (H3)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot([str(c) for c in CONDS], [res[str(c)]["H1_auc"] for c in CONDS], "o-", color="#378ADD",
               label="H1: finds true heterogeneity")
    ax[1].plot([str(c) for c in CONDS], [res[str(c)]["H2_auc"] for c in CONDS], "s-", color="#D85A30",
               label="H2: finds mixed/CSF windows")
    ax[1].axhline(0.5, color="0.6", lw=0.8)
    ax[1].set(ylim=(0.4, 1), ylabel="AUC", xlabel="data (noise level)", title="B  Is the selection right?")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_e6_summary.png", dpi=150)
    log(f"Done. Results in {OUT}")
