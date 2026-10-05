"""
STAGE E, step 1 - The trust map: "revert it back" (experiment E5, checkpoint C3)
===============================================================================
Run AFTER stage_d1_evaluate.py (it reuses the saved predictions and noisy waves).

On a real scan there is no answer key. So for every voxel we compute warning
signals that do NOT use the true stiffness:

 1. PHYSICS RESIDUAL ("revert it back")
    Put the network's stiffness back into the wave equation together with the
    measured waves:   r = div( G*_pred grad u ) + rho w^2 u
    If G*_pred is right, r is just noise. If it is wrong, r is large.
    r is averaged over a small smooth bump (weak-form style) so noise is not
    blown up, and divided by the residual that PURE NOISE would give
    (noise floor) ->  score ~1 = consistent with the data, >>1 = physics says no.
    Where the bump has no usable voxels (inside CSF gaps) -> "unverifiable".

 2. ENSEMBLE SPREAD: how much the 5 ILI networks disagree (std of log G').

 3. COMBINED trust score = average of the two (as percentile ranks).

Then the key test (only possible on synthetic data, where we know the truth):
do high-score voxels really have large errors?  ->  AUC
    AUC 0.5 = coin flip, 1.0 = perfect. Target >= 0.75.
Honest baselines: a "dumb" rule (distrust voxels near CSF; distrust weak waves).
The trust map is only useful if it beats those.

Sanity check: the residual computed with the TRUE stiffness should sit near the
noise floor (score ~1). If not, the check itself is miscalibrated.

Output: mre_project/stage_e/results_E5.json, fig_trust_maps.png,
        fig_trust_auc.png, fig_error_by_trust.png
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from mre_solver import RHO, OMEGA

warnings.filterwarnings("ignore", category=RuntimeWarning)

SNRS = [5, 10, 20, 50]
SNR_MAIN = 20
BIG_ERROR = 0.20          # a voxel is "badly wrong" if |error| > 20 %
BUMP_RADIUS = 1.6         # voxels (3x3-ish support; small so thin cortex is still checked)
N_NOISE_DRAWS = 4
H = 3e-3                  # voxel size (m)

BASE = Path(__file__).resolve().parent
PRED = BASE / "mre_project" / "stage_d" / "predictions"
B_DIR = BASE / "mre_project" / "stage_b" / "variant_A"
OUT = BASE / "mre_project" / "stage_e"
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------------------
# The physics residual
# ---------------------------------------------------------------------------
def strong_residual(u, Gc, mask):
    """div(G* grad u) + rho w^2 u with the same harmonic-face stencil as the
    simulator. Defined where the voxel and its 4 neighbours are brain."""
    up, Gp_, mp = np.pad(u, 1), np.pad(Gc, 1), np.pad(mask, 1)
    s = RHO * OMEGA ** 2 * u
    valid = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        uq = up[1 + dy:up.shape[0] - 1 + dy, 1 + dx:up.shape[1] - 1 + dx]
        Gq = Gp_[1 + dy:Gp_.shape[0] - 1 + dy, 1 + dx:Gp_.shape[1] - 1 + dx]
        mq = mp[1 + dy:mp.shape[0] - 1 + dy, 1 + dx:mp.shape[1] - 1 + dx]
        denom = Gc + Gq
        Gf = np.where(np.abs(denom) > 0, 2 * Gc * Gq / np.where(np.abs(denom) > 0, denom, 1), 0)
        s = s + Gf * (uq - u) / H ** 2
        valid &= mq
    return np.where(valid, s, 0), valid


def bump():
    r = int(np.ceil(BUMP_RADIUS))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = (yy ** 2 + xx ** 2) / BUMP_RADIUS ** 2
    return np.where(d2 < 1, (1 - d2) ** 2, 0.0)


PHI = bump()


def conv(a):
    if np.iscomplexobj(a):
        return (ndimage.convolve(a.real, PHI, mode="constant")
                + 1j * ndimage.convolve(a.imag, PHI, mode="constant"))
    return ndimage.convolve(a, PHI, mode="constant")


def local_residual(u, Gc, mask):
    """|bump-weighted residual| / bump-weighted rho w^2 |u|.
    Returns the map and where it is verifiable (>= half the bump has valid voxels)."""
    s, valid = strong_residual(u, Gc, mask)
    w = conv(valid.astype(float))
    ok = mask & (w >= 0.5 * PHI.sum())
    num = np.abs(conv(s))
    den = conv(RHO * OMEGA ** 2 * np.abs(u) * valid)
    r = np.where(ok & (den > 0), num / np.where(den > 0, den, 1), np.nan)
    return r, ok


def noise_floor(u, Gc, mask, sigma, rng):
    """Residual that pure noise of size sigma would produce with this G* (average of draws).
    Uses the same denominator as the real residual, so the ratio is 'how many times the
    noise level'. On real data, sigma would be estimated from the data."""
    s_v = strong_residual(u, Gc, mask)[1]
    den = conv(RHO * OMEGA ** 2 * np.abs(u) * s_v)
    acc = np.zeros(u.shape)
    for _ in range(N_NOISE_DRAWS):
        n = sigma * (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)) / np.sqrt(2)
        sn, _ = strong_residual(n * mask, Gc, mask)        # residual of pure noise
        acc += np.abs(conv(sn))
    return np.where(den > 0, acc / N_NOISE_DRAWS / np.where(den > 0, den, 1), np.nan)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def auc(score, bad):
    ok = np.isfinite(score)
    if ok.sum() < 10 or bad[ok].all() or (~bad[ok]).all():
        return float("nan")
    return float(roc_auc_score(bad[ok], score[ok]))


def pct_rank(a):
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a)
    out[ok] = (rankdata(a[ok]) - 0.5) / ok.sum()
    return out


if __name__ == "__main__":
    files = sorted(PRED.glob("sim_*.npz"))
    if not files:
        raise SystemExit(f"No predictions in {PRED}. Run stage_d1_evaluate.py first.")
    log(f"{len(files)} test brains")

    pool = {snr: {k: [] for k in ("err", "res", "res_true", "spread", "dist", "amp")} for snr in SNRS}
    maps = []
    for n, f in enumerate(files):
        d = np.load(f)
        sid = int(f.stem.split("_")[1])
        clean = np.load(B_DIR / f"sim_{sid:05d}.npz")
        mask, G3, xi3 = d["mask3"], d["G3"].astype(float), d["xi3"].astype(float)
        dist_mm = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * 3.0
        for snr in SNRS:
            rng = np.random.default_rng([sid, snr, 99])
            u = d[f"u_noisy_snr{snr}"].astype(complex)
            sigma = np.median(np.abs(clean["u3"][mask])) / snr
            lg = d[f"logG_ILI_snr{snr}"].astype(float)              # (members, H, W), log kPa
            xi_p = np.clip(np.nanmean(d[f"xi_ILI_snr{snr}"], axis=0), 0, 0.5)
            G_pred = 1000 * np.exp(np.nanmean(lg, axis=0))
            spread = np.nanstd(lg, axis=0)
            Gc_pred = np.nan_to_num(G_pred * (1 + 2j * xi_p)) * mask
            Gc_true = (G3 * (1 + 2j * xi3)) * mask

            r_pred, ok = local_residual(u, Gc_pred, mask)
            r_true, _ = local_residual(u, Gc_true, mask)
            floor = noise_floor(u, Gc_pred, mask, sigma, rng)
            score_res = r_pred / floor
            score_true = r_true / floor
            err = np.abs(G_pred - G3) / G3

            sel = mask & np.isfinite(G_pred)
            pool[snr]["err"].append(err[sel])
            pool[snr]["res"].append(np.where(ok, score_res, np.nan)[sel])
            pool[snr]["res_true"].append(np.where(ok, score_true, np.nan)[sel])
            pool[snr]["spread"].append(spread[sel])
            pool[snr]["dist"].append(dist_mm[sel])
            pool[snr]["amp"].append(np.abs(u)[sel] / np.median(np.abs(u[mask])))
            if snr == SNR_MAIN and len(maps) < 40:
                maps.append(dict(sid=sid, mask=mask, err=err, res=np.where(ok, score_res, np.nan),
                                 spread=spread, G3=G3, G_pred=G_pred))
        if (n + 1) % 15 == 0 or n + 1 == len(files):
            log(f"  {n + 1}/{len(files)} brains")

    # -------------------- statistics --------------------
    res = {}
    print(f"\nE5: can we flag badly wrong voxels (|error| > {int(BIG_ERROR*100)} %) WITHOUT the truth?")
    print("    AUC: 0.5 = coin flip, 1.0 = perfect\n")
    print(f"  {'SNR':>4} {'bad %':>6} {'residual':>9} {'spread':>7} {'COMBINED':>9} "
          f"{'| near-CSF':>10} {'weak-wave':>9}   verifiable %  resid(true G)/floor")
    for snr in SNRS:
        P = {k: np.concatenate(v) for k, v in pool[snr].items()}
        bad = P["err"] > BIG_ERROR
        r_rank, s_rank = pct_rank(P["res"]), pct_rank(P["spread"])
        combined = np.where(np.isfinite(r_rank), (r_rank + s_rank) / 2, s_rank)
        a = dict(bad_pct=float(100 * bad.mean()),
                 residual=auc(P["res"], bad), spread=auc(P["spread"], bad),
                 combined=auc(combined, bad),
                 near_csf_rule=auc(-P["dist"], bad), weak_wave_rule=auc(-P["amp"], bad),
                 verifiable_pct=float(100 * np.isfinite(P["res"]).mean()),
                 median_true_resid_over_floor=float(np.nanmedian(P["res_true"])),
                 median_pred_resid_over_floor=float(np.nanmedian(P["res"])),
                 spearman_combined_vs_error=float(spearmanr(combined, P["err"]).correlation))
        res[snr] = a
        print(f"  {snr:>4} {a['bad_pct']:>6.1f} {a['residual']:>9.2f} {a['spread']:>7.2f} "
              f"{a['combined']:>9.2f}   {a['near_csf_rule']:>8.2f} {a['weak_wave_rule']:>9.2f}"
              f"   {a['verifiable_pct']:>10.0f} %   {a['median_true_resid_over_floor']:>8.2f}")
        if snr == SNR_MAIN:
            main = dict(P=P, bad=bad, combined=combined)
    (OUT / "results_E5.json").write_text(json.dumps(res, indent=1))

    P, bad, combined = main["P"], main["bad"], main["combined"]
    print(f"\nAt SNR {SNR_MAIN}: median |error| in the 10 % most-trusted voxels vs 10 % least-trusted:")
    lo, hi = np.nanpercentile(combined, [10, 90])
    print(f"  most trusted {100*np.median(P['err'][combined <= lo]):.1f} %   "
          f"least trusted {100*np.median(P['err'][combined >= hi]):.1f} %")

    # -------------------- figures --------------------
    C = {"residual": "#7F77DD", "spread": "#1D9E75", "combined": "#378ADD",
         "near-CSF rule": "#888780", "weak-wave rule": "#BA7517"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, sc in (("residual", P["res"]), ("spread", P["spread"]), ("combined", combined),
                     ("near-CSF rule", -P["dist"]), ("weak-wave rule", -P["amp"])):
        ok = np.isfinite(sc)
        fpr, tpr, _ = roc_curve(bad[ok], sc[ok])
        ax[0].plot(fpr, tpr, color=C[name], lw=2.5 if name == "combined" else 1.5,
                   ls="--" if "rule" in name else "-",
                   label=f"{name} (AUC {auc(sc, bad):.2f})")
    ax[0].plot([0, 1], [0, 1], ":", color="0.6")
    ax[0].set(xlabel="false alarm rate", ylabel="bad voxels caught",
              title=f"A  Finding voxels with >{int(BIG_ERROR*100)} % error (SNR {SNR_MAIN})")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    for name, key in (("residual", "residual"), ("spread", "spread"), ("combined", "combined"),
                      ("near-CSF rule", "near_csf_rule")):
        ax[1].plot(SNRS, [res[s][key] for s in SNRS], "o-", color=C[name], label=name,
                   ls="--" if "rule" in name else "-")
    ax[1].axhline(0.75, color="0.5", ls=":", lw=1); ax[1].text(5.2, 0.76, "target 0.75", color="0.4")
    ax[1].axhline(0.5, color="0.7", lw=0.8)
    ax[1].set(xscale="log", xlabel="SNR", ylabel="AUC", ylim=(0.45, 1.0), title="B  AUC vs noise")
    ax[1].set_xticks(SNRS); ax[1].set_xticklabels(SNRS)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_trust_auc.png", dpi=150)

    # error by trust decile
    q = np.nanpercentile(combined, np.linspace(0, 100, 11))
    med = [100 * np.median(P["err"][(combined >= q[i]) & (combined <= q[i + 1])]) for i in range(10)]
    frac_bad = [100 * bad[(combined >= q[i]) & (combined <= q[i + 1])].mean() for i in range(10)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 11), med, color="#378ADD")
    for i, (m, fb) in enumerate(zip(med, frac_bad)):
        ax.text(i + 1, m + 0.3, f"{fb:.0f}%\nbad", ha="center", fontsize=7, color="0.3")
    ax.set(xlabel="trust decile (1 = most trusted, 10 = least trusted)", ylabel="median |error| (%)",
           title=f"Error grows as trust drops (SNR {SNR_MAIN}, 75 test brains)")
    ax.set_xticks(range(1, 11)); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_error_by_trust.png", dpi=150)

    # maps for 3 brains with the most bad voxels
    maps.sort(key=lambda m: -np.nanmean(m["err"][m["mask"]] > BIG_ERROR))
    pick = maps[:3]
    fig, ax = plt.subplots(len(pick), 5, figsize=(16, 3.3 * len(pick)), squeeze=False)
    for i, m in enumerate(pick):
        nm = lambda a: np.where(m["mask"], a, np.nan)
        bad_m = m["mask"] & (m["err"] > BIG_ERROR)
        panels = [(nm(m["G3"] / 1000), "magma", 0, 5, f"#{m['sid']} true G' (kPa)"),
                  (nm(m["G_pred"] / 1000), "magma", 0, 5, "ILI estimate"),
                  (nm(100 * m["err"]), "Reds", 0, 40, "TRUE error % (unknown in practice)"),
                  (nm(m["res"]), "Purples", 0, np.nanpercentile(m["res"], 95), "physics residual / noise floor"),
                  (nm(m["spread"]), "Greens", 0, np.nanpercentile(m["spread"], 95), "ensemble spread")]
        for j, (a, cm, lo_, hi_, t) in enumerate(panels):
            ax[i, j].imshow(a, cmap=cm, vmin=lo_, vmax=hi_, interpolation="nearest")
            if j >= 2:
                ax[i, j].contour(bad_m, [0.5], colors="k", linewidths=0.6)
            ax[i, j].set_title(t, fontsize=9); ax[i, j].axis("off")
    fig.suptitle(f"Black outline = voxels with >{int(BIG_ERROR*100)} % true error. "
                 "Do the warning maps light up there?", fontsize=11)
    fig.savefig(OUT / "fig_trust_maps.png", dpi=130, bbox_inches="tight")
    log(f"Done. Results and figures in {OUT}")
