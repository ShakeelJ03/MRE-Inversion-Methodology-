"""
STAGE D, step 1 - Evaluate on the TEST brains (experiments E2 and E3)
====================================================================
Run AFTER stage_c2_train.py has finished.

For each of the 75 test brains (never used in training or early stopping),
at fixed noise levels SNR 5, 10, 20, 50:
  1. add noise (fixed seed, so every method sees exactly the same noisy data)
  2. DI   : direct inversion with smoothing (classical baseline)
     HLI  : network trained on uniform patches (ensemble mean)
     ILI  : network trained on realistic patches (ensemble mean + spread)
     Networks use phase cycling: average over 8 global phase shifts (Murphy 2018)
  3. compare with the true stiffness

E2  Does the network beat the formula?  error vs noise; inclusion contrast
E3  How bad is the CSF edge?            error vs distance to CSF

All methods are scored on the SAME voxels (where DI is defined); DI coverage is reported.
Predictions + the noisy data are saved for the physics checks (next step).

Output: mre_project/stage_d/  results_E2_E3.json, fig_maps.png,
        fig_error_vs_noise.png, fig_error_vs_csf.png, fig_scatter.png,
        predictions/sim_XXXXX.npz
"""
import csv, json, pickle, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import mre_patches as P
from mre_di import direct_inversion

warnings.filterwarnings("ignore", category=RuntimeWarning)

SNRS = [5, 10, 20, 50]
SNR_MAPS = 20                     # noise level used for the map / CSF figures
N_PHASES = 8
METHODS = ["DI", "HLI", "ILI"]
COLORS = {"DI": "#888780", "HLI": "#D85A30", "ILI": "#378ADD"}
CSF_BANDS_MM = [(0, 4), (4, 7), (7, 10), (10, 13), (13, 100)]   # distance to CSF/outside

BASE = Path(__file__).resolve().parent
B_DIR = BASE / "mre_project" / "stage_b" / "variant_A"
M_DIR = BASE / "mre_project" / "stage_c" / "models"
OUT = BASE / "mre_project" / "stage_d"
PRED = OUT / "predictions"
PRED.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


def load_nets(name):
    files = sorted(M_DIR.glob(f"{name}_seed*.pkl"))
    if not files:
        raise SystemExit(f"No {name} models in {M_DIR}. Run stage_c2_train.py first.")
    return [pickle.load(open(f, "rb")) for f in files]


def predict(nets, u, mask):
    """Per-member log G' and xi maps, with phase cycling."""
    cen = P.centres(mask)
    Xs = [P.extract(u, mask, cen, phase=np.full(len(cen), 2 * np.pi * j / N_PHASES))
          for j in range(N_PHASES)]
    logG = np.full((len(nets),) + mask.shape, np.nan, np.float32)
    xi = np.full((len(nets),) + mask.shape, np.nan, np.float32)
    for i, net in enumerate(nets):
        Y = np.mean([net.predict(X) for X in Xs], axis=0)
        logG[i][cen[:, 0], cen[:, 1]] = Y[:, 0]                 # log(kPa)
        xi[i][cen[:, 0], cen[:, 1]] = Y[:, 1] / 5
    return logG, xi


def inclusion_contrast(incl_1mm, G_true3, G_pred3, mask3):
    """Contrast transfer efficiency for each inclusion: recovered / true contrast
    between the inclusion and a ring of tissue around it (100 % = perfect)."""
    out = []
    f = 3
    ny, nx = incl_1mm.shape
    for k in range(1, int(incl_1mm.max()) + 1):
        frac = (incl_1mm == k).reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))
        inside = (frac > 0.5) & mask3
        if inside.sum() < 2:
            continue                                            # too small to see at 3 mm
        ring = ndimage.binary_dilation(frac > 0, iterations=3) & ~ndimage.binary_dilation(frac > 0, iterations=1) & mask3
        if ring.sum() < 4:
            continue
        true_c = np.mean(G_true3[inside]) - np.mean(G_true3[ring])
        if abs(true_c) < 500:                                   # skip lumps with < 0.5 kPa contrast
            continue
        pred_c = np.nanmean(G_pred3[inside]) - np.nanmean(G_pred3[ring])
        out.append(dict(true_contrast_kPa=true_c / 1000, cte_pct=100 * pred_c / true_c,
                        size_voxels=int(inside.sum())))
    return out


def errors(G_true, G_pred):
    e = G_pred - G_true
    return dict(rmse_kPa=float(np.sqrt(np.mean(e ** 2)) / 1000),
                median_abs_pct=float(np.median(100 * np.abs(e) / G_true)),
                bias_pct=float(np.median(100 * e / G_true)),
                corr=float(np.corrcoef(G_true, G_pred)[0, 1]))


if __name__ == "__main__":
    nets = {"HLI": load_nets("HLI"), "ILI": load_nets("ILI")}
    log(f"Loaded {len(nets['ILI'])} ILI and {len(nets['HLI'])} HLI networks")
    with open(BASE / "mre_project" / "stage_b" / "catalog_b.csv", newline="") as f:
        test_ids = sorted(int(r["sim_id"]) for r in csv.DictReader(f) if r["split"] == "test")
    log(f"{len(test_ids)} test brains, SNRs {SNRS}")

    # pooled voxel values for statistics
    pool = {snr: {"true": [], "dist": [], **{m: [] for m in METHODS}} for snr in SNRS}
    pool_all = {"true": [], "dist": [], **{m: [] for m in METHODS}}   # E3: ALL brain voxels
    di_cov = {snr: [] for snr in SNRS}
    contrast = {m: [] for m in METHODS}
    show = []

    for n, sid in enumerate(test_ids):
        d = np.load(B_DIR / f"sim_{sid:05d}.npz")
        u3, mask, G3, xi3 = d["u3"].astype(complex), d["mask3"], d["G3"].astype(float), d["xi3"]
        dist_mm = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * 3.0
        save = {}
        for snr in SNRS:
            rng = np.random.default_rng([sid, snr])               # same noise for every method
            un = P.add_noise(u3, mask, snr, rng)
            G_di, xi_di = direct_inversion(un, mask)
            lg_h, _ = predict(nets["HLI"], un, mask)
            lg_i, xi_i = predict(nets["ILI"], un, mask)
            G = {"DI": G_di,
                 "HLI": 1000 * np.exp(np.nanmean(lg_h, axis=0)),
                 "ILI": 1000 * np.exp(np.nanmean(lg_i, axis=0))}
            common = mask & np.isfinite(G_di)
            di_cov[snr].append(common.sum() / mask.sum())
            pool[snr]["true"].append(G3[common])
            pool[snr]["dist"].append(dist_mm[common])
            for m in METHODS:
                pool[snr][m].append(G[m][common])
            if snr == SNR_MAPS:                   # E3 uses every brain voxel (DI is NaN where undefined)
                pool_all["true"].append(G3[mask])
                pool_all["dist"].append(dist_mm[mask])
                for m in METHODS:
                    pool_all[m].append(G[m][mask])
            if snr == SNR_MAPS and "inclusions" in d.files:
                for m in METHODS:
                    contrast[m] += inclusion_contrast(d["inclusions"], G3, G[m], mask)
            save[f"u_noisy_snr{snr}"] = un.astype(np.complex64)
            save[f"G_DI_snr{snr}"] = G_di.astype(np.float32)
            save[f"xi_DI_snr{snr}"] = xi_di.astype(np.float32)
            save[f"logG_HLI_snr{snr}"] = lg_h
            save[f"logG_ILI_snr{snr}"] = lg_i                       # all ensemble members
            save[f"xi_ILI_snr{snr}"] = xi_i
            if snr == SNR_MAPS and len(show) < 40:
                show.append(dict(sid=sid, G3=G3, mask=mask, u=un, G=G,
                                 spread=np.nanstd(lg_i, axis=0),
                                 has_incl="inclusions" in d.files and d["inclusions"].max() > 0))
        np.savez_compressed(PRED / f"sim_{sid:05d}.npz", G3=G3, xi3=xi3, mask3=mask, **save)
        if (n + 1) % 10 == 0 or n + 1 == len(test_ids):
            log(f"  {n + 1}/{len(test_ids)} test brains")

    # ---------------- statistics ----------------
    res = {"snr": {}, "csf_bands": {}, "contrast": {}, "di_coverage_pct": {}}
    print("\nE2: stiffness error on test brains (same voxels for all methods)")
    print(f"  {'SNR':>4} {'method':>6} {'median |err| %':>15} {'RMSE kPa':>9} {'bias %':>7} {'corr':>6}")
    for snr in SNRS:
        t = np.concatenate(pool[snr]["true"])
        res["snr"][snr] = {}
        for m in METHODS:
            r = errors(t, np.concatenate(pool[snr][m]))
            res["snr"][snr][m] = r
            print(f"  {snr:>4} {m:>6} {r['median_abs_pct']:>15.1f} {r['rmse_kPa']:>9.2f} "
                  f"{r['bias_pct']:>+7.1f} {r['corr']:>6.2f}")
        res["di_coverage_pct"][snr] = float(100 * np.mean(di_cov[snr]))
    print(f"  (DI is defined on {res['di_coverage_pct'][SNR_MAPS]:.0f} % of brain voxels; "
          f"the rest are next to CSF)")

    print(f"\nE3: median |error| % by distance to CSF/outside (SNR {SNR_MAPS}, ALL brain voxels)")
    t = np.concatenate(pool_all["true"])
    dist = np.concatenate(pool_all["dist"])
    est = {m: np.concatenate(pool_all[m]) for m in METHODS}
    for lo, hi in CSF_BANDS_MM:
        sel = (dist > lo) & (dist <= hi)
        key = f"{lo}-{hi} mm" if hi < 100 else f">{lo} mm"
        res["csf_bands"][key] = {}
        for m in METHODS:
            ok = sel & np.isfinite(est[m])
            res["csf_bands"][key][m] = (float(np.median(100 * np.abs(est[m][ok] - t[ok]) / t[ok]))
                                        if ok.any() else float("nan"))
        res["csf_bands"][key]["n_voxels"] = int(sel.sum())
        res["csf_bands"][key]["DI_coverage_pct"] = (float(100 * np.isfinite(est["DI"][sel]).mean())
                                                    if sel.any() else 0.0)
        print(f"  {key:>9}: " + "  ".join(f"{m} {res['csf_bands'][key][m]:5.1f} %" for m in METHODS)
              + f"   ({sel.sum()} voxels, DI defined on {res['csf_bands'][key]['DI_coverage_pct']:.0f} %)")

    print(f"\nInclusion contrast transfer (SNR {SNR_MAPS}; 100 % = perfect)")
    for m in METHODS:
        c = [x["cte_pct"] for x in contrast[m] if np.isfinite(x["cte_pct"])]
        res["contrast"][m] = dict(n=len(c), median_cte_pct=float(np.median(c)) if c else None)
        if c:
            print(f"  {m}: median {np.median(c):5.0f} %  over {len(c)} inclusions")
    (OUT / "results_E2_E3.json").write_text(json.dumps(res, indent=1))

    # ---------------- figures ----------------
    # 1. error vs noise
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in METHODS:
        ax[0].plot(SNRS, [res["snr"][s][m]["median_abs_pct"] for s in SNRS], "o-",
                   color=COLORS[m], label=m, lw=2)
        ax[1].plot(SNRS, [res["snr"][s][m]["rmse_kPa"] for s in SNRS], "o-", color=COLORS[m], lw=2)
    ax[0].set(xscale="log", xlabel="SNR (higher = less noise)", ylabel="median |error| (%)",
              title="E2  Stiffness error vs noise (75 test brains)")
    ax[1].set(xscale="log", xlabel="SNR", ylabel="RMSE (kPa)", title="RMSE vs noise")
    for a in ax:
        a.set_xticks(SNRS); a.set_xticklabels(SNRS)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_error_vs_noise.png", dpi=150)

    # 2. error vs distance to CSF
    fig, ax = plt.subplots(figsize=(6.5, 4))
    keys = list(res["csf_bands"])
    for m in METHODS:
        ax.plot(range(len(keys)), [res["csf_bands"][k][m] for k in keys], "o-",
                color=COLORS[m], label=m, lw=2)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
    ax.set(xlabel="distance to CSF / brain edge", ylabel="median |error| (%)",
           title=f"E3  Error near CSF (SNR {SNR_MAPS})")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_error_vs_csf.png", dpi=150)

    # 3. true vs predicted
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    p = np.concatenate(pool[SNR_MAPS]["true"]) / 1000
    for j, m in enumerate(METHODS):
        q = np.concatenate(pool[SNR_MAPS][m]) / 1000
        ax[j].hexbin(p, q, gridsize=50, extent=(1, 5, 0, 7), bins="log", cmap="Blues", mincnt=1)
        ax[j].plot([1, 5], [1, 5], "k--", lw=1)
        ax[j].set(xlim=(1, 5), ylim=(0, 7), xlabel="true G' (kPa)",
                  title=f"{m}  (r = {res['snr'][SNR_MAPS][m]['corr']:.2f})")
    ax[0].set_ylabel("estimated G' (kPa)")
    fig.suptitle(f"Every test voxel, SNR {SNR_MAPS}")
    fig.tight_layout(); fig.savefig(OUT / "fig_scatter.png", dpi=150)

    # 4. maps: 3 brains (prefer ones with inclusions)
    pick = [s for s in show if s["has_incl"]][:2] + [s for s in show if not s["has_incl"]][:1]
    pick = (pick + show)[:3]
    rows = ["true", "wave", "DI", "HLI", "ILI", "spread"]
    fig, ax = plt.subplots(len(rows), len(pick), figsize=(3.4 * len(pick), 3.0 * len(rows)),
                           squeeze=False)
    for j, s in enumerate(pick):
        m = s["mask"]
        nanm = lambda a: np.where(m, a, np.nan)
        lim = np.percentile(np.abs(s["u"][m]), 98)
        panels = {"true": (nanm(s["G3"] / 1000), "magma", 0, 5, f"#{s['sid']} true G' (kPa)"),
                  "wave": (nanm(s["u"].real), "RdBu_r", -lim, lim, f"noisy wave, SNR {SNR_MAPS}"),
                  "DI": (nanm(s["G"]["DI"] / 1000), "magma", 0, 5, "DI"),
                  "HLI": (nanm(s["G"]["HLI"] / 1000), "magma", 0, 5, "HLI network"),
                  "ILI": (nanm(s["G"]["ILI"] / 1000), "magma", 0, 5, "ILI network"),
                  "spread": (nanm(s["spread"]), "viridis", 0, 0.15, "ILI ensemble spread (log)")}
        for i, r in enumerate(rows):
            a, cm, lo, hi, title = panels[r]
            im = ax[i, j].imshow(a, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax[i, j].set_title(title, fontsize=9)
            ax[i, j].axis("off")
            if j == len(pick) - 1 and r in ("true", "spread"):
                fig.colorbar(im, ax=ax[i, :], fraction=0.02)
    fig.savefig(OUT / "fig_maps.png", dpi=130, bbox_inches="tight")
    log(f"Done. Results and figures in {OUT}")