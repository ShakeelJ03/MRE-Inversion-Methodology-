"""
v2.1 - Trust map, round 2
=========================
Three changes compared with v2 (stage_f2_trust_v2.py):

FIX 1  Remove the noise bias from the physics correction.
       In v2 the sanity check failed: with the TRUE stiffness the physics said
       "21 % softer". Cause: noise adds fake curvature, which inflates the
       denominator of the correction formula, so it always says "softer".
       Fix (errors-in-variables / method of moments): we already simulate pure
       noise for the noise floor, so we know how much fake curvature energy
       (and fake correlation) noise adds on average -> subtract it.
           eps = -( <A,r> - E<A_n,r_n> ) / ( <A,A> - E<A_n,A_n> )
       Also new: "info" = fraction of the curvature energy that is real signal
       (near 0 = this spot's waves carry no stiffness information).
       Sanity check printed for EVERY SNR, before and after the fix.
       (On real data the noise level sigma would be estimated from the data.)

FIX 2  Region model trained on REGION answers.
       v2 asked voxel-trained models about regions. Now: 12 mm tiles, features
       averaged over the tile (incl. the SIGNED physics correction, which
       averages out noise and keeps systematic shifts), label = tile mean off by
       more than 10 %. Also reported: does the tile's average physics correction
       predict the tile's actual error (correlation)? If yes, physics could later
       CORRECT the network, not just flag it.

FIX 3  Practical table: keep only the X % most-trusted voxels -> how big is the error?

Usage:  python stage_f3_trust_v21.py dev      (validation brains, 2-fold by brain; use freely)
        python stage_f3_trust_v21.py final    (fit on validation, score test brains ONCE)
Output: mre_project/stage_f/v21_<mode>/
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import rankdata, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from mre_solver import RHO, OMEGA
import mre_patches as P

warnings.filterwarnings("ignore")

MODE = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert MODE in ("dev", "final"), "usage: python stage_f3_trust_v21.py dev|final"
SNRS = [5, 10, 20, 50]
SNR_MAIN = 20
BIG_ERROR, REGION_ERROR = 0.20, 0.10
TILE, MIN_TILE_VOX = 4, 8
RADII = {"s": 1.6, "l": 2.6}
N_DRAWS = 8
MAX_TRAIN = 150_000
H = 3e-3
BIAS_G = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])            # from C1 check 3
BIAS_PCT = np.array([10.19, 6.63, 4.91, 3.90, 3.23, 2.76, 2.41, 2.14, 1.92])

BASE = Path(__file__).resolve().parent
B_DIR = BASE / "mre_project" / "stage_b" / "variant_A"
VAL_DIR = BASE / "mre_project" / "stage_f" / "predictions_val"
TEST_DIR = BASE / "mre_project" / "stage_d" / "predictions"
OUT = BASE / "mre_project" / "stage_f" / f"v21_{MODE}"
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------- physics ---
def div_term(u, Gc, mask):
    up, Gp_, mp = np.pad(u, 1), np.pad(Gc, 1), np.pad(mask, 1)
    A = np.zeros(u.shape, complex)
    valid = mask.copy()
    ny, nx = u.shape
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sl = (slice(1 + dy, ny + 1 + dy), slice(1 + dx, nx + 1 + dx))
        uq, Gq, mq = up[sl], Gp_[sl], mp[sl]
        den = Gc + Gq
        Gf = np.where(np.abs(den) > 0, 2 * Gc * Gq / np.where(np.abs(den) > 0, den, 1), 0)
        A += Gf * (uq - u) / H ** 2
        valid &= mq
    return np.where(valid, A, 0), valid


def bump(radius):
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = (yy ** 2 + xx ** 2) / radius ** 2
    return np.where(d2 < 1, (1 - d2) ** 2, 0.0)


KERNELS = {k: bump(r) for k, r in RADII.items()}


def conv(a, k):
    if np.iscomplexobj(a):
        return (ndimage.convolve(a.real, k, mode="constant")
                + 1j * ndimage.convolve(a.imag, k, mode="constant"))
    return ndimage.convolve(a, k, mode="constant")


def physics_signals(u, Gc, mask, sigma, rng):
    A, valid = div_term(u, Gc, mask)
    B = RHO * OMEGA ** 2 * u
    r = np.where(valid, A + B, 0)
    An, rn = [], []
    for _ in range(N_DRAWS):
        n = sigma * (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)) / np.sqrt(2) * mask
        a, _ = div_term(n, Gc, mask)
        An.append(np.where(valid, a, 0))
        rn.append(np.where(valid, a + RHO * OMEGA ** 2 * n, 0))
    bias = np.interp(np.clip(np.abs(Gc.real) / 1000, 1, 5), BIAS_G, BIAS_PCT) / 100
    out = {}
    for key, k in KERNELS.items():
        ok = mask & (conv(valid.astype(float), k) >= 0.5 * k.sum())
        if key == "s":                                           # v1 residual score
            den = np.maximum(conv(np.abs(B) * valid, k), 1e-30)
            floor = np.mean([np.abs(conv(x, k)) for x in rn], axis=0) / den
            out["res_v1"] = np.where(ok, np.abs(conv(r, k)) / den / np.maximum(floor, 1e-30), np.nan)
        AA = conv(np.abs(A) ** 2, k)
        Ar = conv(np.conj(A) * r, k)
        AnAn = np.mean([conv(np.abs(a) ** 2, k) for a in An], axis=0)      # fake curvature energy
        Anrn = np.mean([conv(np.conj(a) * x, k) for a, x in zip(An, rn)], axis=0)
        info = np.clip(1 - AnAn / np.maximum(AA, 1e-30), 0, 1)             # share that is real signal
        den = np.maximum(AA - AnAn, 0.05 * AA) + 1e-30
        eps_old = (-Ar / np.maximum(AA, 1e-30)).real                       # v2 (biased)
        eps = (-(Ar - Anrn) / den).real                                     # v2.1 (debiased)
        sig = np.sqrt(np.mean([(-conv(np.conj(A) * x, k) / den).real ** 2 for x in rn], axis=0)) + 1e-6
        eps_c = (1 + eps) / (1 + bias) - 1                                  # remove known 3 mm bias
        eps_old_c = (1 + eps_old) / (1 + bias) - 1
        out[f"sgn_{key}"] = np.where(ok, eps_c, np.nan)
        out[f"eps_{key}"] = np.where(ok, np.abs(eps_c), np.nan)
        out[f"z_{key}"] = np.where(ok, np.abs(eps_c) / sig, np.nan)
        out[f"info_{key}"] = np.where(ok, info, np.nan)
        out[f"old_{key}"] = np.where(ok, eps_old_c, np.nan)
    return out


# ------------------------------------------------------------ voxel table ---
FEATS_ANAT = ["dist", "maskfrac", "brainfrac", "amp", "inv_snr"]
FEATS_NET = ["spread", "logG", "xi"]
FEATS_PHYS = ["res_v1", "eps_s", "z_s", "sgn_s", "info_s", "eps_l", "z_l", "sgn_l", "info_l",
              "verif_s", "verif_l"]
ALL = FEATS_ANAT + FEATS_NET + FEATS_PHYS


def build_table(pred_dir):
    keys = ALL + ["err", "relerr", "bad", "brain", "snr", "tile", "G_pred", "G_true",
                  "true_new", "true_old"]
    rows = {k: [] for k in keys}
    patches = []
    files = sorted(pred_dir.glob("sim_*.npz"))
    for n, f in enumerate(files):
        d = np.load(f)
        sid = int(f.stem.split("_")[1])
        c = np.load(B_DIR / f"sim_{sid:05d}.npz")
        mask, G3, xi3 = d["mask3"], d["G3"].astype(float), d["xi3"].astype(float)
        dist = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * 3.0
        maskfrac = ndimage.uniform_filter(mask.astype(float), P.W, mode="constant")
        cen = P.centres(mask)
        yy, xx = cen[:, 0], cen[:, 1]
        tiles = sid * 10000 + (yy // TILE) * 100 + xx // TILE
        for snr in SNRS:
            rng = np.random.default_rng([sid, snr, 321])
            u = d[f"u_noisy_snr{snr}"].astype(complex)
            sigma = np.median(np.abs(c["u3"][mask])) / snr
            lg = d[f"logG_ILI_snr{snr}"].astype(float)
            xi_p = np.clip(np.nanmean(d[f"xi_ILI_snr{snr}"], axis=0), 0, 0.5)
            G_pred = 1000 * np.exp(np.nanmean(lg, axis=0))
            ph = physics_signals(u, np.nan_to_num(G_pred * (1 + 2j * xi_p)) * mask, mask, sigma, rng)
            pt = physics_signals(u, (G3 * (1 + 2j * xi3)) * mask, mask, sigma, rng)   # sanity
            err = np.abs(G_pred - G3) / G3
            vals = dict(dist=dist, maskfrac=maskfrac, brainfrac=c["brainfrac3"],
                        amp=np.abs(u) / np.median(np.abs(u[mask])), inv_snr=np.full(mask.shape, 1 / snr),
                        spread=np.nanstd(lg, axis=0), logG=np.nanmean(lg, axis=0), xi=xi_p,
                        verif_s=np.isfinite(ph["eps_s"]).astype(float),
                        verif_l=np.isfinite(ph["eps_l"]).astype(float),
                        err=err, relerr=(G3 - G_pred) / G_pred, bad=(err > BIG_ERROR).astype(float),
                        brain=np.full(mask.shape, sid), snr=np.full(mask.shape, snr),
                        G_pred=G_pred, G_true=G3, true_new=pt["sgn_l"], true_old=pt["old_l"])
            for k in FEATS_PHYS:
                if k in ph:
                    vals[k] = ph[k]
            for k in keys:
                if k != "tile":
                    rows[k].append(np.asarray(vals[k], float)[yy, xx])
            rows["tile"].append(tiles.astype(float))
            patches.append(P.extract(u, mask, cen, phase=np.zeros(len(cen))))
        if (n + 1) % 15 == 0 or n + 1 == len(files):
            log(f"  {pred_dir.name}: {n + 1}/{len(files)} brains")
    T = {k: np.concatenate(v) for k, v in rows.items()}
    T["patch"] = np.concatenate(patches)
    return T


# ----------------------------------------------------------- region table ---
REG_ANAT = FEATS_ANAT + ["nvox"]
REG_NET = FEATS_NET
REG_PHYS = ["z_s", "z_l", "eps_s", "eps_l", "sgn_s", "sgn_l", "info_s", "info_l", "verif_s", "verif_l"]


def build_regions(T):
    key = T["tile"] * 100 + T["snr"]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    def mean(a):
        ok = np.isfinite(a)
        s = np.bincount(inv, weights=np.where(ok, a, 0), minlength=len(uniq))
        c = np.bincount(inv, weights=ok.astype(float), minlength=len(uniq))
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    R = {k: mean(T[k]) for k in set(REG_ANAT + REG_NET + REG_PHYS) - {"nvox"}}
    R["nvox"] = cnt.astype(float)
    Gp, Gt = mean(T["G_pred"]), mean(T["G_true"])
    R["relerr"] = (Gt - Gp) / Gp
    R["bad"] = (np.abs(Gp - Gt) / Gt > REGION_ERROR).astype(float)
    R["brain"] = mean(T["brain"])
    R["snr"] = mean(T["snr"])
    R["negdist"] = -R["dist"]
    r_rank, s_rank = pct(T["res_v1"]), pct(T["spread"])
    R["v1c"] = mean(np.where(np.isfinite(r_rank), (r_rank + s_rank) / 2, s_rank))
    keep = cnt >= MIN_TILE_VOX
    return {k: v[keep] for k, v in R.items()}


# ---------------------------------------------------------------- helpers ---
def design(T, cols):
    return np.nan_to_num(np.stack([T[c] for c in cols], axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def auc(score, y):
    ok = np.isfinite(score)
    if ok.sum() < 20 or y[ok].min() == y[ok].max():
        return float("nan")
    return float(roc_auc_score(y[ok], score[ok]))


def pct(a):
    o = np.full(a.shape, np.nan)
    ok = np.isfinite(a)
    o[ok] = rankdata(a[ok]) / ok.sum()
    return o


def logit_models(groups, Ttr, Tev, rng_seed=0):
    idx = np.random.default_rng(rng_seed).permutation(len(Ttr["bad"]))[:MAX_TRAIN]
    out = {}
    for name, cols in groups.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(design(Ttr, cols)[idx], Ttr["bad"][idx])
        out[name] = m.predict_proba(design(Tev, cols))[:, 1]
    return out


VOX_GROUPS = {"anatomy only": FEATS_ANAT, "anatomy + network": FEATS_ANAT + FEATS_NET,
              "anatomy + network + PHYSICS": ALL}
REG_GROUPS = {"anatomy only": REG_ANAT, "anatomy + network": REG_ANAT + REG_NET,
              "anatomy + network + PHYSICS": REG_ANAT + REG_NET + REG_PHYS}


def fit_predict(Ttr, Tev):
    sc = logit_models(VOX_GROUPS, Ttr, Tev)
    idx = np.random.default_rng(0).permutation(len(Ttr["bad"]))[:MAX_TRAIN]
    st = StandardScaler().fit(design(Ttr, ALL)[idx])
    net = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, max_iter=40,
                        early_stopping=True, n_iter_no_change=5, random_state=0)
    net.fit(np.hstack([Ttr["patch"][idx], st.transform(design(Ttr, ALL)[idx])]), Ttr["bad"][idx])
    sc["error-predicting network"] = net.predict_proba(
        np.hstack([Tev["patch"], st.transform(design(Tev, ALL))]))[:, 1]
    return sc


def two_fold(T, fitter):
    brains = np.unique(T["brain"])
    fold = dict(zip(brains, np.random.default_rng(1).permutation(len(brains)) % 2))
    f_of = np.array([fold[b] for b in T["brain"]])
    sub = lambda m: {k: v[m] for k, v in T.items()}
    out = {}
    for k in (0, 1):
        for name, v in fitter(sub(f_of != k), sub(f_of == k)).items():
            out.setdefault(name, np.full(len(f_of), np.nan))[f_of == k] = v
    return out


# ------------------------------------------------------------------- main ---
if __name__ == "__main__":
    log(f"MODE = {MODE}")
    Tval = build_table(VAL_DIR)
    Rval = build_regions(Tval)
    log(f"validation: {len(Tval['bad'])} voxel-rows, {len(Rval['bad'])} region-rows")
    if MODE == "dev":
        log("2-fold cross-validation over validation brains")
        vs = two_fold(Tval, fit_predict)
        rs = two_fold(Rval, lambda a, b: logit_models(REG_GROUPS, a, b))
        Tev, Rev = Tval, Rval
    else:
        log("fit on ALL validation brains, score the TEST brains once")
        Tev = build_table(TEST_DIR)
        Rev = build_regions(Tev)
        vs = fit_predict(Tval, Tev)
        rs = logit_models(REG_GROUPS, Rval, Rev)

    res = {"sanity": {}, "voxel": {}, "region": {}, "keep_most_trusted": {}, "region_correction_corr": {}}

    print("\n=== FIX 1: sanity check (physics correction with the TRUE stiffness; should be ~0 %) ===")
    for snr in SNRS:
        s = Tev["snr"] == snr
        old, new = np.nanmedian(Tev["true_old"][s]), np.nanmedian(Tev["true_new"][s])
        res["sanity"][snr] = dict(before=float(old), after=float(new))
        print(f"  SNR {snr:>2}: before fix {100*old:+6.1f} %   after fix {100*new:+6.1f} %")

    print(f"\n=== Voxel AUC (find >{int(BIG_ERROR*100)} % errors) ===")
    singles = {"near-CSF rule": -Tev["dist"], "v2.1 physics z (large)": Tev["z_l"],
               "v2.1 physics z (small)": Tev["z_s"]}
    print(f"  {'method':32s}" + "".join(f"  SNR{s_:>3}" for s_ in SNRS))
    for name, scr in list(singles.items()) + list(vs.items()):
        row = {snr: auc(scr[Tev["snr"] == snr], Tev["bad"][Tev["snr"] == snr]) for snr in SNRS}
        res["voxel"][name] = row
        print(f"  {name:32s}" + "".join(f"  {row[s_]:6.2f}" for s_ in SNRS))
    gain = {snr: res["voxel"]["anatomy + network + PHYSICS"][snr] - res["voxel"]["anatomy + network"][snr]
            for snr in SNRS}
    print("  physics gain over anatomy+network: " + "  ".join(f"SNR{s_} {gain[s_]:+.3f}" for s_ in SNRS))

    print(f"\n=== FIX 2: region AUC (12 mm tiles, mean off >{int(REGION_ERROR*100)} %) ===")
    reg_single = {"near-CSF rule": Rev["negdist"], "v1 combined": Rev["v1c"]}
    print(f"  {'method':32s}" + "".join(f"  SNR{s_:>3}" for s_ in SNRS))
    for name, scr in list(reg_single.items()) + list(rs.items()):
        row = {snr: auc(scr[Rev["snr"] == snr], Rev["bad"][Rev["snr"] == snr]) for snr in SNRS}
        res["region"][name] = row
        print(f"  {name:32s}" + "".join(f"  {row[s_]:6.2f}" for s_ in SNRS))
    print("  Does the region's average physics correction predict its actual error? (Pearson r)")
    for snr in SNRS:
        s = (Rev["snr"] == snr) & np.isfinite(Rev["sgn_l"])
        r = pearsonr(Rev["sgn_l"][s], Rev["relerr"][s])[0] if s.sum() > 10 else float("nan")
        res["region_correction_corr"][snr] = float(r)
        print(f"    SNR {snr:>2}: r = {r:+.2f}")

    print(f"\n=== FIX 3: keep only the most-trusted voxels (SNR {SNR_MAIN}, error-predicting network) ===")
    s = Tev["snr"] == SNR_MAIN
    sc = vs["error-predicting network"][s]
    for keep in (100, 75, 50, 25, 10):
        thr = np.nanpercentile(sc, keep)
        m = sc <= thr
        res["keep_most_trusted"][keep] = dict(median_err=float(np.median(Tev["err"][s][m])),
                                              bad_pct=float(100 * Tev["bad"][s][m].mean()))
        print(f"  keep {keep:>3} % -> median error {100*np.median(Tev['err'][s][m]):4.1f} %, "
              f"bad voxels {100*Tev['bad'][s][m].mean():4.1f} %")
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    # figure: voxel + region AUC at SNR_MAIN, and the keep-curve
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    vnames = ["near-CSF rule", "v2.1 physics z (large)", "anatomy only", "anatomy + network",
              "anatomy + network + PHYSICS", "error-predicting network"]
    cols = ["#888780", "#7F77DD", "#9FE1CB", "#5DCAA5", "#1D9E75", "#378ADD"]
    v = [res["voxel"][n][SNR_MAIN] for n in vnames]
    ax[0].barh(range(len(v)), v, color=cols); ax[0].set_yticks(range(len(v))); ax[0].set_yticklabels(vnames)
    ax[0].set(xlim=(0.45, 1), xlabel="AUC", title=f"A  Voxel level (SNR {SNR_MAIN}, {MODE})")
    rnames = ["near-CSF rule", "v1 combined", "anatomy only", "anatomy + network", "anatomy + network + PHYSICS"]
    rv = [res["region"][n][SNR_MAIN] for n in rnames]
    ax[1].barh(range(len(rv)), rv, color=["#888780", "#B4B2A9", "#9FE1CB", "#5DCAA5", "#1D9E75"])
    ax[1].set_yticks(range(len(rv))); ax[1].set_yticklabels(rnames)
    ax[1].set(xlim=(0.45, 1), xlabel="AUC", title="B  Region level (12 mm, mean off >10 %)")
    for a, vals in ((ax[0], v), (ax[1], rv)):
        a.axvline(0.5, color="0.6", lw=0.8); a.axvline(0.75, color="0.4", ls=":")
        for i, x in enumerate(vals):
            a.text(x + 0.005, i, f"{x:.2f}", va="center", fontsize=9)
    ks = [100, 75, 50, 25, 10]
    ax[2].plot(ks, [100 * res["keep_most_trusted"][k]["median_err"] for k in ks], "o-", color="#378ADD",
               label="median error")
    ax[2].plot(ks, [res["keep_most_trusted"][k]["bad_pct"] for k in ks], "s--", color="#D85A30",
               label="% voxels >20 % wrong")
    ax[2].invert_xaxis()
    ax[2].set(xlabel="% of voxels kept (most trusted first)", ylabel="%",
              title="C  Keeping only trusted voxels")
    ax[2].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_trust_v21.png", dpi=150)
    log(f"Done. Results in {OUT}")
