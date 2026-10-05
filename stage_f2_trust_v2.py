"""
v2, STEPS 2-4 - A better trust map
==================================
Each improvement targets one cause found in v1 (E5 AUC ~0.60, no better than
"distrust voxels near CSF"):

STEP 2  (cause 1: noise / wave nulls)  PHYSICS CORRECTION z-SCORE
   v1 asked "how big is the imbalance?". v2 asks "by how much must the stiffness
   change so the wave equation balances here, and how sure are we given the noise?"
       if true G = G_net (1 + eps) nearby, then  (1+eps) div(G_net grad u) + rho w^2 u = 0
       -> eps = -<A, r> / <A, A>   with A = div(G_net grad u), r = residual (bump-weighted)
   Its noise uncertainty sigma_eps is found by pushing pure noise through the same
   formula. z = |eps| / sigma_eps.  At wave nulls sigma_eps is huge -> z small
   ("can't tell"), not falsely "fine".
   Bias correction: at 3 mm the discrete operator reads tissue too stiff by a known
   amount (your C1 check 3 table); eps is corrected with that table.
   Computed at two bump sizes (small: thin cortex; large: less noise).

STEP 3  (cause 3: fuzzy mixed voxels -> "can't be known" uncertainty)
   ERROR-PREDICTING NETWORK: a small classifier that looks at the same 7x7 patch
   plus all warning signals and predicts "is the stiffness here >20 % wrong?".
   Trained on VALIDATION brains only (never the test brains).

STEP 4  (cause 2: tiny scattered errors) + the key scientific test
   - "PHYSICS BEYOND ANATOMY": logistic models with
         anatomy only  ->  + network signals  ->  + physics signals
     If AUC rises when physics is added, physics carries information that the
     anatomy (distance to CSF etc.) does not.
   - REGION-LEVEL AUC: 12 mm tiles; is the tile's MEAN stiffness >10 % off?
     (clinical MRE reports regional means)

All signals are available on a real scan (no truth used). Labels (true error)
are only used to train/score.

Usage:
   python stage_f2_trust_v2.py dev     develop: 2-fold cross-validation over the 75
                                       VALIDATION brains (split by brain). Use freely.
   python stage_f2_trust_v2.py final   fit on all validation brains, evaluate ONCE on
                                       the TEST brains, compare with v1.
Output: mre_project/stage_f/<dev|final>/ results.json + figures
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from mre_solver import RHO, OMEGA
import mre_patches as P

warnings.filterwarnings("ignore")

MODE = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert MODE in ("dev", "final"), "usage: python stage_f2_trust_v2.py dev|final"
SNRS = [5, 10, 20, 50]
SNR_MAIN = 20
BIG_ERROR = 0.20
REGION_ERROR = 0.10
TILE = 4                       # 4 voxels = 12 mm regions
RADII = {"s": 1.6, "l": 2.6}   # small and large bump (voxels)
N_DRAWS = 4
MAX_TRAIN = 150_000
H = 3e-3
# DI / operator bias at 3 mm from C1 check 3 (G' kPa -> % too stiff)
BIAS_G = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
BIAS_PCT = np.array([10.19, 6.63, 4.91, 3.90, 3.23, 2.76, 2.41, 2.14, 1.92])

BASE = Path(__file__).resolve().parent
B_DIR = BASE / "mre_project" / "stage_b" / "variant_A"
VAL_DIR = BASE / "mre_project" / "stage_f" / "predictions_val"
TEST_DIR = BASE / "mre_project" / "stage_d" / "predictions"
OUT = BASE / "mre_project" / "stage_f" / MODE
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------------------
# physics pieces
# ---------------------------------------------------------------------------
def div_term(u, Gc, mask):
    """A = div(G* grad u) (harmonic faces) and where it is defined."""
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


def conv(a, k):
    if np.iscomplexobj(a):
        return (ndimage.convolve(a.real, k, mode="constant")
                + 1j * ndimage.convolve(a.imag, k, mode="constant"))
    return ndimage.convolve(a, k, mode="constant")


def physics_signals(u, Gc, mask, sigma, rng):
    """v1 residual score + v2 correction z-scores at two bump sizes."""
    out = {}
    A, valid = div_term(u, Gc, mask)
    B = RHO * OMEGA ** 2 * u
    r = np.where(valid, A + B, 0)
    noise = [sigma * (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)) / np.sqrt(2)
             for _ in range(N_DRAWS)]
    noise_r = []
    for n in noise:
        An, _ = div_term(n * mask, Gc, mask)
        noise_r.append(np.where(valid, An + RHO * OMEGA ** 2 * n, 0))
    G_kpa = np.abs(Gc.real) / 1000
    bias = np.interp(np.clip(G_kpa, 1, 5), BIAS_G, BIAS_PCT) / 100
    for key, rad in RADII.items():
        k = bump(rad)
        ok = mask & (conv(valid.astype(float), k) >= 0.5 * k.sum())
        # v1-style residual / noise floor (small bump only)
        if key == "s":
            den = conv(np.abs(B) * valid, k)
            res = np.abs(conv(r, k)) / np.maximum(den, 1e-30)
            floor = np.mean([np.abs(conv(nr, k)) for nr in noise_r], axis=0) / np.maximum(den, 1e-30)
            out["res_v1"] = np.where(ok, res / np.maximum(floor, 1e-30), np.nan)
        # v2: relative stiffness correction physics asks for
        AA = conv(np.abs(A) ** 2 * valid, k)
        eps = -conv(np.conj(A) * r, k) / np.maximum(AA, 1e-30)
        eps_n = [(-conv(np.conj(A) * nr, k) / np.maximum(AA, 1e-30)).real for nr in noise_r]
        sig_eps = np.sqrt(np.mean(np.square(eps_n), axis=0)) + 1e-6
        eps_c = (1 + eps.real) / (1 + bias) - 1                 # remove the known 3 mm bias
        out[f"eps_{key}"] = np.where(ok, np.abs(eps_c), np.nan)
        out[f"z_{key}"] = np.where(ok, np.abs(eps_c) / sig_eps, np.nan)
        out[f"epsraw_{key}"] = np.where(ok, eps_c, np.nan)      # signed, for the sanity check
    return out


# ---------------------------------------------------------------------------
# build one table of voxels (features + labels) from a prediction folder
# ---------------------------------------------------------------------------
FEATS_ANAT = ["dist", "maskfrac", "brainfrac", "amp", "inv_snr"]
FEATS_NET = ["spread", "logG", "xi"]
FEATS_PHYS = ["res_v1", "eps_s", "z_s", "eps_l", "z_l", "verif_s", "verif_l"]


def build_table(pred_dir):
    rows = {k: [] for k in FEATS_ANAT + FEATS_NET + FEATS_PHYS +
            ["err", "bad", "brain", "snr", "tile", "G_pred", "G_true", "check_true"]}
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
        tiles = (sid * 10000 + (yy // TILE) * 100 + xx // TILE)
        for snr in SNRS:
            rng = np.random.default_rng([sid, snr, 123])
            u = d[f"u_noisy_snr{snr}"].astype(complex)
            sigma = np.median(np.abs(c["u3"][mask])) / snr
            lg = d[f"logG_ILI_snr{snr}"].astype(float)
            xi_p = np.clip(np.nanmean(d[f"xi_ILI_snr{snr}"], axis=0), 0, 0.5)
            G_pred = 1000 * np.exp(np.nanmean(lg, axis=0))
            Gc = np.nan_to_num(G_pred * (1 + 2j * xi_p)) * mask
            ph = physics_signals(u, Gc, mask, sigma, rng)
            # sanity: the same correction computed with the TRUE stiffness should be ~0
            ph_true = physics_signals(u, (G3 * (1 + 2j * xi3)) * mask, mask, sigma, rng)
            err = np.abs(G_pred - G3) / G3
            vals = dict(dist=dist, maskfrac=maskfrac, brainfrac=c["brainfrac3"],
                        amp=np.abs(u) / np.median(np.abs(u[mask])),
                        inv_snr=np.full(mask.shape, 1.0 / snr),
                        spread=np.nanstd(lg, axis=0), logG=np.nanmean(lg, axis=0), xi=xi_p,
                        res_v1=ph["res_v1"], eps_s=ph["eps_s"], z_s=ph["z_s"],
                        eps_l=ph["eps_l"], z_l=ph["z_l"],
                        verif_s=np.isfinite(ph["eps_s"]).astype(float),
                        verif_l=np.isfinite(ph["eps_l"]).astype(float),
                        err=err, bad=(err > BIG_ERROR).astype(float),
                        brain=np.full(mask.shape, sid), snr=np.full(mask.shape, snr),
                        G_pred=G_pred, G_true=G3, check_true=ph_true["epsraw_l"])
            for k, v in vals.items():
                rows[k].append(np.asarray(v, float)[yy, xx])
            rows["tile"].append(tiles.astype(float))
            patches.append(P.extract(u, mask, cen, phase=np.zeros(len(cen))))
        if (n + 1) % 15 == 0 or n + 1 == len(files):
            log(f"  {pred_dir.name}: {n + 1}/{len(files)} brains")
    T = {k: np.concatenate(v) for k, v in rows.items()}
    T["patch"] = np.concatenate(patches)
    return T


def design(T, cols):
    X = np.stack([T[c] for c in cols], axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


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


MODELS = {
    "anatomy only": FEATS_ANAT,
    "anatomy + network": FEATS_ANAT + FEATS_NET,
    "anatomy + network + PHYSICS": FEATS_ANAT + FEATS_NET + FEATS_PHYS,
}


def fit_predict(Ttr, Tev):
    """Fit all trust models on Ttr, return scores on Tev."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Ttr["bad"]))[:MAX_TRAIN]
    y = Ttr["bad"][idx]
    scores = {}
    for name, cols in MODELS.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        m.fit(design(Ttr, cols)[idx], y)
        scores[name] = m.predict_proba(design(Tev, cols))[:, 1]
    # step 3: error-predicting network (patch + every signal)
    allc = FEATS_ANAT + FEATS_NET + FEATS_PHYS
    sc = StandardScaler().fit(design(Ttr, allc)[idx])
    Xtr = np.hstack([Ttr["patch"][idx], sc.transform(design(Ttr, allc)[idx])])
    Xev = np.hstack([Tev["patch"], sc.transform(design(Tev, allc))])
    net = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                        max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=0)
    net.fit(Xtr, y)
    scores["v2 error-predicting network"] = net.predict_proba(Xev)[:, 1]
    return scores


def evaluate(Tev, scores):
    """Voxel AUC per SNR for single signals and models + region AUC at SNR_MAIN."""
    res = {"voxel": {}, "region": {}}
    singles = {"v1 residual": Tev["res_v1"], "v1 spread": Tev["spread"],
               "v1 combined": None, "near-CSF rule": -Tev["dist"],
               "v2 physics z (small)": Tev["z_s"], "v2 physics z (large)": Tev["z_l"],
               "v2 physics |eps| (large)": Tev["eps_l"]}
    for snr in SNRS:
        s = Tev["snr"] == snr
        y = Tev["bad"][s]
        r_rank, s_rank = pct(Tev["res_v1"][s]), pct(Tev["spread"][s])
        v1c = np.where(np.isfinite(r_rank), (r_rank + s_rank) / 2, s_rank)
        row = {}
        for name, sc in singles.items():
            row[name] = auc(v1c if name == "v1 combined" else sc[s], y)
        for name, sc in scores.items():
            row[name] = auc(sc[s], y)
        row["bad_pct"] = float(100 * y.mean())
        res["voxel"][snr] = row
    # region level at SNR_MAIN
    s = Tev["snr"] == SNR_MAIN
    tiles = Tev["tile"][s]
    uniq, inv, cnt = np.unique(tiles, return_inverse=True, return_counts=True)
    keep = cnt >= 8
    def tile_mean(a):
        return np.bincount(inv, weights=np.nan_to_num(a), minlength=len(uniq)) / cnt
    Gp, Gt = tile_mean(Tev["G_pred"][s]), tile_mean(Tev["G_true"][s])
    y_reg = (np.abs(Gp - Gt) / Gt > REGION_ERROR)[keep].astype(float)
    r_rank, s_rank = pct(Tev["res_v1"][s]), pct(Tev["spread"][s])
    v1c = np.where(np.isfinite(r_rank), (r_rank + s_rank) / 2, s_rank)
    reg_scores = {"v1 combined": v1c, "near-CSF rule": -Tev["dist"][s]}
    reg_scores.update({k: v[s] for k, v in scores.items()})
    for name, sc in reg_scores.items():
        res["region"][name] = auc(tile_mean(sc)[keep], y_reg)
    res["region"]["bad_region_pct"] = float(100 * y_reg.mean())
    res["region"]["n_regions"] = int(keep.sum())
    return res


if __name__ == "__main__":
    log(f"MODE = {MODE}")
    if not VAL_DIR.exists():
        raise SystemExit("Run stage_f1_predict_val.py first.")
    Tval = build_table(VAL_DIR)
    log(f"validation table: {len(Tval['bad'])} voxel-rows")

    if MODE == "dev":
        brains = np.unique(Tval["brain"])
        rng = np.random.default_rng(1)
        fold = dict(zip(brains, rng.permutation(len(brains)) % 2))
        f_of = np.array([fold[b] for b in Tval["brain"]])
        sub = lambda T, m: {k: v[m] for k, v in T.items()}
        scores = {}
        for k in (0, 1):
            log(f"fold {k + 1}/2: fit on half the validation brains, score the other half")
            sc = fit_predict(sub(Tval, f_of != k), sub(Tval, f_of == k))
            for name, v in sc.items():
                scores.setdefault(name, np.full(len(f_of), np.nan))[f_of == k] = v
        Tev = Tval
    else:
        log("fit on ALL validation brains, evaluate ONCE on the test brains")
        Ttest = build_table(TEST_DIR)
        scores = fit_predict(Tval, Ttest)
        Tev = Ttest

    res = evaluate(Tev, scores)
    s = Tev["snr"] == SNR_MAIN
    res["sanity_eps_with_true_G_median"] = float(np.nanmedian(Tev["check_true"][s]))
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    names = list(res["voxel"][SNR_MAIN].keys())
    names.remove("bad_pct")
    print(f"\n=== {MODE.upper()} : voxel AUC for finding >{int(BIG_ERROR*100)} % errors ===")
    print(f"  {'method':32s}" + "".join(f"  SNR{snr:>3}" for snr in SNRS))
    for nm in names:
        print(f"  {nm:32s}" + "".join(f"  {res['voxel'][snr][nm]:6.2f}" for snr in SNRS))
    print(f"\n=== Physics beyond anatomy (SNR {SNR_MAIN}) ===")
    a0 = res["voxel"][SNR_MAIN]["anatomy only"]
    a1 = res["voxel"][SNR_MAIN]["anatomy + network"]
    a2 = res["voxel"][SNR_MAIN]["anatomy + network + PHYSICS"]
    print(f"  anatomy only {a0:.3f}  -> + network {a1:.3f} ({a1-a0:+.3f})  -> + physics {a2:.3f} ({a2-a1:+.3f})")
    print(f"\n=== Region level (12 mm tiles, mean off by >{int(REGION_ERROR*100)} %, SNR {SNR_MAIN}) ===")
    print(f"  {res['region']['n_regions']} regions, {res['region']['bad_region_pct']:.1f} % bad")
    for k, v in res["region"].items():
        if k not in ("bad_region_pct", "n_regions"):
            print(f"  {k:32s} {v:6.2f}")
    print(f"\nSanity: physics correction computed with the TRUE stiffness, median "
          f"{100*res['sanity_eps_with_true_G_median']:+.1f} % (should be close to 0 after bias correction)")

    # figure: AUC by method at SNR_MAIN (voxel + region)
    show = ["near-CSF rule", "v1 combined", "v2 physics z (large)", "anatomy only",
            "anatomy + network", "anatomy + network + PHYSICS", "v2 error-predicting network"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    vals = [res["voxel"][SNR_MAIN][m] for m in show]
    cols = ["#888780", "#B4B2A9", "#7F77DD", "#9FE1CB", "#5DCAA5", "#1D9E75", "#378ADD"]
    ax[0].barh(range(len(show)), vals, color=cols)
    ax[0].set_yticks(range(len(show))); ax[0].set_yticklabels(show)
    ax[0].axvline(0.5, color="0.6", lw=0.8); ax[0].axvline(0.75, color="0.4", ls=":")
    ax[0].set(xlim=(0.45, 1.0), xlabel="AUC", title=f"A  Voxel level (SNR {SNR_MAIN}, {MODE})")
    for i, v in enumerate(vals):
        ax[0].text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    rshow = ["near-CSF rule", "v1 combined", "anatomy + network + PHYSICS", "v2 error-predicting network"]
    rv = [res["region"][m] for m in rshow]
    ax[1].barh(range(len(rshow)), rv, color=["#888780", "#B4B2A9", "#1D9E75", "#378ADD"])
    ax[1].set_yticks(range(len(rshow))); ax[1].set_yticklabels(rshow)
    ax[1].axvline(0.5, color="0.6", lw=0.8); ax[1].axvline(0.75, color="0.4", ls=":")
    ax[1].set(xlim=(0.45, 1.0), xlabel="AUC", title="B  Region level (12 mm, mean off >10 %)")
    for i, v in enumerate(rv):
        ax[1].text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_trust_v2_auc.png", dpi=150)

    # decile plot for the best model
    best = "v2 error-predicting network"
    sc = scores[best][s]
    q = np.nanpercentile(sc, np.linspace(0, 100, 11))
    err = Tev["err"][s]
    med = [100 * np.median(err[(sc >= q[i]) & (sc <= q[i + 1])]) for i in range(10)]
    fb = [100 * Tev["bad"][s][(sc >= q[i]) & (sc <= q[i + 1])].mean() for i in range(10)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 11), med, color="#378ADD")
    for i, (m, b) in enumerate(zip(med, fb)):
        ax.text(i + 1, m + 0.3, f"{b:.0f}%\nbad", ha="center", fontsize=7, color="0.3")
    ax.set(xlabel="trust decile (1 = most trusted, 10 = least)", ylabel="median |error| (%)",
           title=f"v2 trust: error by decile (SNR {SNR_MAIN}, {MODE})")
    ax.set_xticks(range(1, 11)); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_trust_v2_deciles.png", dpi=150)
    log(f"Done. Results in {OUT}")
