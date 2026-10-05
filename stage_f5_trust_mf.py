"""
v3, STEP 4 - Multi-frequency trust map (+ bootstrap error bars)
==============================================================
The key question: does the physics check get stronger with 3 frequencies?

For the multi-frequency network (ILI-MF) we build four trust models, step by step:
   1. anatomy only
   2. + network signals (ensemble spread, predicted G', xi)
   3. + physics at 60 Hz only           (what v2.1 had)
   4. + physics at 40 + 60 + 80 Hz      (v3)
The difference 4 - 3 isolates what the EXTRA frequencies add to the physics check.

Physics per frequency = the v2.1 debiased "how much must G change" check, now at each
frequency (its own omega, its own 3 mm bias correction). Then the three frequencies
are combined:
   joint correction  = noise-weighted average of the three corrections
   joint z           = how many standard deviations the joint correction is from 0
   consistency (chi2)= do the three frequencies AGREE? If one stiffness can't explain
                       all three wave patterns, something is off (mixed voxel, CSF,
                       wrong network value). This is only possible with >1 frequency.

Usage:
   python stage_f5_trust_mf.py dev     2-fold over VALIDATION brains (use freely)
   python stage_f5_trust_mf.py final   fit on validation, score TEST once + bootstrap error bars
Output: mre_project/stage_g/trust_<mode>/results.json, fig_trust_mf.png
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from mre_solver import RHO
import mre_patches as P

warnings.filterwarnings("ignore")
MODE = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert MODE in ("dev", "final"), "usage: python stage_f5_trust_mf.py dev|final"
FREQS = [40, 60, 80]
SNRS = [5, 10, 20, 50]
SNR_MAIN = 20
BIG_ERROR, REGION_ERROR = 0.20, 0.10
TILE, MIN_TILE_VOX = 4, 8
RADII = {"s": 1.6, "l": 2.6}
N_DRAWS = 8
MAX_TRAIN = 150_000
N_BOOT = 1000
H = 3e-3

BASE = Path(__file__).resolve().parent
G_DIR = BASE / "mre_project" / "stage_g"
B60 = BASE / "mre_project" / "stage_b" / "variant_A"
OUT = G_DIR / f"trust_{MODE}"
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


# ------------------------------------------------------------------ physics --
def bias_3mm(G_pa, freq):
    """How much the 3 mm discrete operator over-reads stiffness. Analytic plane-wave
    formula, scaled by 0.89 to match the measured C1 table at 60 Hz."""
    k = 2 * np.pi * freq * np.sqrt(RHO / np.clip(G_pa, 500, None))
    kh = np.clip(k * H, 1e-3, 2.5)
    return 0.89 * (kh ** 2 / (2 * (1 - np.cos(kh))) - 1)


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


def bump(r):
    n = int(np.ceil(r)); yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d2 = (yy ** 2 + xx ** 2) / r ** 2
    return np.where(d2 < 1, (1 - d2) ** 2, 0.0)


KER = {k: bump(r) for k, r in RADII.items()}


def conv(a, k):
    if np.iscomplexobj(a):
        return ndimage.convolve(a.real, k, mode="constant") + 1j * ndimage.convolve(a.imag, k, mode="constant")
    return ndimage.convolve(a, k, mode="constant")


def physics_one_freq(u, Gc, mask, sigma, rng, freq):
    """Debiased stiffness correction (v2.1 method) at one frequency.
    Also returns the raw pieces (numerator N, denominator D, noise pieces) so the
    three frequencies can be POOLED into one joint estimate (see combine)."""
    w2 = (2 * np.pi * freq) ** 2
    A, valid = div_term(u, Gc, mask)
    r = np.where(valid, A + RHO * w2 * u, 0)
    An, rn = [], []
    for _ in range(N_DRAWS):
        n = sigma * (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)) / np.sqrt(2) * mask
        a, _ = div_term(n, Gc, mask)
        An.append(np.where(valid, a, 0)); rn.append(np.where(valid, a + RHO * w2 * n, 0))
    bias = bias_3mm(np.abs(Gc.real), freq)
    out = {}
    for key, k in KER.items():
        ok = mask & (conv(valid.astype(float), k) >= 0.5 * k.sum())
        AA = conv(np.abs(A) ** 2, k)
        Ar = conv(np.conj(A) * r, k)
        AnAn = np.mean([conv(np.abs(a) ** 2, k) for a in An], axis=0)
        Anrn = np.mean([conv(np.conj(a) * x, k) for a, x in zip(An, rn)], axis=0)
        D = AA - AnAn                                   # debiased curvature energy
        N = (Ar - Anrn).real                            # debiased <A, r>
        den = np.maximum(D, 0.05 * AA) + 1e-30
        Nn = [conv(np.conj(A) * x, k).real for x in rn]  # noise-only numerators
        sig = np.sqrt(np.mean([(x / den) ** 2 for x in Nn], axis=0)) + 1e-6
        eps_c = (1 - N / den) / (1 + bias) - 1
        out[f"sgn_{key}"] = np.where(ok, eps_c, np.nan)
        out[f"sig_{key}"] = np.where(ok, sig, np.nan)
        out[f"info_{key}"] = np.where(ok, np.clip(1 - AnAn / np.maximum(AA, 1e-30), 0, 1), np.nan)
        out[f"pool_{key}"] = dict(N=N, D=D, AA=AA, b=bias, Nn=Nn, ok=ok,
                                  w=1.0 / (np.mean([conv(np.abs(x) ** 2, k) for x in rn], axis=0) + 1e-30))
    return out


def combine(per_f, key):
    """POOLED joint estimate over frequencies (weighted method of moments):
        1 + eps = sum_f w_f (D_f - N_f) / sum_f w_f D_f (1 + b_f)
    w_f = 1 / (noise residual energy) depends only on the noise level, NOT on the
    measured waves -> no selection bias (unlike averaging per-frequency ratios).
    Pooling sums the curvature energy of all frequencies before dividing, so the
    noise share of the denominator is smaller and the debiasing stays stable.
    Also returns z (joint correction / its noise std) and chi2 consistency."""
    pc = [per_f[f][f"pool_{key}"] for f in FREQS]
    ok = np.any([p["ok"] for p in pc], axis=0)
    num = sum(p["w"] * (p["D"] - p["N"]) * p["ok"] for p in pc)
    den = sum(p["w"] * p["D"] * (1 + p["b"]) * p["ok"] for p in pc)
    floor = sum(0.05 * p["w"] * p["AA"] * (1 + p["b"]) * p["ok"] for p in pc)
    den = np.maximum(den, floor) + 1e-30
    ej = np.where(ok, num / den - 1, np.nan)
    noise = [sum(p["w"] * p["Nn"][i] * p["ok"] for p in pc) / den for i in range(N_DRAWS)]
    sj = np.sqrt(np.mean(np.square(noise), axis=0)) + 1e-6
    e = np.stack([per_f[f][f"sgn_{key}"] for f in FREQS])
    s = np.stack([per_f[f][f"sig_{key}"] for f in FREQS])
    chi = np.nansum((e - ej) ** 2 / s ** 2, axis=0)
    return ej, np.where(ok, np.abs(ej) / sj, np.nan), np.where(ok, chi, np.nan)


# -------------------------------------------------------------- voxel table --
ANAT = ["dist", "maskfrac", "brainfrac", "amp", "inv_snr"]
NET = ["spread", "logG", "xi"]
PH60 = [f"{m}_{k}_60" for k in RADII for m in ("z", "sgn", "eps", "info")] + ["verif_s", "verif_l"]
PHMF = [f"{m}_{k}_{f}" for f in (40, 80) for k in RADII for m in ("z", "sgn", "eps", "info")] + \
       [f"{m}_{k}" for k in RADII for m in ("zj", "sgnj", "chi")]
GROUPS = {"anatomy only": ANAT, "anatomy + network": ANAT + NET,
          "+ physics 60 Hz only": ANAT + NET + PH60,
          "+ physics 40+60+80 Hz": ANAT + NET + PH60 + PHMF}


def build_table(split):
    rows = {}
    def add(k, v):
        rows.setdefault(k, []).append(v)
    files = sorted((G_DIR / f"predictions_{split}").glob("sim_*.npz"))
    for n, f in enumerate(files):
        d = np.load(f)
        sid = int(f.stem.split("_")[1])
        c60 = np.load(B60 / f"sim_{sid:05d}.npz")
        clean = {60: c60["u3"]}
        for fr in (40, 80):
            clean[fr] = np.load(BASE / "mre_project" / "stage_b" / f"variant_A_f{fr}" / f"sim_{sid:05d}.npz")["u3"]
        mask, G3, xi3 = d["mask3"], d["G3"].astype(float), d["xi3"].astype(float)
        dist = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * 3.0
        maskfrac = ndimage.uniform_filter(mask.astype(float), P.W, mode="constant")
        cen = P.centres(mask); yy, xx = cen[:, 0], cen[:, 1]
        tiles = sid * 10000 + (yy // TILE) * 100 + xx // TILE
        for snr in SNRS:
            rng = np.random.default_rng([sid, snr, 555])
            lg = d[f"logG_MF_snr{snr}"].astype(float)
            xi_p = np.clip(np.nanmean(d[f"xi_MF_snr{snr}"], axis=0), 0, 0.5)
            G_pred = 1000 * np.exp(np.nanmean(lg, axis=0))
            Gc = np.nan_to_num(G_pred * (1 + 2j * xi_p)) * mask
            Gt = (G3 * (1 + 2j * xi3)) * mask
            per, per_true = {}, {}
            for fr in FREQS:
                u = d[f"u_f{fr}_snr{snr}"].astype(complex)
                sigma = np.median(np.abs(clean[fr][mask])) / snr
                per[fr] = physics_one_freq(u, Gc, mask, sigma, rng, fr)
                per_true[fr] = physics_one_freq(u, Gt, mask, sigma, rng, fr)
            err = np.abs(G_pred - G3) / G3
            u60 = d[f"u_f60_snr{snr}"].astype(complex)
            vals = dict(dist=dist, maskfrac=maskfrac, brainfrac=c60["brainfrac3"],
                        amp=np.abs(u60) / np.median(np.abs(u60[mask])), inv_snr=np.full(mask.shape, 1 / snr),
                        spread=np.nanstd(lg, axis=0), logG=np.nanmean(lg, axis=0), xi=xi_p,
                        verif_s=np.isfinite(per[60]["sgn_s"]).astype(float),
                        verif_l=np.isfinite(per[60]["sgn_l"]).astype(float),
                        err=err, bad=(err > BIG_ERROR).astype(float),
                        brain=np.full(mask.shape, sid), snr=np.full(mask.shape, snr),
                        G_pred=G_pred, G_true=G3)
            for fr in FREQS:
                for k in RADII:
                    sg = per[fr][f"sgn_{k}"]
                    vals[f"sgn_{k}_{fr}"] = sg
                    vals[f"eps_{k}_{fr}"] = np.abs(sg)
                    vals[f"z_{k}_{fr}"] = np.abs(sg) / per[fr][f"sig_{k}"]
                    vals[f"info_{k}_{fr}"] = per[fr][f"info_{k}"]
                vals[f"true_{fr}"] = per_true[fr]["sgn_l"]
            vals["true_joint"] = combine(per_true, "l")[0]
            for k in RADII:
                ej, zj, chi = combine(per, k)
                vals[f"sgnj_{k}"], vals[f"zj_{k}"], vals[f"chi_{k}"] = ej, zj, chi
            for k, v in vals.items():
                add(k, np.asarray(v, float)[yy, xx])
            add("tile", tiles.astype(float))
        if (n + 1) % 15 == 0 or n + 1 == len(files):
            log(f"  {split}: {n + 1}/{len(files)} brains")
    return {k: np.concatenate(v) for k, v in rows.items()}


def build_regions(T):
    key = T["tile"] * 100 + T["snr"]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    def mean(a):
        ok = np.isfinite(a)
        s = np.bincount(inv, weights=np.where(ok, a, 0), minlength=len(uniq))
        c = np.bincount(inv, weights=ok.astype(float), minlength=len(uniq))
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    feats = set(ANAT + NET + PH60 + PHMF)
    R = {k: mean(T[k]) for k in feats}
    Gp, Gt = mean(T["G_pred"]), mean(T["G_true"])
    R["bad"] = (np.abs(Gp - Gt) / Gt > REGION_ERROR).astype(float)
    R["brain"], R["snr"] = mean(T["brain"]), mean(T["snr"])
    R["negdist"] = -R["dist"]
    keep = cnt >= MIN_TILE_VOX
    return {k: v[keep] for k, v in R.items()}


# ---------------------------------------------------------------- modelling --
def design(T, cols):
    return np.nan_to_num(np.stack([T[c] for c in cols], axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(Ttr, Tev):
    idx = np.random.default_rng(0).permutation(len(Ttr["bad"]))[:MAX_TRAIN]
    out = {}
    for name, cols in GROUPS.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(design(Ttr, cols)[idx], Ttr["bad"][idx])
        out[name] = m.predict_proba(design(Tev, cols))[:, 1]
    return out


def two_fold(T):
    brains = np.unique(T["brain"])
    fold = dict(zip(brains, np.random.default_rng(1).permutation(len(brains)) % 2))
    f_of = np.array([fold[b] for b in T["brain"]])
    sub = lambda m: {k: v[m] for k, v in T.items()}
    out = {}
    for k in (0, 1):
        for name, v in fit_predict(sub(f_of != k), sub(f_of == k)).items():
            out.setdefault(name, np.full(len(f_of), np.nan))[f_of == k] = v
    return out


def auc(s, y):
    ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if ok.sum() > 20 and y[ok].min() != y[ok].max() else float("nan")


def bootstrap(S, y, brains, rng):
    ub = np.unique(brains)
    groups = {b: np.flatnonzero(brains == b) for b in ub}
    out = {k: np.empty(N_BOOT) for k in S}
    for i in range(N_BOOT):
        idx = np.concatenate([groups[b] for b in rng.choice(ub, len(ub), replace=True)])
        for k, s in S.items():
            out[k][i] = auc(s[idx], y[idx])
    return out


# --------------------------------------------------------------------- main --
if __name__ == "__main__":
    log(f"MODE = {MODE}")
    Tval = build_table("val"); Rval = build_regions(Tval)
    if MODE == "dev":
        Tev, Rev = Tval, Rval
        vs, rs = two_fold(Tval), two_fold(Rval)
    else:
        Tev = build_table("test"); Rev = build_regions(Tev)
        vs, rs = fit_predict(Tval, Tev), fit_predict(Rval, Rev)
    vs["near-CSF rule"], rs["near-CSF rule"] = -Tev["dist"], Rev["negdist"]
    order = ["near-CSF rule"] + list(GROUPS)
    res = {"sanity": {}, "voxel": {}, "region": {}}

    print("\n=== Sanity: correction with the TRUE stiffness (should be ~0 %) ===")
    for snr in SNRS:
        s = Tev["snr"] == snr
        res["sanity"][snr] = {fr: float(np.nanmedian(Tev[f"true_{fr}"][s])) for fr in FREQS}
        res["sanity"][snr]["joint"] = float(np.nanmedian(Tev["true_joint"][s]))
        print(f"  SNR {snr:>2}: " + "   ".join(f"{fr} Hz {100*res['sanity'][snr][fr]:+5.1f} %" for fr in FREQS)
              + f"   JOINT {100*res['sanity'][snr]['joint']:+5.1f} %")

    for level, T, S in (("voxel", Tev, vs), ("region", Rev, rs)):
        print(f"\n=== {level.upper()} AUC ({'>20 % error' if level == 'voxel' else '12 mm tile mean off >10 %'}) ===")
        print(f"  {'model':26s}" + "".join(f"  SNR{s_:>3}" for s_ in SNRS))
        for name in order:
            row = {snr: auc(S[name][T["snr"] == snr], T["bad"][T["snr"] == snr]) for snr in SNRS}
            res[level][name] = row
            print(f"  {name:26s}" + "".join(f"  {row[s_]:6.3f}" for s_ in SNRS))
        g60 = {s_: res[level]["+ physics 60 Hz only"][s_] - res[level]["anatomy + network"][s_] for s_ in SNRS}
        gmf = {s_: res[level]["+ physics 40+60+80 Hz"][s_] - res[level]["anatomy + network"][s_] for s_ in SNRS}
        print("  physics gain, 60 Hz only : " + "  ".join(f"SNR{s_} {g60[s_]:+.3f}" for s_ in SNRS))
        print("  physics gain, 3 freqs    : " + "  ".join(f"SNR{s_} {gmf[s_]:+.3f}" for s_ in SNRS))

    if MODE == "final":
        log(f"bootstrap: {N_BOOT} redraws of the test brains (paired)")
        rng = np.random.default_rng(2026)
        res["bootstrap"] = {}
        for level, T, S in (("voxel", Tev, vs), ("region", Rev, rs)):
            res["bootstrap"][level] = {}
            print(f"\n=== {level.upper()} bootstrap, 95 % intervals ===")
            print(f"  {'SNR':>4}  {'gain 60 Hz physics':>26}  {'gain 3-freq physics':>26}  {'extra from 40+80 Hz':>26}")
            for snr in SNRS:
                m = T["snr"] == snr
                b = bootstrap({k: S[k][m] for k in GROUPS}, T["bad"][m], T["brain"][m], rng)
                g60 = b["+ physics 60 Hz only"] - b["anatomy + network"]
                gmf = b["+ physics 40+60+80 Hz"] - b["anatomy + network"]
                ext = b["+ physics 40+60+80 Hz"] - b["+ physics 60 Hz only"]
                q = lambda a: (float(np.nanmean(a)), float(np.nanpercentile(a, 2.5)),
                               float(np.nanpercentile(a, 97.5)), float(np.mean(a > 0)))
                res["bootstrap"][level][snr] = dict(gain60=q(g60), gainMF=q(gmf), extra=q(ext))
                f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}] {100*t[3]:3.0f}%"
                print(f"  {snr:>4}  {f(q(g60)):>26}  {f(q(gmf)):>26}  {f(q(ext)):>26}")
            log(f"  {level} done")
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    cols = {"near-CSF rule": "#888780", "anatomy only": "#9FE1CB", "anatomy + network": "#5DCAA5",
            "+ physics 60 Hz only": "#378ADD", "+ physics 40+60+80 Hz": "#1D9E75"}
    for j, level in enumerate(("voxel", "region")):
        for name in order:
            ax[j].plot(SNRS, [res[level][name][s_] for s_ in SNRS], "o-", color=cols[name],
                       lw=2.5 if "40+60+80" in name else 1.5, label=name)
        ax[j].axhline(0.75, color="0.5", ls=":", lw=1)
        ax[j].set(xscale="log", xlabel="SNR", ylabel="AUC", ylim=(0.5, 0.9),
                  title=f"{'A  Voxel' if level == 'voxel' else 'B  Region'} level ({MODE})")
        ax[j].set_xticks(SNRS); ax[j].set_xticklabels(SNRS)
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_trust_mf.png", dpi=150)
    log(f"Done. Results in {OUT}")
