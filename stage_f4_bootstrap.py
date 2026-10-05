"""
v2.1 - Error bars for the final (test) results: bootstrap over brains
====================================================================
Question: is the trust map's improvement real, or luck of which 75 test brains we had?

Bootstrap: draw 75 brains WITH replacement from the 75 test brains, recompute every AUC,
repeat N_BOOT times. The spread = how much the result would wobble with other brains.
Both methods are always scored on the SAME redraw (paired), so brain-to-brain
differences cancel and only "adding physics" remains.

Reports, per SNR: AUC with 95 % interval, physics gain with 95 % interval, and the
share of redraws where the gain is > 0.
Models are refit exactly as in the final run (fit on validation brains, score test brains).

Usage:  python stage_f4_bootstrap.py
Output: mre_project/stage_f/v21_final/bootstrap.json, fig_bootstrap.png
"""
import sys
sys.argv = [sys.argv[0], "final"]           # reuse stage_f3 settings in "final" mode
import json, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import stage_f3_trust_v21 as F

N_BOOT = 1000
SEED = 2026
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


def boot(scores, labels, brains, n_boot, rng):
    """scores: dict name -> array. Returns dict name -> array of n_boot AUCs (paired redraws)."""
    ub = np.unique(brains)
    groups = {b: np.flatnonzero(brains == b) for b in ub}
    out = {k: np.empty(n_boot) for k in scores}
    for i in range(n_boot):
        pick = rng.choice(ub, size=len(ub), replace=True)
        idx = np.concatenate([groups[b] for b in pick])
        y = labels[idx]
        for k, s in scores.items():
            sc = s[idx]
            ok = np.isfinite(sc)
            out[k][i] = roc_auc_score(y[ok], sc[ok]) if y[ok].min() != y[ok].max() else np.nan
    return out


def ci(a):
    return float(np.nanmean(a)), float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


if __name__ == "__main__":
    log("rebuilding tables and refitting models exactly as in the final run...")
    Tval = F.build_table(F.VAL_DIR)
    Rval = F.build_regions(Tval)
    Tte = F.build_table(F.TEST_DIR)
    Rte = F.build_regions(Tte)
    vs = F.logit_models(F.VOX_GROUPS, Tval, Tte)
    rs = F.logit_models(F.REG_GROUPS, Rval, Rte)
    vs["near-CSF rule"] = -Tte["dist"]
    rs["near-CSF rule"] = Rte["negdist"]
    log(f"bootstrapping {N_BOOT} redraws of the test brains...")

    rng = np.random.default_rng(SEED)
    res = {"voxel": {}, "region": {}}
    for level, T, S in (("voxel", Tte, vs), ("region", Rte, rs)):
        for snr in F.SNRS:
            m = T["snr"] == snr
            b = boot({k: v[m] for k, v in S.items()}, T["bad"][m], T["brain"][m], N_BOOT, rng)
            gain = b["anatomy + network + PHYSICS"] - b["anatomy + network"]
            over_rule = b["anatomy + network + PHYSICS"] - b["near-CSF rule"]
            res[level][snr] = {k: ci(v) for k, v in b.items()}
            res[level][snr]["physics gain"] = ci(gain)
            res[level][snr]["physics gain > 0 (share of redraws)"] = float(np.mean(gain > 0))
            res[level][snr]["full model minus CSF rule"] = ci(over_rule)
            res[level][snr]["full model beats CSF rule (share)"] = float(np.mean(over_rule > 0))
        log(f"  {level} done")
    (F.OUT / "bootstrap.json").write_text(json.dumps(res, indent=1))

    for level in ("voxel", "region"):
        print(f"\n=== {level.upper()} level (test brains, 95 % intervals from {N_BOOT} redraws) ===")
        print(f"  {'SNR':>4}  {'CSF rule':>18}  {'anat+net+PHYSICS':>18}  {'physics gain':>22}  gain>0   beats rule")
        for snr in F.SNRS:
            r = res[level][snr]
            f = lambda t: f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"
            print(f"  {snr:>4}  {f(r['near-CSF rule']):>18}  {f(r['anatomy + network + PHYSICS']):>18}  "
                  f"{r['physics gain'][0]:+.3f} [{r['physics gain'][1]:+.3f},{r['physics gain'][2]:+.3f}]  "
                  f"{100*r['physics gain > 0 (share of redraws)']:5.1f} %  "
                  f"{100*r['full model beats CSF rule (share)']:5.1f} %")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for j, level in enumerate(("voxel", "region")):
        for name, col, off in (("near-CSF rule", "#888780", -0.06), ("anatomy + network", "#5DCAA5", 0.0),
                               ("anatomy + network + PHYSICS", "#1D9E75", 0.06)):
            mu = [res[level][s][name][0] for s in F.SNRS]
            lo = [res[level][s][name][0] - res[level][s][name][1] for s in F.SNRS]
            hi = [res[level][s][name][2] - res[level][s][name][0] for s in F.SNRS]
            x = np.log10(F.SNRS) + off
            ax[j].errorbar(x, mu, yerr=[lo, hi], fmt="o-", color=col, capsize=3, label=name)
        ax[j].axhline(0.75, color="0.5", ls=":", lw=1)
        ax[j].set_xticks(np.log10(F.SNRS)); ax[j].set_xticklabels(F.SNRS)
        ax[j].set(xlabel="SNR", ylabel="AUC (95 % interval)", ylim=(0.5, 0.9),
                  title=("A  Voxel level (>20 % error)" if level == "voxel"
                         else "B  Region level (12 mm, mean off >10 %)"))
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Trust map on 75 held-out test brains (bootstrap over brains)")
    fig.tight_layout(); fig.savefig(F.OUT / "fig_bootstrap.png", dpi=150)
    log(f"Done. Saved in {F.OUT}")
