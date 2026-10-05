"""
v3, STEP 3 - Evaluate the multi-frequency network (ILI-MF) vs the 60 Hz network (v1 ILI)
======================================================================================
Fair comparison: the 60 Hz noisy data are EXACTLY the ones v1 used (re-used from the
saved v1 predictions); 40 and 80 Hz get their own noise at the same SNR.
So v1 and v3 see identical 60 Hz scans; any difference comes from the extra frequencies.

Runs on the TEST brains (E2, E3, inclusion contrast) and on the VALIDATION brains
(saved only, needed to develop the multi-frequency trust map in step 4).

Output: mre_project/stage_g/predictions_test/, predictions_val/, results_E2_E3_mf.json,
        fig_mf_vs_v1.png
"""
import csv, json, pickle, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import mre_patches as P
import mre_patches_mf as MF
from stage_d1_evaluate import inclusion_contrast

warnings.filterwarnings("ignore")
SNRS = [5, 10, 20, 50]
SNR_MAIN = 20
N_PHASES = 8
BANDS = [(0, 4), (4, 7), (7, 10), (10, 13), (13, 100)]
BASE = Path(__file__).resolve().parent
V1_PRED = {"test": BASE / "mre_project" / "stage_d" / "predictions",
           "val": BASE / "mre_project" / "stage_f" / "predictions_val"}
OUT = BASE / "mre_project" / "stage_g"
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


def predict_mf(nets, us, mask):
    cen = P.centres(mask)
    Xs = []
    for j in range(N_PHASES):                       # phase cycling, same shift at every frequency
        ph = {f: np.full(len(cen), 2 * np.pi * j / N_PHASES) for f in MF.FREQS}
        Xs.append(MF.extract_mf(us, mask, cen, ph))
    lg = np.full((len(nets),) + mask.shape, np.nan, np.float32)
    xi = np.full((len(nets),) + mask.shape, np.nan, np.float32)
    for i, net in enumerate(nets):
        Y = np.mean([net.predict(X) for X in Xs], axis=0)
        lg[i][cen[:, 0], cen[:, 1]] = Y[:, 0]
        xi[i][cen[:, 0], cen[:, 1]] = Y[:, 1] / 5
    return lg, xi


if __name__ == "__main__":
    nets = [pickle.load(open(f, "rb")) for f in sorted((BASE / "mre_project" / "stage_c_mf" / "models")
                                                       .glob("ILI_MF_seed*.pkl"))]
    log(f"loaded {len(nets)} multi-frequency networks")
    pool = {snr: {"true": [], "v1": [], "mf": [], "dist": []} for snr in SNRS}
    contrast = {"v1": [], "mf": []}
    show = []
    for split in ("test", "val"):
        (OUT / f"predictions_{split}").mkdir(parents=True, exist_ok=True)
        files = sorted(V1_PRED[split].glob("sim_*.npz"))
        for n, f in enumerate(files):
            sid = int(f.stem.split("_")[1])
            v1 = np.load(f)
            b = MF.load_brain(BASE, sid)
            mask, G3 = b["mask"], b["G3"]
            dist = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * 3.0
            save = dict(G3=G3, xi3=b["xi3"], mask3=mask)
            for snr in SNRS:
                rng = np.random.default_rng([sid, snr, 4080])
                us = {60: v1[f"u_noisy_snr{snr}"].astype(complex)}         # identical to v1
                for fr in (40, 80):
                    us[fr] = P.add_noise(b["u"][fr], mask, snr, rng)
                lg, xi = predict_mf(nets, us, mask)
                for fr in MF.FREQS:
                    save[f"u_f{fr}_snr{snr}"] = us[fr].astype(np.complex64)
                save[f"logG_MF_snr{snr}"] = lg
                save[f"xi_MF_snr{snr}"] = xi
                save[f"logG_ILI_snr{snr}"] = v1[f"logG_ILI_snr{snr}"]
                save[f"xi_ILI_snr{snr}"] = v1[f"xi_ILI_snr{snr}"]
                if split == "test":
                    G_v1 = 1000 * np.exp(np.nanmean(v1[f"logG_ILI_snr{snr}"], axis=0))
                    G_mf = 1000 * np.exp(np.nanmean(lg, axis=0))
                    pool[snr]["true"].append(G3[mask]); pool[snr]["dist"].append(dist[mask])
                    pool[snr]["v1"].append(G_v1[mask]); pool[snr]["mf"].append(G_mf[mask])
                    if snr == SNR_MAIN:
                        bt = np.load(BASE / "mre_project" / "stage_b" / "variant_A" / f"sim_{sid:05d}.npz")
                        if "inclusions" in bt.files:
                            contrast["v1"] += inclusion_contrast(bt["inclusions"], G3, G_v1, mask)
                            contrast["mf"] += inclusion_contrast(bt["inclusions"], G3, G_mf, mask)
                        if len(show) < 3 and "inclusions" in bt.files and bt["inclusions"].max() > 0:
                            show.append(dict(sid=sid, mask=mask, G3=G3, v1=G_v1, mf=G_mf))
            np.savez_compressed(OUT / f"predictions_{split}" / f"sim_{sid:05d}.npz", **save)
            if (n + 1) % 25 == 0 or n + 1 == len(files):
                log(f"  {split}: {n + 1}/{len(files)} brains")

    res = {"E2": {}, "E3": {}, "contrast": {}}
    print("\nE2 (test brains, ALL brain voxels): median |error| %  /  RMSE kPa  /  correlation")
    print(f"  {'SNR':>4}   {'v1 (60 Hz)':>24}   {'v3 (40+60+80 Hz)':>24}")
    for snr in SNRS:
        t = np.concatenate(pool[snr]["true"])
        row = {}
        for k in ("v1", "mf"):
            p = np.concatenate(pool[snr][k])
            row[k] = dict(median_abs_pct=float(np.median(100 * np.abs(p - t) / t)),
                          rmse_kPa=float(np.sqrt(np.mean((p - t) ** 2)) / 1000),
                          corr=float(np.corrcoef(p, t)[0, 1]))
        res["E2"][snr] = row
        fm = lambda r: f"{r['median_abs_pct']:5.1f} % / {r['rmse_kPa']:.2f} / {r['corr']:.2f}"
        print(f"  {snr:>4}   {fm(row['v1']):>24}   {fm(row['mf']):>24}")

    print(f"\nE3 (SNR {SNR_MAIN}): median |error| % by distance to CSF")
    t = np.concatenate(pool[SNR_MAIN]["true"]); d = np.concatenate(pool[SNR_MAIN]["dist"])
    for lo, hi in BANDS:
        s = (d > lo) & (d <= hi)
        key = f"{lo}-{hi} mm" if hi < 100 else f">{lo} mm"
        res["E3"][key] = {k: float(np.median(100 * np.abs(np.concatenate(pool[SNR_MAIN][k])[s] - t[s]) / t[s]))
                          for k in ("v1", "mf")}
        print(f"  {key:>9}: v1 {res['E3'][key]['v1']:5.1f} %   v3 {res['E3'][key]['mf']:5.1f} %")

    print(f"\nInclusion contrast transfer (SNR {SNR_MAIN}; 100 % = perfect)")
    for k, name in (("v1", "v1 (60 Hz)"), ("mf", "v3 (40+60+80 Hz)")):
        c = [x["cte_pct"] for x in contrast[k] if np.isfinite(x["cte_pct"])]
        res["contrast"][k] = float(np.median(c))
        print(f"  {name}: median {np.median(c):.0f} % over {len(c)} inclusions")
    (OUT / "results_E2_E3_mf.json").write_text(json.dumps(res, indent=1))

    fig = plt.figure(figsize=(15, 4 + 3 * len(show)))
    ax0 = fig.add_subplot(1 + len(show), 2, 1)
    for k, c, lab in (("v1", "#378ADD", "v1: 60 Hz"), ("mf", "#1D9E75", "v3: 40+60+80 Hz")):
        ax0.plot(SNRS, [res["E2"][s][k]["median_abs_pct"] for s in SNRS], "o-", color=c, lw=2, label=lab)
    ax0.set(xscale="log", xlabel="SNR", ylabel="median |error| (%)", title="E2  Error vs noise (test)")
    ax0.set_xticks(SNRS); ax0.set_xticklabels(SNRS); ax0.legend(frameon=False)
    ax1 = fig.add_subplot(1 + len(show), 2, 2)
    keys = list(res["E3"])
    for k, c in (("v1", "#378ADD"), ("mf", "#1D9E75")):
        ax1.plot(range(len(keys)), [res["E3"][q][k] for q in keys], "o-", color=c, lw=2)
    ax1.set_xticks(range(len(keys))); ax1.set_xticklabels(keys)
    ax1.set(xlabel="distance to CSF", ylabel="median |error| (%)", title=f"E3  Error near CSF (SNR {SNR_MAIN})")
    for a in (ax0, ax1):
        a.spines[["top", "right"]].set_visible(False)
    for i, s in enumerate(show):
        for j, (img, title) in enumerate(((s["G3"], f"#{s['sid']} true G'"), (s["v1"], "v1 (60 Hz)"),
                                          (s["mf"], "v3 (40+60+80 Hz)"))):
            a = fig.add_subplot(1 + len(show), 3, 3 * (i + 1) + j + 1)
            a.imshow(np.where(s["mask"], img / 1000, np.nan), cmap="magma", vmin=0, vmax=5,
                     interpolation="nearest")
            a.set_title(title, fontsize=9); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_mf_vs_v1.png", dpi=140)
    log(f"Done. Results in {OUT}")
