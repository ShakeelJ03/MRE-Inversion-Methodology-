"""
STAGE C, step 2 - Train the feedforward inversion networks
=========================================================
Network (method document, section 4.3): fully connected,
147 inputs -> 128 -> 128 -> 128 -> 2 outputs [log G' (kPa), 5 x xi]
Adam optimiser, mean-squared-error loss, early stopping on the validation set
(validation brains are different brains from training brains).

Two models:
  ILI : trained on realistic patches  -> main model, ensemble of 5 (different random starts)
  HLI : trained on uniform patches    -> Murphy-style comparison, ensemble of 2
The ensemble's disagreement is later used as an uncertainty map (trust check).

Usage:
  python stage_c2_train.py quick     speed test: 1 net, 50k patches, 3 epochs (~1-2 min)
  python stage_c2_train.py           full training (safe to stop & restart: finished nets are skipped)

Output: mre_project/stage_c/models/ILI_seed0.pkl ... HLI_seed1.pkl
        mre_project/stage_c/training_history.json, training_curves.png
"""
import copy, json, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import mre_patches as P

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_SEEDS = {"ILI": 5, "HLI": 2}
MAX_EPOCHS = 40
PATIENCE = 5          # stop after this many epochs without validation improvement
HIDDEN = (128, 128, 128)
BATCH = 256
LR = 1e-3

BASE = Path(__file__).resolve().parent
C_DIR = BASE / "mre_project" / "stage_c"
M_DIR = C_DIR / "models"
M_DIR.mkdir(parents=True, exist_ok=True)
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


def stiffness_error(net, X, Y):
    """Median and mean absolute % error in G' (easier to read than MSE)."""
    G_pred = P.decode(net.predict(X))[0]
    G_true = P.decode(Y)[0]
    e = 100 * np.abs(G_pred - G_true) / G_true
    return float(np.median(e)), float(np.mean(e))


def train_one(name, seed, Xtr, Ytr, Xva, Yva, Xva_real, Yva_real, max_epochs):
    net = MLPRegressor(hidden_layer_sizes=HIDDEN, activation="relu", solver="adam",
                       learning_rate_init=LR, batch_size=BATCH, alpha=1e-5,
                       random_state=seed)
    rng = np.random.default_rng(seed)
    best, best_loss, bad, hist = None, np.inf, 0, []
    for ep in range(max_epochs):
        t = time.time()
        order = rng.permutation(len(Xtr))
        net.partial_fit(Xtr[order], Ytr[order])          # one pass over all training patches
        val_mse = float(np.mean((net.predict(Xva) - Yva) ** 2))
        med, _ = stiffness_error(net, Xva, Yva)
        hist.append(dict(epoch=ep, val_mse=val_mse, val_median_err_pct=med,
                         seconds=time.time() - t))
        flag = ""
        if val_mse < best_loss - 1e-5:
            best, best_loss, bad, flag = copy.deepcopy(net), val_mse, 0, "  <- best so far"
        else:
            bad += 1
        log(f"  {name} seed {seed} | epoch {ep:2d} | val MSE {val_mse:.4f} | "
            f"median G' error {med:5.1f} % | {time.time() - t:.0f} s/epoch{flag}")
        if bad >= PATIENCE:
            log(f"  {name} seed {seed}: no improvement for {PATIENCE} epochs, stopping")
            break
    # how does it do on REALISTIC validation patches? (matters most for HLI)
    med_r, mean_r = stiffness_error(best, Xva_real, Yva_real)
    log(f"  {name} seed {seed} DONE | on realistic val patches: median G' error {med_r:.1f} %, "
        f"mean {mean_r:.1f} %")
    return best, hist, dict(realistic_val_median_err_pct=med_r, realistic_val_mean_err_pct=mean_r)


if __name__ == "__main__":
    log("Loading patches...")
    tr = np.load(C_DIR / "patches_train.npz")
    va = np.load(C_DIR / "patches_val.npz")
    data = {
        "ILI": (tr["X_ili"].astype(np.float32), tr["Y_ili"],
                va["X_ili"].astype(np.float32), va["Y_ili"]),
        "HLI": (tr["X_hli"].astype(np.float32), tr["Y_hli"],
                va["X_hli"].astype(np.float32), va["Y_hli"]),
    }
    Xva_real, Yva_real = data["ILI"][2], data["ILI"][3]
    log(f"Train {len(data['ILI'][0])} patches, val {len(Xva_real)} patches")

    if QUICK:
        log("QUICK speed test: 1 ILI net, 50k patches, 3 epochs (nothing is saved)")
        Xtr, Ytr, Xva, Yva = data["ILI"]
        train_one("ILI", 0, Xtr[:50_000], Ytr[:50_000], Xva[:10_000], Yva[:10_000],
                  Xva_real[:10_000], Yva_real[:10_000], 3)
        per_epoch_full = (time.time() - START) / 3 * (len(Xtr) / 50_000)
        log(f"Estimate: ~{per_epoch_full / 60:.1f} min per full epoch, "
            f"~{per_epoch_full * 20 * sum(N_SEEDS.values()) / 3600:.1f} h for everything "
            f"(if ~20 epochs per net)")
        sys.exit()

    hist_path = C_DIR / "training_history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    for name in ("ILI", "HLI"):
        Xtr, Ytr, Xva, Yva = data[name]
        for seed in range(N_SEEDS[name]):
            out = M_DIR / f"{name}_seed{seed}.pkl"
            if out.exists():
                log(f"{name} seed {seed} already trained, skipping")
                continue
            log(f"Training {name} seed {seed}...")
            net, hist, extra = train_one(name, seed, Xtr, Ytr, Xva, Yva, Xva_real, Yva_real,
                                         MAX_EPOCHS)
            with open(out, "wb") as f:
                pickle.dump(net, f)
            history[f"{name}_seed{seed}"] = dict(epochs=hist, **extra)
            hist_path.write_text(json.dumps(history, indent=1))

    # training curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for key, h in history.items():
        ep = [e["epoch"] for e in h["epochs"]]
        c = "C0" if key.startswith("ILI") else "C3"
        ax[0].plot(ep, [e["val_mse"] for e in h["epochs"]], color=c, alpha=0.7, label=key)
        ax[1].plot(ep, [e["val_median_err_pct"] for e in h["epochs"]], color=c, alpha=0.7)
    ax[0].set(xlabel="epoch", ylabel="validation MSE", title="Validation loss (own val set)")
    ax[1].set(xlabel="epoch", ylabel="median G' error (%)", title="Validation stiffness error")
    ax[0].legend(fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C_DIR / "training_curves.png", dpi=130)

    print("\nSummary (median G' error on REALISTIC validation patches, all SNRs mixed):")
    for key, h in history.items():
        print(f"  {key:12s} {h['realistic_val_median_err_pct']:5.1f} %   "
              f"({len(h['epochs'])} epochs)")
    log("Done")
