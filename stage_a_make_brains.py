"""
STAGE A - Fake brains (stiffness maps) for synthetic-data MRE inversion
=======================================================================
Builds the "database" of fake brains that Stage B will shake to simulate
MRE waves. No MRE data is imported here: you CHOOSE the true stiffness.

Input : SynthSeg segmentations of the healthy aging templates (ages 60-90),
        example/CN_templates/age_XXdisease_0_SynthSeg.nii.gz  (DBM_with_DL repo)
Output: mre_project/stage_a/variant_A/sim_00000.npz ...   (CSF = soft solid)
        mre_project/stage_a/variant_B/sim_00000.npz ...   (CSF = fluid-like)
        mre_project/stage_a/catalog.csv                    (one row per brain)
        mre_project/stage_a/config.json                    (all settings)
        mre_project/stage_a/qc_examples.png                (check by eye)

Recipe for each fake brain (from our method document, section 4.1):
  1. Take one 2D axial slice of one template; label it CSF / grey / white.
  2. Light augmentation: left-right flip, small rotation, small warp.
  3. Regional means: grey and white matter stiffness drawn independently.
  4. Smooth random variation (+-5..20%, correlation length 2-8 mm).
  5. 0-3 inclusions (ellipses or blobs, 5-20 mm) with independent stiffness.
  6. Damping ratio xi as a smooth field; complex modulus G* = G'(1 + 2i*xi).
  7. CSF: variant A = 1 kPa, xi 0.5 (soft solid);  variant B = 0.2 kPa, xi 0.5.
Train / val / test are split by SLICE LEVEL (templates are one population
average, so different ages are not independent subjects).

Needs: numpy, scipy, nibabel, matplotlib
"""
print("Starting Stage A...", flush=True)
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Settings (change these, everything else follows)
# ---------------------------------------------------------------------------
N_PER_VARIANT = 500        # fake brains per CSF variant (start with 50 to test)
VARIANTS = ["A"]
PIXEL_MM = 1.0               # templates are 1 mm; Stage B can refine the grid
SEED = 2026
MIN_PIECE_PX = 1500
GM_MEAN_KPA = (1.5, 3.5)     # grey-matter mean stiffness range  (placeholder:
WM_MEAN_KPA = (1.8, 4.0)     # white-matter mean stiffness range  update after MRE134)
TISSUE_CLIP_KPA = (1.0, 5.0) # overall training range for tissue
FIELD_AMP = (0.05, 0.20)     # smooth variation, +-5..20 %
FIELD_CORR_MM = (2.0, 8.0)   # size of the smooth variations
N_INCLUSIONS = (0, 3)        # per brain
INCL_DIAM_MM = (5.0, 20.0)
INCL_KPA = (1.0, 5.0)
XI_MEAN = (0.08, 0.25)       # damping ratio, mean per brain (lowered so waves reach the centre)
XI_CLIP = (0.05, 0.35)
CSF = {"A": {"Gp_kPa": 1.0, "xi": 0.5}, "B": {"Gp_kPa": 0.2, "xi": 0.5}}
AUG_ROT_DEG = 10.0           # +- rotation
AUG_WARP_PX = 2.0     
CLOSE_MM = 6.0               # bridges gaps between lobes / cerebellum
RIM_MM = 2.0                 # guaranteed CSF layer between brain and skull
MIN_ISLAND_PX = 200 

          # max elastic displacement (pixels)

# SynthSeg labels -> tissue classes (0 outside, 1 CSF, 2 grey, 3 white)
CSF_LABELS = [4, 5, 14, 15, 24, 43, 44]
GM_LABELS = [3, 42, 8, 47, 10, 11, 12, 13, 17, 18, 26, 28,
             49, 50, 51, 52, 53, 54, 58, 60]
WM_LABELS = [2, 41, 7, 46, 16]
CLASS_NAMES = {0: "outside", 1: "CSF", 2: "grey", 3: "white"}

# ---------------------------------------------------------------------------
# Paths: works if this script sits in minnelab/ or in DBM_with_DL-main/
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
TDIR = next((p for p in [BASE / "example" / "CN_templates",
                         BASE / "DBM_with_DL-main" / "example" / "CN_templates"]
             if (p / "age_60disease_0_SynthSeg.nii.gz").exists()), None)
if TDIR is None:
    raise SystemExit("Can't find example/CN_templates. Put this script in minnelab/ "
                     "or DBM_with_DL-main/.")
OUT = BASE / "mre_project" / "stage_a"
START = time.time()


def log(msg):
    print(f"{msg}   ({time.time() - START:.0f} s)", flush=True)


# ---------------------------------------------------------------------------
# Step 1: load the templates and pick usable axial slice levels
# ---------------------------------------------------------------------------
AGES = list(range(60, 91))


def load_seg(age):
    img = nib.as_closest_canonical(nib.load(TDIR / f"age_{age}disease_0_SynthSeg.nii.gz"))
    return np.rint(img.get_fdata()).astype(np.int16)      # axes: x (R), y (A), z (S)


log("Loading 31 template segmentations...")
SEGS = {a: load_seg(a) for a in AGES}
area = np.array([(SEGS[75][:, :, z] > 0).sum() for z in range(SEGS[75].shape[2])])
levels = [int(z) for z in np.where(area >= 0.5 * area.max())[0]]   # big brain slices

# Split by slice level: interleaved blocks of 6 slices, 2-slice guard gaps,
# so neighbouring (near-identical) slices never land in different splits.
pattern = ["train", "train", "val", "train", "train", "test"]
split_of, block = {}, 0
for i in range(0, len(levels), 8):
    for z in levels[i:i + 6]:
        split_of[z] = pattern[block % len(pattern)]
    block += 1                                             # levels[i+6:i+8] = guard
SPLIT_LEVELS = {s: [z for z, t in split_of.items() if t == s] for s in ("train", "val", "test")}
log(f"Usable axial levels: {len(levels)}  ->  "
    + ", ".join(f"{s}: {len(v)}" for s, v in SPLIT_LEVELS.items()))


def to_classes(lab):
    cls = np.zeros(lab.shape, np.uint8)
    cls[np.isin(lab, CSF_LABELS)] = 1
    cls[np.isin(lab, GM_LABELS)] = 2
    cls[np.isin(lab, WM_LABELS)] = 3
    return cls

def disk(r_mm):
    r = int(round(r_mm / PIXEL_MM))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return np.hypot(yy, xx) <= r


def make_skull_interior(cls):
    """One connected head region with CSF all round; everything added is CSF."""
    # 1. drop tiny specks
    cc, n = ndimage.label(cls > 0)
    if n > 0:
        sizes = ndimage.sum(np.ones_like(cc), cc, index=np.arange(1, n + 1))
        keep = np.isin(cc, 1 + np.flatnonzero(sizes >= MIN_ISLAND_PX))
        cls = np.where(keep, cls, 0).astype(np.uint8)
    # 2. close gaps, fill holes, add a CSF rim (pad so the image border doesn't cut the head)
    p = int(CLOSE_MM + RIM_MM) + 2
    head = np.pad(cls > 0, p)
    head = ndimage.binary_closing(head, disk(CLOSE_MM))
    head = ndimage.binary_fill_holes(head)
    head = ndimage.binary_dilation(head, disk(RIM_MM))[p:-p, p:-p]
    out = cls.copy()
    out[head & (cls == 0)] = 1
    cc, n = ndimage.label(out > 0)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(cc), cc, index=np.arange(1, n + 1))
        out[np.isin(cc, 1 + np.flatnonzero(sizes < MIN_PIECE_PX))] = 0
    return out


# ---------------------------------------------------------------------------
# Steps 2-7: one fake brain
# ---------------------------------------------------------------------------
def augment(lab, rng):
    """Flip, small rotation and small smooth warp (nearest-neighbour: labels stay labels)."""
    flip = bool(rng.random() < 0.5)
    if flip:
        lab = lab[::-1, :]
    ang = float(rng.uniform(-AUG_ROT_DEG, AUG_ROT_DEG))
    lab = ndimage.rotate(lab, ang, reshape=False, order=0, mode="constant")
    dy, dx = (ndimage.gaussian_filter(rng.standard_normal(lab.shape), 8) for _ in range(2))
    scale = AUG_WARP_PX / max(np.abs(dy).max(), np.abs(dx).max(), 1e-9)
    yy, xx = np.mgrid[0:lab.shape[0], 0:lab.shape[1]].astype(float)
    lab = ndimage.map_coordinates(lab, [yy + scale * dy, xx + scale * dx], order=0)
    return lab, flip, ang


def smooth_field(shape, corr_mm, rng):
    """Smooth random field scaled to max |value| = 1."""
    f = ndimage.gaussian_filter(rng.standard_normal(shape), corr_mm / PIXEL_MM)
    return f / (np.abs(f).max() + 1e-12)


def add_inclusions(Gp, tissue, rng):
    yy, xx = np.mgrid[0:Gp.shape[0], 0:Gp.shape[1]]
    ys, xs = np.nonzero(tissue)
    n = int(rng.integers(N_INCLUSIONS[0], N_INCLUSIONS[1] + 1))
    kinds = []
    inc_mask = np.zeros(Gp.shape, np.uint8)      # 0 = none, 1..n = inclusion number
    for _ in range(n):
        k = rng.integers(len(ys))
        cy, cx = ys[k], xs[k]
        r1, r2 = (rng.uniform(*INCL_DIAM_MM, 2) / 2) / PIXEL_MM
        th = rng.uniform(0, np.pi)
        a = ((xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)) / r1
        b = (-(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)) / r2
        shape = a ** 2 + b ** 2 <= 1
        if rng.random() < 0.5:                             # irregular blob
            blob = ndimage.gaussian_filter(rng.standard_normal(Gp.shape), 2)
            shape &= blob > np.percentile(blob[shape], 30) if shape.any() else shape
            kinds.append("blob")
        else:
            kinds.append("ellipse")
        Gp[shape & tissue] = rng.uniform(*INCL_KPA) * 1000
        inc_mask[shape & tissue] = len(kinds)
    return n, kinds, inc_mask


def make_brain(sim_id, variant, split, rng):
    age = int(rng.choice(AGES))
    z = int(rng.choice(SPLIT_LEVELS[split]))
    lab = np.rot90(SEGS[age][:, :, z]).copy()               # anatomical orientation
    lab, flip, ang = augment(lab, rng)
    cls = to_classes(lab)
    cls = make_skull_interior(cls)

    # crop to the head with a small margin (smaller arrays = faster Stage B)
    ys, xs = np.nonzero(cls)
    m = 6
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, cls.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, cls.shape[1])
    lab, cls = lab[y0:y1, x0:x1], cls[y0:y1, x0:x1]

    # pad so height and width are multiples of 3 (Stage B averages 3x3 blocks to 3 mm)
    ph, pw = (-cls.shape[0]) % 3, (-cls.shape[1]) % 3
    cls = np.pad(cls, ((0, ph), (0, pw)))
    lab = np.pad(lab, ((0, ph), (0, pw)))
    tissue, csf = cls >= 2, cls == 1

    # storage modulus G' (Pa)
    gm, wm = rng.uniform(*GM_MEAN_KPA) * 1000, rng.uniform(*WM_MEAN_KPA) * 1000
    Gp = np.zeros(cls.shape)
    Gp[cls == 2], Gp[cls == 3] = gm, wm
    amp, corr = rng.uniform(*FIELD_AMP), rng.uniform(*FIELD_CORR_MM)
    Gp *= 1 + amp * smooth_field(cls.shape, corr, rng)
    n_inc, kinds, inc_mask = add_inclusions(Gp, tissue, rng)
    Gp[tissue] = np.clip(Gp[tissue], TISSUE_CLIP_KPA[0] * 1000, TISSUE_CLIP_KPA[1] * 1000)
    Gp[csf] = CSF[variant]["Gp_kPa"] * 1000

    # damping ratio xi
    xi_mean = rng.uniform(*XI_MEAN)
    xi = xi_mean * (1 + 0.3 * smooth_field(cls.shape, rng.uniform(*FIELD_CORR_MM), rng))
    xi = np.clip(xi, *XI_CLIP)
    xi[csf] = CSF[variant]["xi"]
    xi[cls == 0] = 0
    # complex modulus G* = G'(1 + 2i xi) is formed in Stage B from Gp and xi

    meta = dict(sim_id=sim_id, variant=variant, split=split, age=age, slice_z=z,
                flip=flip, rot_deg=round(ang, 2), gm_mean_kPa=round(gm / 1000, 3),
                wm_mean_kPa=round(wm / 1000, 3), field_amp=round(amp, 3),
                field_corr_mm=round(corr, 2), n_inclusions=n_inc,
                inclusion_kinds="+".join(kinds), xi_mean=round(xi_mean, 3),
                height=cls.shape[0], width=cls.shape[1], pixel_mm=PIXEL_MM,
                frac_csf=round(float(csf.sum() / (cls > 0).sum()), 3))
    data = dict(labels=lab.astype(np.int16), tissue=cls, Gp=Gp.astype(np.float32),
                xi=xi.astype(np.float32), domain=(cls > 0), inclusions=inc_mask)
    return data, meta


# ---------------------------------------------------------------------------
# Build the database
# ---------------------------------------------------------------------------
splits = (["train"] * int(0.7 * N_PER_VARIANT) + ["val"] * int(0.15 * N_PER_VARIANT))
splits += ["test"] * (N_PER_VARIANT - len(splits))
rows, examples = [], []
for variant in VARIANTS:
    vdir = OUT / f"variant_{variant}"
    vdir.mkdir(parents=True, exist_ok=True)
    log(f"Building {N_PER_VARIANT} fake brains, CSF variant {variant}...")
    for i, split in enumerate(splits):
        rng = np.random.default_rng([SEED, ord(variant), i])   # reproducible per brain
        data, meta = make_brain(i, variant, split, rng)
        fname = vdir / f"sim_{i:05d}.npz"
        np.savez_compressed(fname, **data)
        meta["file"] = str(fname.relative_to(OUT))
        rows.append(meta)
        if len(examples) < 6 and variant == VARIANTS[0] and i % 7 == 0:
            examples.append((data, meta))
        if (i + 1) % 100 == 0 or i + 1 == N_PER_VARIANT:
            done = int(20 * (i + 1) / N_PER_VARIANT)
            log(f"  [{'#' * done}{'.' * (20 - done)}] {i + 1}/{N_PER_VARIANT}")

with open(OUT / "catalog.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
cfg = {k: v for k, v in globals().items() if k.isupper() and k not in ("SEGS", "BASE",
                                                                        "TDIR", "OUT", "START")}
cfg["SPLIT_LEVELS"] = SPLIT_LEVELS
(OUT / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
log("Saved catalog.csv and config.json")

# ---------------------------------------------------------------------------
# Quality check figure: tissue classes, stiffness G', damping xi
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, len(examples), figsize=(3.2 * len(examples), 9.5))
for j, (d, mt) in enumerate(examples):
    gp = np.where(d["domain"], d["Gp"] / 1000, np.nan)
    xi = np.where(d["domain"], d["xi"], np.nan)
    axes[0, j].imshow(d["tissue"], cmap="viridis", vmin=0, vmax=3, interpolation="nearest")
    axes[0, j].set_title(f"#{mt['sim_id']} {mt['split']}\nage {mt['age']}, z {mt['slice_z']}",
                         fontsize=9)
    im1 = axes[1, j].imshow(gp, cmap="magma", vmin=0, vmax=5)
    if d["inclusions"].any():
        axes[1, j].contour(d["inclusions"] > 0, [0.5], colors="cyan", linewidths=1)
    axes[1, j].set_title(f"G' (kPa), {mt['n_inclusions']} inclusions", fontsize=9)
    im2 = axes[2, j].imshow(xi, cmap="cividis", vmin=0, vmax=0.5)
    axes[2, j].set_title(f"damping xi (mean {mt['xi_mean']})", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
fig.colorbar(im1, ax=axes[1, :], fraction=0.015, label="kPa")
fig.colorbar(im2, ax=axes[2, :], fraction=0.015, label="xi")
fig.suptitle(f"Stage A quality check: fake brains (CSF variant {VARIANTS[0]})\n"
             "Top: tissue (purple outside, blue CSF, green grey, yellow white). "
             "Middle: stiffness, cyan = inclusions. Bottom: damping.")
fig.savefig(OUT / "qc_examples.png", dpi=130, bbox_inches="tight")
log(f"Saved qc_examples.png. Done. Everything is in {OUT}")

# Quick summary
for variant in VARIANTS:
    vr = [r for r in rows if r["variant"] == variant]
    print(f"\nVariant {variant}: {len(vr)} brains | "
          + ", ".join(f"{s}: {sum(r['split'] == s for r in vr)}" for s in ("train", "val", "test"))
          + f" | mean CSF fraction {np.mean([r['frac_csf'] for r in vr]):.2f}"
          + f" | inclusions per brain {np.mean([r['n_inclusions'] for r in vr]):.1f}")
plt.show()
