# MRE inversion with synthetic ground truth: a 2D pilot

Neural-network stiffness inversion for brain MR elastography, trained and evaluated on synthetic brains with known stiffness, together with a physics-based reliability map and weak-form SINDy system identification.

![One synthetic brain: true stiffness, simulated wave, amplitude, and what the scanner sees at 3 mm](figures/b_one_brain_wave.png)

*Figure: one synthetic brain, from true stiffness to the simulated 60 Hz shear wave, its amplitude, and the 3 mm data the inversion works on.*

## Contents

- [Summary](#summary)
- [Motivation](#motivation)
- [Pipeline](#pipeline)
- [Repository layout](#repository-layout)
- [Results](#results)
- [Limitations and next steps](#limitations-and-next-steps)
- [Getting started](#getting-started)
- [References](#references)

---

## Summary

- A multilayer perceptron trained on realistic brain patches reaches 6.8 % median stiffness error at SNR 20, against 10.3 % for direct inversion, and recovers 35 % of inclusion contrast against 22 %.
- A reliability map combining a noise- and grid-debiased wave-equation check with anatomy and ensemble spread ranks badly wrong voxels with AUC 0.72 at SNR 20 (near-CSF baseline 0.61). Retaining the 10 % most reliable voxels reduces the fraction of voxels with more than 20 % error from 10.8 % to 1.7 %.
- Weak-form SINDy identifies where the homogeneous-tissue assumption holds and estimates stiffness from the physics alone with 7.7 % median error at SNR 20, compared with 21.7 % for the strong form.

All methods were developed on validation brains and applied once to 75 held-out test brains; uncertainty is reported as paired bootstrap 95 % intervals over brains.

---

## Motivation

MR elastography (MRE) vibrates the head, images the shear waves travelling through the brain, and converts the wave pattern into a stiffness map. This step is called inversion. Brain stiffness changes with ageing and neurodegeneration, which makes it a promising biomarker.

Standard inversion has two problems:

1. It is unreliable near CSF and tissue boundaries, because it assumes the tissue is uniform around each voxel.
2. It gives no warning when it is wrong, and real scans have no ground truth to check against.

This pilot uses synthetic brains with known stiffness to:

1. train a neural network to do the inversion (following Murphy et al. 2018 and Scott et al. 2020),
2. measure exactly where and why it fails,
3. build a reliability map from wave physics that flags those failures,
4. identify from the data alone which wave equation holds in each region (SINDy; Brunton et al. 2016, Messenger & Bortz 2021).

---

## Pipeline

```
 SynthSeg templates
        │
        ▼
 ┌─────────────┐   500 synthetic brains: tissue → stiffness G′, damping ξ, random inclusions
 │ A  Brains   │   split by slice level: 350 train / 75 validation / 75 test
 └──────┬──────┘
        ▼
 ┌─────────────┐   2D anti-plane shear, conservative finite differences,
 │ B  Waves    │   solved at 0.5 mm, block-averaged to 3 mm (40 / 60 / 80 Hz)
 └──────┬──────┘   checkpoint C1: matches theory, 2nd-order convergence
        ▼
 ┌─────────────┐   7×7 patches, noise at SNR 5–50
 │ C  Network  │   MLP ensembles (ILI realistic patches / HLI uniform patches)
 └──────┬──────┘
        ▼
 ┌─────────────┐   E2 error vs noise · E3 error vs distance to CSF
 │ D  Evaluate │   inclusion contrast · baseline: direct inversion
 └──────┬──────┘
        ▼
 ┌─────────────┐   wave-equation check (noise- and grid-debiased)
 │ E5  Check   │   + anatomy + ensemble spread → probability a voxel is wrong
 └──────┬──────┘
        ▼
 ┌─────────────┐   which equation fits each 15 mm window?
 │ E6  SINDy   │   uniform · varying stiffness · extra term · both
 └─────────────┘   strong form (E6) → weak form (E6b)
```

![Synthetic brains: tissue labels, stiffness with inclusions, damping](figures/a_qc_brains.png)

*Synthetic brains: tissue labels (top), true stiffness with inclusions outlined in cyan (middle), damping (bottom).*

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── figures/                         result figures used in this README
│
├── mre_solver.py                    wave simulator (shared)
├── mre_patches.py                   noise model, patch extraction (shared)
├── mre_patches_mf.py                same, for 40 / 60 / 80 Hz (shared)
├── mre_di.py                        direct-inversion baseline (shared)
│
├── stage_a_make_brains.py           A    synthetic brains
├── b1_checks.py                     B    checkpoint C1
├── stage_b2_simulate.py             B    60 Hz waves
├── stage_b3_simulate_multifreq.py   v3   40 and 80 Hz waves
├── stage_c1_make_patches.py         C    training patches
├── stage_c2_train.py                C    network training
├── stage_c3_patches_train_mf.py     v3   multi-frequency patches + network
├── stage_d1_evaluate.py             D    E2 / E3 / inclusion contrast
├── stage_d2_evaluate_mf.py          v3   multi-frequency evaluation
├── stage_e1_trust_map.py            E5   reliability map v1
├── stage_f1_predict_val.py          v2   validation predictions
├── stage_f2_trust_v2.py             v2   reliability map v2
├── stage_f3_trust_v21.py            v2.1 debiased reliability map (dev / final)
├── stage_f4_bootstrap.py            v2.1 bootstrap over test brains
├── stage_f5_trust_mf.py             v3   three-frequency reliability map
├── stage_e6_model_selection.py      E6   strong-form SINDy
└── stage_e6b_weak_sindy.py          E6b  weak-form SINDy
```

Generated data (brains, waves, patches, models, predictions; several GB) is written to `mre_project/` and is not tracked. It can be rebuilt by running the scripts in order.

<details>
<summary>Script descriptions</summary>

| Stage | Script | Purpose |
| --- | --- | --- |
| A | `stage_a_make_brains.py` | 500 brains from SynthSeg templates; one connected head with a 2 mm CSF rim; stiffness and damping by tissue, plus random inclusions |
| B | `b1_checks.py` | Checkpoint C1: damped plane wave vs theory, grid convergence, DI bias at 3 mm |
| B | `stage_b2_simulate.py` | 60 Hz waves for all brains (restartable, parallel) |
| C | `stage_c1_make_patches.py` | 500k train / 80k validation patches with random phase and noise at SNR 5–50 |
| C | `stage_c2_train.py` | MLP 3 × 128, Adam, early stopping; ensembles of 5 (ILI) and 2 (HLI) |
| D | `stage_d1_evaluate.py` | E2, E3, inclusion contrast transfer; saves predictions |
| E5 v1 | `stage_e1_trust_map.py` | Physics residual + ensemble spread |
| v2 | `stage_f1_predict_val.py` | Predictions on validation brains, for development |
| v2 | `stage_f2_trust_v2.py` | Dev/final protocol, noise floor, fair baselines |
| v2.1 | `stage_f3_trust_v21.py` | Method-of-moments noise debiasing, 3 mm grid-bias correction, region model. `python stage_f3_trust_v21.py final` runs on test |
| v2.1 | `stage_f4_bootstrap.py` | Paired bootstrap over test brains (1,000 redraws) |
| v3 | `stage_b3_simulate_multifreq.py` | Same brains and drive at 40 and 80 Hz |
| v3 | `stage_c3_patches_train_mf.py` | 343-input patches + network. Modes: `patches`, `quick`, `train` |
| v3 | `stage_d2_evaluate_mf.py` | Multi- vs single-frequency network on identical 60 Hz noisy data |
| v3 | `stage_f5_trust_mf.py` | Physics pooled over three frequencies (weighted method of moments) |
| E6 | `stage_e6_model_selection.py` | Strong-form SINDy with STLSQ pruning + significance test |
| E6b | `stage_e6b_weak_sindy.py` | Weak-form SINDy with discrete-consistent test functions |

</details>

---

## Results

All numbers are on the 75 held-out test brains.

### Simulator validation

Wavelength error is below 0.6 % with second-order grid convergence. Direct inversion is exact on a fine grid but biased by +10 % (1 kPa) to +2 % (5 kPa) at the 3 mm scanner resolution, which the later physics checks correct for.

![Checkpoint C1](figures/b_c1_checks.png)

### Network inversion versus direct inversion

| SNR 20 | Direct inversion | HLI network | **ILI network** |
| --- | :---: | :---: | :---: |
| Median error | 10.3 % | 8.6 % | **6.8 %** |
| Correlation with truth | 0.50 | 0.77 | **0.84** |
| Inclusion contrast recovered | 22 % | 23 % | **35 %** |

Within 4 mm of CSF (28 % of voxels), direct inversion is undefined; the ILI network's error there is 10.2 %. Adding 40 and 80 Hz lowers the error at every noise level and every distance from CSF (7.6 % → 7.2 % over all voxels at SNR 20).

![Single- vs multi-frequency network](figures/g_mf_vs_v1.png)

### Reliability map

How well each method ranks badly wrong voxels (> 20 % error) above good ones (AUC; 0.5 = guessing):

| Version | SNR 5 | SNR 20 | SNR 50 |
| --- | :---: | :---: | :---: |
| Near-CSF rule (baseline) | 0.57 | 0.61 | 0.61 |
| v1: residual + spread | — | 0.60 | — |
| v2.1: debiased 60 Hz physics | 0.66 | 0.70 | 0.74 |
| **v3: three-frequency physics** | **0.69** | **0.72** | **0.76** |

The physics gain is significant (bootstrap 95 % interval excludes zero) and grows as noise falls. At SNR 5, 60 Hz physics alone adds nothing, but three frequencies add +0.020 [0.009, 0.032]. Region-level maps (12 mm tiles) reach AUC 0.79–0.82.

![v2.1 reliability map](figures/f3_v21_final.png)
![v3 reliability map](figures/f5_mf_final.png)

### System identification

Candidate equations per 15 mm window: **M0** uniform tissue (the direct-inversion assumption), **M1** stiffness varies, **M3** extra term (identifiable only with several frequencies), **M1+3** both. Selection uses SINDy-style thresholding plus a significance test. Plain BIC was tried first and failed on clean data.

| Data | Finds true heterogeneity (AUC)<br>strong → weak | Physics-only stiffness error<br>strong → weak |
| --- | :---: | :---: |
| Clean | 0.86 → **0.89** | 9.9 % → **5.4 %** |
| SNR 50 | 0.73 → **0.82** | 10.8 % → **6.0 %** |
| SNR 20 | 0.54 → **0.68** | 21.7 % → **7.7 %** |
| SNR 10 | 0.47 → **0.55** | 84.9 % → **11.6 %** |

The strong form needs second derivatives of noisy data and breaks down. The weak form moves the derivatives onto smooth test functions and stays usable down to SNR 10. Where uniform tissue (M0) is selected, the network is badly wrong in only 4.1 % of voxels; where nothing fits, 14.7 %.

![Strong vs weak SINDy](figures/e6b_strong_vs_weak.png)
![Model selection maps](figures/e6_maps.png)
*Light green = uniform (M0) · blue = varying stiffness (M1) · orange = extra term (M3) · purple = both · grey = nothing fits.*

---

## Limitations and next steps

| Limitation | Next step |
| --- | --- |
| 2D, synthetic data only | 3D simulation; real MRE data |
| CSF simulated as a soft solid, so CSF "missing physics" cannot be detected | Fluid or poroelastic CSF model |
| Physics gains are real but small (+0.01 to +0.03 AUC) | Feed weak-SINDy outputs into the reliability map |
| Physics flags errors but cannot yet correct the network (r ≈ 0.05) | Weak-form residual as a training loss |
| Inclusion contrast limited to ~35 % by the 7×7 patch | Larger receptive field (U-Net) |
| Strong and weak SINDy scored on different voxel sets | Re-score on a common set |
| Bootstrap covers test-brain choice, not network retraining | Retrain with several seeds |

---

## Getting started

Python 3.10+, CPU only.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Place the SynthSeg CN templates in `example/` (not included), then run the scripts in the order of the table above, starting with `stage_a_make_brains.py`.

---

## References

### MRE and inversion

1. Hiscox LV et al. Magnetic resonance elastography (MRE) of the human brain: technique, findings and clinical applications. *Phys Med Biol* 2016;61(24):R401. [doi:10.1088/0031-9155/61/24/R401](https://doi.org/10.1088/0031-9155/61/24/R401)
2. Hiscox LV et al. Standard-space atlas of the viscoelastic properties of the human brain (MRE134). *Hum Brain Mapp* 2020;41(18):5282–5300. [doi:10.1002/hbm.25192](https://doi.org/10.1002/hbm.25192)
3. Papazoglou S et al. Multifrequency inversion in magnetic resonance elastography. *Phys Med Biol* 2012;57(8):2329. [doi:10.1088/0031-9155/57/8/2329](https://doi.org/10.1088/0031-9155/57/8/2329)
4. Meyer T et al. Comparison of inversion methods in MR elastography: an open-access pipeline (BIOQIC). *Magn Reson Med* 2022;88(4):1840–1850. [doi:10.1002/mrm.29320](https://doi.org/10.1002/mrm.29320)
5. Lilaj L et al. Inversion-recovery MR elastography of the human brain for improved stiffness quantification near fluid–solid boundaries. *Magn Reson Med* 2021;86(5):2552–2561. [doi:10.1002/mrm.28898](https://doi.org/10.1002/mrm.28898)
6. McGarry MDJ et al. A heterogenous, time harmonic, nearly incompressible transverse isotropic finite element brain simulation platform for MR elastography (NITI). *Phys Med Biol* 2020. [doi:10.1088/1361-6560/ab9a84](https://doi.org/10.1088/1361-6560/ab9a84)
7. Palme H, Moreno R. MRE-FIn: open-source finite element framework for inversion in MRE. Accepted, *ISMRM* 2026.

### Neural-network inversion

8. Murphy MC et al. Artificial neural networks for stiffness estimation in magnetic resonance elastography. *Magn Reson Med* 2018;80(1):351–360. [doi:10.1002/mrm.27019](https://doi.org/10.1002/mrm.27019)
9. Scott JM et al. Artificial neural networks for magnetic resonance elastography stiffness estimation in inhomogeneous materials. *Med Image Anal* 2020;63:101710. [doi:10.1016/j.media.2020.101710](https://doi.org/10.1016/j.media.2020.101710)
10. Scott JM et al. Impact of material homogeneity assumption on cortical stiffness estimates by MR elastography. *Magn Reson Med* 2022;88:916–929. [doi:10.1002/mrm.29226](https://doi.org/10.1002/mrm.29226)
11. Ragoza M, Batmanghelich K. Physics-informed neural networks for tissue elasticity reconstruction in magnetic resonance elastography. *MICCAI* 2023, LNCS 14229:333–343. [doi:10.1007/978-3-031-43999-5_32](https://doi.org/10.1007/978-3-031-43999-5_32) · [code](https://github.com/batmanlab/MRE-PINN)
12. Bustin H et al. ElastoNet: neural network-based multicomponent MR elastography wave inversion with uncertainty quantification. *Med Image Anal* 2025;105:103642. [doi:10.1016/j.media.2025.103642](https://doi.org/10.1016/j.media.2025.103642)
13. Iftikhar H, Ahmad R, Kolipaka A. Deep learning-driven inversion framework for shear modulus estimation in MRE (DIME). [arXiv:2512.13010](https://arxiv.org/abs/2512.13010)

### System identification

14. Brunton SL, Proctor JL, Kutz JN. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS* 2016;113(15):3932–3937. [doi:10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113)
15. Rudy SH, Brunton SL, Proctor JL, Kutz JN. Data-driven discovery of partial differential equations. *Sci Adv* 2017;3(4):e1602614. [doi:10.1126/sciadv.1602614](https://doi.org/10.1126/sciadv.1602614)
16. Messenger DA, Bortz DM. Weak SINDy for partial differential equations. *J Comput Phys* 2021;443:110525. [doi:10.1016/j.jcp.2021.110525](https://doi.org/10.1016/j.jcp.2021.110525)
17. Mangan NM, Kutz JN, Brunton SL, Proctor JL. Model selection for dynamical systems via sparse regression and information criteria. *Proc R Soc A* 2017;473:20170009. [doi:10.1098/rspa.2017.0009](https://doi.org/10.1098/rspa.2017.0009)
18. Nikolov DP et al. Ogden material calibration via magnetic resonance cartography, parameter sensitivity, and variational system identification. [arXiv:2204.03122](https://arxiv.org/abs/2204.03122)
19. Möller H. HetSI: heterogeneous system identification for the analysis of MRE data. KTH Biomedical Imaging seminar, 20 October 2025.

### Data and statistics

20. Billot B et al. SynthSeg: segmentation of brain MRI scans of any contrast and resolution without retraining. *Med Image Anal* 2023;86:102789. [doi:10.1016/j.media.2023.102789](https://doi.org/10.1016/j.media.2023.102789)
21. Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman & Hall/CRC, 1993.

