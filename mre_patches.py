"""
Shared functions for Stage C: noise, patches, uniform (HLI) patches.
Used by stage_c1_make_patches.py now, and by the evaluation script later.

A patch = the 7 x 7 voxels (2.1 cm) around one brain voxel, at 3 mm.
Input  : real part, imaginary part, mask   -> 3 x 49 = 147 numbers
Answer : true stiffness G' and damping xi at the centre voxel
"""
import numpy as np
from mre_solver import RHO, OMEGA, complex_modulus

W = 7                 # patch width in voxels
R = W // 2
N_IN = 3 * W * W      # 147 inputs


def add_noise(u3, mask3, snr, rng):
    """Complex Gaussian noise. SNR = median |u| in brain / noise std."""
    sigma = np.median(np.abs(u3[mask3])) / snr
    noise = sigma * (rng.standard_normal(u3.shape) + 1j * rng.standard_normal(u3.shape)) / np.sqrt(2)
    return u3 + noise


def centres(mask3):
    """All brain voxels (row, col) = one patch each."""
    return np.argwhere(mask3)


def extract(u, mask, cen, phase=None):
    """Cut 7x7 patches around each centre.

    - voxels outside the brain mask (CSF, outside head, off the image) are set to 0
      and flagged in the mask channel, so the network can tell 'missing' from 'zero'
    - optional global phase shift per patch (data augmentation / phase cycling)
    - each patch is divided by its largest amplitude, so only the ripple PATTERN
      matters, not how strongly that region was shaken (Murphy 2018)
    Returns X with shape (N, 147), float32.
    """
    up = np.pad(u, R)
    mp = np.pad(mask, R)
    oy, ox = np.mgrid[0:W, 0:W]
    yy = cen[:, 0, None, None] + oy
    xx = cen[:, 1, None, None] + ox
    P = up[yy, xx]
    M = mp[yy, xx].astype(np.float32)
    if phase is not None:
        P = P * np.exp(1j * np.asarray(phase))[:, None, None]
    P = P * M
    scale = np.abs(P).reshape(len(P), -1).max(axis=1) + 1e-12
    P = P / scale[:, None, None]
    n = len(P)
    return np.concatenate([P.real.reshape(n, -1), P.imag.reshape(n, -1),
                           M.reshape(n, -1)], axis=1).astype(np.float32)


def targets(G3, xi3, cen):
    """Answers: log of stiffness in kPa (handles the 1-5 kPa range evenly),
    and damping x 5 (so both outputs have a similar size)."""
    g = G3[cen[:, 0], cen[:, 1]] / 1000.0
    x = xi3[cen[:, 0], cen[:, 1]]
    return np.stack([np.log(g), 5 * x], axis=1).astype(np.float32)


def decode(Y):
    """Network outputs back to G' (Pa) and xi."""
    Y = np.asarray(Y, dtype=float)
    return 1000.0 * np.exp(Y[..., 0]), Y[..., 1] / 5


def uniform_patches(G, xi, masks, snr, rng, factor=3, h_mm=1.0):
    """Murphy-style training patches: UNIFORM tissue (one G', one xi per patch).

    1-4 damped plane waves from random directions, built on a 1 mm grid and
    averaged to 3 mm, with the same masks and noise model as the realistic
    patches, so the only difference from the realistic set is uniformity.
    G, xi, snr: arrays of length N.  masks: (N, 49).
    """
    n = len(G)
    nf = W * factor
    c = (np.arange(nf) - (nf - 1) / 2) * h_mm * 1e-3
    yy, xx = np.meshgrid(c, c, indexing="ij")
    k = OMEGA * np.sqrt(RHO / complex_modulus(G, xi))          # complex wavenumber
    U = np.zeros((n, nf, nf), complex)
    n_waves = rng.integers(1, 5, size=n)
    for j in range(4):
        on = j < n_waves
        th = rng.uniform(0, 2 * np.pi, n)
        amp = rng.uniform(0.3, 1.0, n) * np.exp(1j * rng.uniform(0, 2 * np.pi, n)) * on
        d = np.cos(th)[:, None, None] * xx + np.sin(th)[:, None, None] * yy
        U += amp[:, None, None] * np.exp(-1j * k[:, None, None] * d)
    U3 = U.reshape(n, W, factor, W, factor).mean(axis=(2, 4))
    sig = np.median(np.abs(U3).reshape(n, -1), axis=1) / snr
    U3 = U3 + sig[:, None, None] * (rng.standard_normal(U3.shape)
                                    + 1j * rng.standard_normal(U3.shape)) / np.sqrt(2)
    M = masks.reshape(n, W, W).astype(np.float32)
    P = U3 * M
    scale = np.abs(P).reshape(n, -1).max(axis=1) + 1e-12
    P = P / scale[:, None, None]
    return np.concatenate([P.real.reshape(n, -1), P.imag.reshape(n, -1),
                           M.reshape(n, -1)], axis=1).astype(np.float32)
