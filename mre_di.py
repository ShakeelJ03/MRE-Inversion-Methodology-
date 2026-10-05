"""
Baseline: direct inversion (DI, also called algebraic Helmholtz inversion)
=========================================================================
Rearranges the wave equation assuming stiffness is locally uniform:
        G* = -rho w^2 u / laplacian(u)
Steps: smooth the noisy waves a little (Gaussian, 1 voxel), take the 5-point
Laplacian where a voxel and its 4 neighbours are all brain, then solve for G*
by least squares over each 3x3 neighbourhood (more stable than voxel by voxel).
DI is undefined where neighbours are missing (next to CSF) -> those voxels are NaN.
"""
import numpy as np
from scipy import ndimage
from mre_solver import RHO, OMEGA


def masked_smooth(u, mask, sigma=1.0):
    """Gaussian smoothing that ignores voxels outside the mask."""
    m = mask.astype(float)
    num = (ndimage.gaussian_filter(u.real * m, sigma)
           + 1j * ndimage.gaussian_filter(u.imag * m, sigma))
    den = ndimage.gaussian_filter(m, sigma)
    return np.where(mask, num / np.maximum(den, 1e-6), 0)


def laplacian(u, mask, h):
    up, mp = np.pad(u, 1), np.pad(mask, 1)
    lap = (up[2:, 1:-1] + up[:-2, 1:-1] + up[1:-1, 2:] + up[1:-1, :-2] - 4 * u) / h ** 2
    ok = mask & mp[2:, 1:-1] & mp[:-2, 1:-1] & mp[1:-1, 2:] & mp[1:-1, :-2]
    return np.where(ok, lap, 0), ok


def direct_inversion(u, mask, voxel_mm=3.0, sigma=1.0):
    """Returns G' (Pa, NaN where undefined) and xi."""
    h = voxel_mm * 1e-3
    us = masked_smooth(u, mask, sigma) if sigma > 0 else u
    L, ok = laplacian(us, mask, h)
    box = lambda a: ndimage.uniform_filter(a, 3, mode="constant") * 9
    prod = us * np.conj(L) * ok
    num = -RHO * OMEGA ** 2 * (box(prod.real) + 1j * box(prod.imag))
    den = box(np.abs(L) ** 2 * ok)
    good = ok & (den > 0)
    Gc = np.where(good, num / np.where(good, den, 1), np.nan)
    Gp = np.clip(Gc.real, 100, 10000)                  # keep within 0.1-10 kPa
    xi = np.clip(Gc.imag / (2 * np.maximum(Gc.real, 1)), 0, 1)
    Gp[~good] = np.nan
    xi[~good] = np.nan
    return Gp, xi