"""
STAGE B - the wave simulator (shared functions)
===============================================
Physics: 2D time-harmonic shear waves polarised out of the slice ("anti-plane").
For this wave type the heterogeneous viscoelastic equation is exact:

        div( G*(x) grad u ) + rho * w^2 * u = 0          G* = G'(1 + 2i xi)

  u    complex displacement (amplitude and phase of the 60 Hz motion)
  G*   complex shear modulus: G' = springiness (stiffness), xi = damping
  rho  density (1000 kg/m^3, like water),  w = 2*pi*60 Hz

Discretisation: each pixel is a little box. Wave "flux" between two neighbouring
boxes = G at their shared face x (difference in u) / h.  Sum of fluxes into a
box + rho w^2 u = 0.  That gives one linear equation per pixel -> a big sparse
linear system A u = b, which SciPy solves directly.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

RHO = 1000.0              # kg/m^3
FREQ = 60.0               # Hz
OMEGA = 2 * np.pi * FREQ  # rad/s


def complex_modulus(Gp, xi):
    """G* = G'(1 + 2i xi).  With this convention waves travel as exp(-ikx)."""
    return Gp * (1 + 2j * xi)


def wavenumber(Gp, xi, freq=FREQ):
    """Complex wavenumber k = w sqrt(rho / G*).  Wavelength = 2 pi / Re(k)."""
    return 2 * np.pi * freq * np.sqrt(RHO / complex_modulus(Gp, xi))


def solve_helmholtz(Gc, domain, u_bc, h_mm, neumann_edges=False, freq=FREQ):
    """Solve div(G* grad u) + rho w^2 u = 0 for u inside `domain`.

    Gc      complex modulus G* (Pa) for every pixel
    domain  True where u is unknown (CSF + brain)
    u_bc    prescribed displacement; used on pixels just outside the domain
            (the skull, which the driver shakes)
    h_mm    pixel size in mm
    neumann_edges  only for tests: array edges become free (no flux) instead of fixed
    Returns u (complex, 0 outside domain) and the relative residual |Au-b|/|b|.
    """
    h = h_mm * 1e-3
    ny, nx = domain.shape
    idx = -np.ones(domain.shape, int)            # unknown number of each domain pixel
    idx[domain] = np.arange(domain.sum())
    n = int(domain.sum())

    py, px = np.nonzero(domain)
    p = idx[py, px]
    Gp = Gc[py, px]
    omega = 2 * np.pi * freq                                   # v3: any frequency (default 60 Hz)
    diag = np.full(n, RHO * omega ** 2 * h ** 2, complex)   # the rho w^2 u term (x h^2)
    rhs = np.zeros(n, complex)
    rows, cols, vals = [], [], []

    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):      # the 4 neighbours
        qy, qx = py + dy, px + dx
        inside = (qy >= 0) & (qy < ny) & (qx >= 0) & (qx < nx)
        qyc, qxc = np.clip(qy, 0, ny - 1), np.clip(qx, 0, nx - 1)
        q_unknown = inside & domain[qyc, qxc]
        q_fixed = inside & ~q_unknown                         # neighbour is skull

        # G at the shared face: harmonic mean keeps the flux right across sharp
        # tissue boundaries (like resistors in series)
        Gq = np.where(q_unknown, Gc[qyc, qxc], Gp)
        Gf = 2 * Gp * Gq / (Gp + Gq)
        # skull faces: the skull sits on the box EDGE, half a pixel away -> 2*G.
        # (Without this the boundary moves when you refine the grid and the
        #  simulation converges slowly.)
        Gf = np.where(q_fixed, 2 * Gp, Gf)
        if neumann_edges:
            Gf = np.where(inside, Gf, 0)

        diag -= Gf
        rows.append(p[q_unknown]); cols.append(idx[qyc, qxc][q_unknown]); vals.append(Gf[q_unknown])
        rhs[q_fixed] -= Gf[q_fixed] * u_bc[qyc, qxc][q_fixed]

    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    x = spla.spsolve(A, rhs)
    residual = np.linalg.norm(A @ x - rhs) / max(np.linalg.norm(rhs), 1e-300)
    u = np.zeros(domain.shape, complex)
    u[domain] = x
    return u, residual


def skull_drive(shape, rng):
    """Displacement on the skull: smooth random amplitude and phase around the
    head (the driver moves the head, and the skull passes it on unevenly)."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    th = np.arctan2(yy - (ny - 1) / 2, xx - (nx - 1) / 2)
    amp, ph = np.ones(shape), np.zeros(shape)
    for m in (1, 2, 3):                    # a few smooth "lumps" around the head
        amp += 0.3 * rng.normal() * np.cos(m * th + rng.uniform(0, 2 * np.pi))
        ph += 1.0 * rng.normal() * np.cos(m * th + rng.uniform(0, 2 * np.pi))
    return np.clip(amp, 0.2, None) * np.exp(1j * ph)


def block_average(a, f, weights=None):
    """Average f x f blocks (optionally weighted). Returns (averaged, weight fraction)."""
    ny, nx = a.shape
    w = np.ones(a.shape) if weights is None else weights
    s = (a * w).reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))
    ws = w.reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))
    return np.where(ws > 0, s / np.maximum(ws, 1e-12), 0), ws / f ** 2


def upsample(a, r):
    """Split every 1 mm pixel into r x r smaller pixels (same values)."""
    return np.kron(a, np.ones((r, r), a.dtype))


def simulate_brain(tissue, Gp, xi, rng, refine=2, voxel_mm=3.0, pixel_mm=1.0, freq=FREQ):
    """Shake one Stage A brain.

    tissue  0 outside, 1 CSF, 2 grey, 3 white (1 mm grid, size multiple of 3)
    refine  solve on a grid `refine` times finer (2 -> 0.5 mm; see the C1 checks)
    Returns a dict with the 1 mm wave field and the 3 mm "acquired" fields.
    """
    r = refine
    t, G, x = upsample(tissue, r), upsample(Gp, r), upsample(xi, r)
    domain = t > 0
    u_fine, res = solve_helmholtz(complex_modulus(G, x), domain, skull_drive(t.shape, rng),
                                  pixel_mm / r, freq=freq)

    dom = domain.astype(float)
    u_1mm, _ = block_average(u_fine, r, dom)                 # for figures

    f = int(round(voxel_mm / pixel_mm)) * r                  # fine pixels per 3 mm voxel
    brain = (t >= 2).astype(float)
    u3, _ = block_average(u_fine, f, dom)                    # what the scanner sees
    G3, brainfrac = block_average(G, f, brain)               # true stiffness per voxel
    xi3, _ = block_average(x, f, brain)
    csffrac = block_average((t == 1).astype(float), f)[0]
    mask3 = brainfrac > 0.5                                  # voxels that are mostly brain
    return dict(u_1mm=u_1mm, u3=u3, G3=G3, xi3=xi3, mask3=mask3,
                brainfrac3=brainfrac, csffrac3=csffrac, residual=res)
