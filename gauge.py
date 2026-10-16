"""
Module holding the gauge solver: arrival time under a homogeneous metric.

The gauge (or Minkowski functional) of the wavelet answers "here is a
displacement v, how long until the fire covers it". Arrival time from a source
is then a minimum of that over the source boundary, which is a direct
computation rather than a differential equation. That single choice is what
removes the PDE solver from the method.

The condition that x is reached from y by time t is that v = x - y lies inside
the wavelet scaled by t:

    | A^-1 ( v/t - c*u ) | <= 1

Rotating v into the u frame and stretching by A^-1 gives p; with q = (c/a, 0)
the condition becomes |p/t - q| <= 1, a quadratic in t whose smallest
admissible root is explicit:

    tau(v) = [ -p.q + sqrt( (p.q)^2 + (1 - |q|^2)*|p|^2 ) ] / ( 1 - |q|^2 )

Minimising over the source *boundary* rather than the whole source region is
exact: tau scales linearly, so for any point outside the source the closest
point in the metric always lies on the boundary.
"""

import numpy as np

# Working arrays held per chunk: vx, vy, px, py, pq, and the sqrt result.
_ARRAYS_PER_CHUNK = 6
_BYTES_PER_VALUE = 8


def chunk_size_for(n_sources, memory_budget_mb=64.0):
    """
    Largest chunk of target cells whose intermediate arrays fit a memory budget.

    Cost is cells x source vertices and the intermediate array is
    chunk x vertices, so an unchunked 501^2 grid against 1200 vertices will
    exhaust memory. This picks the chunk from the vertex count instead of
    leaving a fixed number that silently stops fitting.

    Parameters:
    -----------
    n_sources : int
        Number of source vertices
    memory_budget_mb : float
        Budget for the intermediate arrays of one chunk, in megabytes

    Returns:
    --------
    chunk : int
        Number of target cells to process at once, at least 1
    """
    per_cell = _ARRAYS_PER_CHUNK * _BYTES_PER_VALUE * max(int(n_sources), 1)
    return max(int(memory_budget_mb * 1024 * 1024 / per_cell), 1)


def gauge_arrival(sx, sy, uhat, a, b, c, X, Y, where=None, chunk=None,
                  memory_budget_mb=64.0):
    """
    Exact arrival time under a homogeneous elliptical metric.

    Parameters:
    -----------
    sx, sy : numpy.ndarray
        1D arrays of source vertex coordinates in metres
    uhat : tuple or numpy.ndarray
        Unit heading (ux, uy) of the wavelet
    a, b, c : float
        Semi-major axis, semi-minor axis, and offset of the ignition point
        behind the ellipse centre, all in metres per second. c = 0 with a = b
        gives an isotropic metric of speed a.
    X, Y : numpy.ndarray
        Target cell-centre coordinates, any matching shape
    where : numpy.ndarray, optional
        Boolean array selecting the cells to solve for. Cells outside it come
        back NaN. Restricting the solve to the cells that matter is exact, not
        an approximation, and is the cheapest speed-up available.
    chunk : int, optional
        Target cells per chunk. Defaults to whatever fits memory_budget_mb.
    memory_budget_mb : float
        Memory budget per chunk when `chunk` is not given

    Returns:
    --------
    T : numpy.ndarray
        Arrival time in seconds, shape of X. NaN where `where` is False.
    """
    sx = np.ascontiguousarray(sx, dtype=float).ravel()
    sy = np.ascontiguousarray(sy, dtype=float).ravel()
    if sx.size == 0:
        raise ValueError("no source vertices to spread from")
    if sx.size != sy.size:
        raise ValueError(f"source arrays disagree: {sx.size} x, {sy.size} y")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y disagree: {X.shape} vs {Y.shape}")
    if a <= 0 or b <= 0:
        raise ValueError(f"wavelet axes must be positive; got a={a}, b={b}")

    ux, uy = np.asarray(uhat, dtype=float).ravel()
    norm = np.hypot(ux, uy)
    if norm <= 0:
        raise ValueError("heading must be a nonzero vector")
    ux, uy = ux / norm, uy / norm

    qx = c / a
    den = 1.0 - qx * qx
    if den <= 0:
        # c/a is the eccentricity, so this is unreachable for any finite LB.
        # It is a guard against a corrupted parameterisation, not a physical
        # limit on wind speed.
        raise ValueError(f"c >= a (c={c}, a={a}): backing rate is non-positive")

    gx, gy = X.ravel(), Y.ravel()
    T = np.full(gx.size, np.nan)

    if where is None:
        targets = np.arange(gx.size)
    else:
        where = np.asarray(where, dtype=bool)
        if where.shape != X.shape:
            raise ValueError(f"where shape {where.shape} does not match {X.shape}")
        targets = np.nonzero(where.ravel())[0]
        if targets.size == 0:
            return T.reshape(X.shape)

    if chunk is None:
        chunk = chunk_size_for(sx.size, memory_budget_mb)

    for i0 in range(0, targets.size, chunk):
        idx = targets[i0:i0 + chunk]
        vx = gx[idx, None] - sx[None, :]
        vy = gy[idx, None] - sy[None, :]
        px = (vx * ux + vy * uy) / a          # rotate into the u frame,
        py = (-vx * uy + vy * ux) / b         #   then stretch by A^-1
        pq = px * qx
        np.square(px, out=px)
        np.square(py, out=py)
        px += py                              # |p|^2
        T[idx] = ((-pq + np.sqrt(pq * pq + den * px)) / den).min(axis=1)

    return T.reshape(X.shape)


def head_rate(a, c):
    """Spread rate along the heading: the fastest direction."""
    return a + c


def backing_rate(a, c):
    """Spread rate against the heading: the slowest direction."""
    return a - c


def flanking_rate(a, b, c):
    """
    Spread rate perpendicular to the heading, as the gauge measures it.

    Not the semi-minor axis: b is the wavelet's half-width, but a displacement
    perpendicular to the heading *from the ignition point* leaves the ellipse
    at a point offset from the widest one, so the transit rate is
    b*sqrt(1 - c^2/a^2) rather than b.
    """
    return b * np.sqrt(1.0 - (c / a) ** 2)
