"""
Tests for the gauge solver.

The closed-form spot checks are the cheap unit tests with no tolerance to
argue about: along the heading arrival time must be exactly d/(a+c), against
it exactly d/(a-c), and perpendicular exactly d/[b*sqrt(1-c^2/a^2)].
"""

import numpy as np
import pytest

from gauge import (backing_rate, chunk_size_for, flanking_rate, gauge_arrival,
                   head_rate)


def _point(x, y):
    """A one-cell target grid at a single point."""
    return np.array([[float(x)]]), np.array([[float(y)]])


def _tau(uhat, a, b, c, vx, vy):
    """Arrival time at one displacement from a single source at the origin."""
    X, Y = _point(vx, vy)
    return float(gauge_arrival([0.0], [0.0], uhat, a, b, c, X, Y)[0, 0])


# ----------------------------------------------------------------------
# The closed-form spot checks
# ----------------------------------------------------------------------

@pytest.mark.parametrize("LB", [1.2, 2.0, 3.0, 6.0, 10.0])
@pytest.mark.parametrize("bearing", [0.0, 30.0, 137.0, 250.0])
def test_axis_rates_are_exact(LB, bearing):
    head = 0.25
    HB = (LB + np.sqrt(LB ** 2 - 1)) / (LB - np.sqrt(LB ** 2 - 1))
    back = head / HB
    a, c = 0.5 * (head + back), 0.5 * (head - back)
    b = a / LB

    theta = np.deg2rad(bearing)
    u = np.array([np.cos(theta), np.sin(theta)])
    perp = np.array([-u[1], u[0]])
    d = 1000.0

    assert _tau(u, a, b, c, *(d * u)) == pytest.approx(d / head_rate(a, c), rel=1e-12)
    assert _tau(u, a, b, c, *(-d * u)) == pytest.approx(d / backing_rate(a, c), rel=1e-12)
    assert _tau(u, a, b, c, *(d * perp)) == pytest.approx(
        d / flanking_rate(a, b, c), rel=1e-12)
    assert _tau(u, a, b, c, *(-d * perp)) == pytest.approx(
        d / flanking_rate(a, b, c), rel=1e-12)


def test_the_focus_condition_is_self_consistent(wavelet):
    """b = a/LB and b = sqrt(a^2 - c^2) have to agree, or the ellipse is not one."""
    a, b, c = wavelet["a"], wavelet["b"], wavelet["c"]
    assert c == pytest.approx(np.sqrt(a ** 2 - b ** 2), rel=1e-12)
    assert c / a < 1.0                                    # the eccentricity


def test_euclidean_case_is_distance_over_speed():
    for speed in (0.5, 1.0, 7.3):
        assert _tau((1.0, 0.0), speed, speed, 0.0, 3.0, 4.0) == pytest.approx(
            5.0 / speed, rel=1e-12)


def test_gauge_scales_linearly_with_displacement(wavelet):
    """tau(k v) = k tau(v) - which is why minimising over the boundary is exact."""
    base = _tau(wavelet["uhat"], wavelet["a"], wavelet["b"], wavelet["c"], 300.0, 700.0)
    for k in (0.5, 2.0, 10.0):
        assert _tau(wavelet["uhat"], wavelet["a"], wavelet["b"], wavelet["c"],
                    k * 300.0, k * 700.0) == pytest.approx(k * base, rel=1e-12)


def test_heading_is_the_fastest_direction(wavelet):
    """Anisotropy, stated as an ordering: head > flank > back."""
    a, b, c, u = wavelet["a"], wavelet["b"], wavelet["c"], wavelet["uhat"]
    perp = np.array([-u[1], u[0]])
    d = 500.0
    t_head = _tau(u, a, b, c, *(d * u))
    t_flank = _tau(u, a, b, c, *(d * perp))
    t_back = _tau(u, a, b, c, *(-d * u))
    assert t_head < t_flank < t_back
    assert t_back / t_head == pytest.approx(wavelet["HB"], rel=1e-12)


def test_reversing_the_heading_swaps_head_and_back(wavelet):
    """What "the reversed metric" means concretely: the offset is negated."""
    a, b, c, u = wavelet["a"], wavelet["b"], wavelet["c"], wavelet["uhat"]
    d = 500.0
    forward_head = _tau(u, a, b, c, *(d * u))
    reverse_back = _tau(-u, a, b, c, *(d * u))
    assert reverse_back == pytest.approx(d / backing_rate(a, c), rel=1e-12)
    assert forward_head == pytest.approx(d / head_rate(a, c), rel=1e-12)


# ----------------------------------------------------------------------
# Behaviour over a grid
# ----------------------------------------------------------------------

def test_arrival_is_the_minimum_over_all_sources(wavelet):
    X, Y = np.meshgrid(np.linspace(-500, 500, 21), np.linspace(-500, 500, 21))
    sx, sy = np.array([-300.0, 250.0]), np.array([100.0, -220.0])
    both = gauge_arrival(sx, sy, wavelet["uhat"], wavelet["a"], wavelet["b"],
                         wavelet["c"], X, Y)
    first = gauge_arrival(sx[:1], sy[:1], wavelet["uhat"], wavelet["a"],
                          wavelet["b"], wavelet["c"], X, Y)
    second = gauge_arrival(sx[1:], sy[1:], wavelet["uhat"], wavelet["a"],
                           wavelet["b"], wavelet["c"], X, Y)
    assert np.allclose(both, np.minimum(first, second))


def test_source_points_have_zero_arrival_time(wavelet):
    sx, sy = np.array([-300.0, 250.0]), np.array([100.0, -220.0])
    T = gauge_arrival(sx, sy, wavelet["uhat"], wavelet["a"], wavelet["b"],
                      wavelet["c"], sx.reshape(1, -1), sy.reshape(1, -1))
    assert np.allclose(T, 0.0, atol=1e-12)


def test_chunking_does_not_change_the_answer(wavelet):
    X, Y = np.meshgrid(np.linspace(-500, 500, 37), np.linspace(-500, 500, 29))
    sx = np.linspace(-100, 100, 53)
    sy = np.zeros_like(sx)
    args = (sx, sy, wavelet["uhat"], wavelet["a"], wavelet["b"], wavelet["c"], X, Y)
    reference = gauge_arrival(*args, chunk=10_000_000)
    for chunk in (1, 7, 64, 1000):
        assert np.array_equal(gauge_arrival(*args, chunk=chunk), reference)


def test_where_restricts_the_solve_without_changing_it(wavelet):
    X, Y = np.meshgrid(np.linspace(-500, 500, 31), np.linspace(-500, 500, 31))
    sx, sy = np.array([0.0]), np.array([0.0])
    args = (sx, sy, wavelet["uhat"], wavelet["a"], wavelet["b"], wavelet["c"], X, Y)
    full = gauge_arrival(*args)
    where = np.hypot(X, Y) < 300.0
    partial = gauge_arrival(*args, where=where)
    assert np.isnan(partial[~where]).all()
    assert np.array_equal(partial[where], full[where])


def test_empty_where_returns_all_nan(wavelet):
    X, Y = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 10, 5))
    T = gauge_arrival([0.0], [0.0], wavelet["uhat"], wavelet["a"], wavelet["b"],
                      wavelet["c"], X, Y, where=np.zeros(X.shape, bool))
    assert np.isnan(T).all()


def test_chunk_size_shrinks_as_vertices_grow():
    assert chunk_size_for(100) > chunk_size_for(10_000)
    assert chunk_size_for(1200, memory_budget_mb=64) * 1200 * 48 <= 64 * 1024 ** 2
    assert chunk_size_for(10 ** 9) >= 1


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------

def test_bad_inputs_are_rejected(wavelet):
    X, Y = _point(1.0, 1.0)
    a, b, c, u = wavelet["a"], wavelet["b"], wavelet["c"], wavelet["uhat"]

    with pytest.raises(ValueError, match="no source vertices"):
        gauge_arrival([], [], u, a, b, c, X, Y)
    with pytest.raises(ValueError, match="source arrays disagree"):
        gauge_arrival([0.0, 1.0], [0.0], u, a, b, c, X, Y)
    with pytest.raises(ValueError, match="X and Y disagree"):
        gauge_arrival([0.0], [0.0], u, a, b, c, X, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="wavelet axes must be positive"):
        gauge_arrival([0.0], [0.0], u, 0.0, b, c, X, Y)
    with pytest.raises(ValueError, match="heading must be a nonzero"):
        gauge_arrival([0.0], [0.0], (0.0, 0.0), a, b, c, X, Y)
    with pytest.raises(ValueError, match="where shape"):
        gauge_arrival([0.0], [0.0], u, a, b, c, X, Y, where=np.zeros((3, 3), bool))


def test_c_at_or_above_a_is_rejected_as_a_corrupted_parameterisation():
    """
    Unreachable for any finite LB, because c/a is the eccentricity. The guard
    exists to catch a broken parameterisation, not to mark a wind limit.
    """
    X, Y = _point(1.0, 1.0)
    with pytest.raises(ValueError, match="backing rate is non-positive"):
        gauge_arrival([0.0], [0.0], (1.0, 0.0), 1.0, 0.5, 1.0, X, Y)


def test_heading_need_not_arrive_normalised(wavelet):
    a, b, c = wavelet["a"], wavelet["b"], wavelet["c"]
    unit = _tau((1.0, 0.0), a, b, c, 100.0, 50.0)
    scaled = _tau((17.0, 0.0), a, b, c, 100.0, 50.0)
    assert scaled == pytest.approx(unit, rel=1e-12)
