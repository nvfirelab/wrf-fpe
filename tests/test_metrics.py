"""Tests for the metric knob: the Euclidean metric and its two backends."""

import numpy as np
import pytest

from metrics import EuclideanMetric, Source


def test_source_normalises_its_inputs():
    source = Source(x=[1, 2, 3], y=[4, 5, 6])
    assert source.x.dtype == float and source.n_vertices == 3
    assert source.direction == "outward"
    with pytest.raises(ValueError, match="unknown direction"):
        Source(x=[0], y=[0], direction="sideways")


def test_euclidean_rejects_nonsense():
    with pytest.raises(ValueError, match="speed must be positive"):
        EuclideanMetric(speed=0.0)
    with pytest.raises(ValueError, match="unknown backend"):
        EuclideanMetric(backend="magic")


def test_euclidean_is_its_own_reverse():
    """Euclidean distance is symmetric, so the inward solve is the same metric."""
    metric = EuclideanMetric()
    assert metric.reversed() is metric
    assert not metric.is_anisotropic


def test_gauge_backend_gives_distance_over_speed(pair):
    initial, _, grid = pair
    source = Source(*initial.boundary_vertices(grid), mask=initial.mask(grid))
    slow = EuclideanMetric(speed=1.0).arrival(source, grid)
    fast = EuclideanMetric(speed=4.0).arrival(source, grid)
    assert np.allclose(fast, slow / 4.0)


def test_backends_agree_away_from_the_boundary(pair):
    """
    The distance transform quantises to whole cells near the boundary, where
    the gauge measures the true distance. Away from it the two converge.
    """
    initial, _, grid = pair
    source = Source(*initial.boundary_vertices(grid), mask=initial.mask(grid),
                    direction="outward")
    gauge = EuclideanMetric(backend="gauge").arrival(source, grid)
    edt = EuclideanMetric(backend="edt").arrival(source, grid)

    outside = ~initial.mask(grid)
    far = outside & (edt > 10 * grid.cell_size)
    assert np.abs(gauge[far] - edt[far]).max() < grid.cell_size
    # The gauge is the more accurate of the two: it is never the one that
    # reports a whole cell where the true distance is a fraction of one.
    near = outside & (edt <= grid.cell_size)
    assert gauge[near].min() < 0.5 * grid.cell_size


def test_edt_backend_needs_a_mask(pair):
    initial, _, grid = pair
    source = Source(*initial.boundary_vertices(grid))
    with pytest.raises(ValueError, match="needs the source's rasterised mask"):
        EuclideanMetric(backend="edt").arrival(source, grid)


def test_edt_backend_respects_the_direction(pair):
    """Outward is zero behind the front; inward is zero ahead of it."""
    _, final, grid = pair
    mask = final.mask(grid)
    metric = EuclideanMetric(backend="edt")
    outward = metric.arrival(Source(*final.boundary_vertices(grid), mask=mask,
                                    direction="outward"), grid)
    inward = metric.arrival(Source(*final.boundary_vertices(grid), mask=mask,
                                   direction="inward"), grid)
    assert (outward[mask] == 0.0).all() and (outward[~mask] > 0.0).all()
    assert (inward[~mask] == 0.0).all() and (inward[mask] > 0.0).all()


def test_where_masks_both_backends(pair):
    initial, final, grid = pair
    where = final.mask(grid)
    source = Source(*initial.boundary_vertices(grid), mask=initial.mask(grid))
    for backend in ("gauge", "edt"):
        T = EuclideanMetric(backend=backend).arrival(source, grid, where=where)
        assert np.isnan(T[~where]).all()
        assert np.isfinite(T[where]).all()


def test_describe_reports_the_backend():
    assert "edt" in EuclideanMetric(backend="edt").describe()
    assert "gauge" in EuclideanMetric().describe()
