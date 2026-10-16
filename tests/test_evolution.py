"""
Tests for GaugeEvolution, including the Euclidean regression.

The regression is the free test the unification claim buys: run the new code
with a Euclidean metric and it must reproduce the linear prototype. It proves
the claim and catches almost every plumbing bug.
"""

import numpy as np
import pytest

from clock import TwoSidedClock
from evolution import GaugeEvolution
from grid import GridSpec
from metrics import EuclideanMetric
from perimeter_evolution import PerimeterEvolution


def jaccard(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def prototype_progress(initial, final, grid):
    """
    The linear prototype's progress field, written out algebraically.

    Its zero level set sits where (1-s)*phi_i + s*phi_f = 0, which rearranges
    to s = phi_i / (phi_i - phi_f). In the region between the perimeters
    phi_i is the distance to the initial perimeter and -phi_f the distance to
    the final one, so this is already a two-sided distance blend.
    """
    phi_i, phi_f = initial.sdf(grid), final.sdf(grid)
    band = (phi_i > 0) & (phi_f < 0)
    return phi_i / (phi_i - phi_f), band


# ----------------------------------------------------------------------
# The Euclidean regression
# ----------------------------------------------------------------------

def test_edt_backend_reproduces_the_prototype_exactly(pair):
    """
    The linear prototype is a two-sided Euclidean distance blend in disguise.

    Computing the distances the way the prototype does - off the same signed
    distance functions - the agreement is not approximate. It is the same
    arithmetic.
    """
    initial, final, grid = pair
    s_proto, band = prototype_progress(initial, final, grid)
    evolution = GaugeEvolution(initial, final, grid,
                               metric=EuclideanMetric(backend="edt"))
    assert np.abs(evolution.progress[band] - s_proto[band]).max() < 1e-12


def test_gauge_backend_matches_the_prototype_to_the_discretisation_floor(pair):
    initial, final, grid = pair
    s_proto, band = prototype_progress(initial, final, grid)
    evolution = GaugeEvolution(initial, final, grid,
                               metric=EuclideanMetric(backend="gauge"))
    difference = np.abs(evolution.progress[band] - s_proto[band])
    assert np.median(difference) < 1e-2
    assert np.percentile(difference, 99) < 1e-1


def test_the_difference_from_the_prototype_converges_under_refinement(pair):
    """
    What separates a discretisation floor from a bug: refine the grid and the
    disagreement has to shrink. It does not converge to the prototype because
    the prototype is right - it converges because both are approaching the
    same continuous answer, and the prototype's distance transform is the one
    quantised to whole cells near the boundary.
    """
    initial, final, grid = pair
    errors = []
    for cell in (50.0, 25.0, 12.5):
        fine = GridSpec.from_bounds(grid.bounds, cell_size=cell, crs=grid.crs)
        s_proto, band = prototype_progress(initial, final, fine)
        evolution = GaugeEvolution(initial, final, fine,
                                   metric=EuclideanMetric(backend="gauge"))
        errors.append(np.median(np.abs(evolution.progress[band] - s_proto[band])))
    assert errors[1] < errors[0] and errors[2] < errors[1]
    assert errors[2] < 0.5 * errors[0]


@pytest.mark.parametrize("backend,floor", [("edt", 0.99), ("gauge", 0.98)])
def test_frames_match_the_prototype_class(pair, backend, floor):
    """Against the prototype itself, not a restatement of its algebra."""
    initial, final, grid = pair
    X, Y = grid.meshgrid()
    prototype = PerimeterEvolution(initial.sdf(grid), final.sdf(grid), X, Y,
                                   T=1.0, num_steps=20)
    evolution = GaugeEvolution(initial, final, grid,
                               metric=EuclideanMetric(backend=backend))
    for level in (0.25, 0.5, 0.75):
        burned = ((1 - level) * prototype.phi_inner + level * prototype.phi_outer) < 0
        ours = (evolution.progress <= level) & evolution.mask_end
        assert jaccard(burned, ours) > floor


# ----------------------------------------------------------------------
# The constraints
# ----------------------------------------------------------------------

def test_endpoints_are_exact(pair):
    """C2: the evolution begins on the initial perimeter and ends on the final."""
    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid)
    frames = evolution.frames(num_steps=20)
    assert np.array_equal(frames[0], evolution.mask_start & evolution.mask_end)
    assert np.array_equal(frames[-1], evolution.mask_end)
    start_j, end_j = evolution.endpoint_jaccard()
    assert start_j == 1.0 and end_j == 1.0


def test_progress_is_continuous_and_bounded(pair):
    """C3 and C7: no lookup, nothing to jump across, and progress on [0, 1]."""
    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid)
    inside = evolution.progress[evolution.mask_end]
    assert np.isfinite(inside).all()
    assert inside.min() == 0.0
    assert inside.max() <= 1.0
    assert np.isnan(evolution.progress[~evolution.mask_end]).all()


def test_detached_components_receive_finite_arrival_times(pair):
    """
    C5: a gauge distance is defined between any two points regardless of
    connectivity, so a spot fire is reached with no special handling. It
    burns late, because it is far away.
    """
    from scipy.ndimage import label

    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid)
    labels, n_components = label(evolution.mask_end)
    assert n_components == 2

    main = evolution.progress[labels == 1]
    spot = evolution.progress[labels == 2]
    assert np.isfinite(spot).all()
    assert np.nanmin(spot) > np.nanmedian(main)


def test_burned_area_only_grows(pair):
    """C7: each frame contains the one before it, and nothing needed fixing."""
    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid)
    frames = evolution.frames(num_steps=20)
    for earlier, later in zip(frames[:-1], frames[1:]):
        assert not (earlier & ~later).any()
        assert later.sum() >= earlier.sum()
    assert evolution.monotonicity_violations == 0
    assert frames[-1].sum() > frames[0].sum()


# ----------------------------------------------------------------------
# Mechanics
# ----------------------------------------------------------------------

def test_restricting_the_solve_is_exact_not_approximate(pair):
    """Cells outside the final perimeter never ignite, so skipping them is free."""
    initial, final, grid = pair
    restricted = GaugeEvolution(initial, final, grid, restrict_to_final=True)
    full = GaugeEvolution(initial, final, grid, restrict_to_final=False)
    inside = restricted.mask_end
    assert np.allclose(restricted.T_f[inside], full.T_f[inside])
    assert np.allclose(restricted.progress[inside], full.progress[inside])
    assert restricted.summary()["cells_solved"] < grid.size


def test_the_speed_cancels_out_of_the_progress_field(pair):
    """
    A two-sided ratio of travel times is scale-free, so a Euclidean metric's
    speed changes the seconds but not the progress. That is why the default
    of 1 m/s is harmless.
    """
    initial, final, grid = pair
    slow = GaugeEvolution(initial, final, grid, metric=EuclideanMetric(speed=1.0))
    fast = GaugeEvolution(initial, final, grid, metric=EuclideanMetric(speed=9.0))
    inside = slow.mask_end
    assert np.allclose(slow.progress[inside], fast.progress[inside])


def test_arrival_seconds_spans_the_observed_interval(pair):
    """
    s reaches 1 only where T_b is exactly zero, which is on a source vertex.
    No cell centre lands on one, so the latest cell ignites a fraction of a
    cell's travel before the interval ends - 1.4 s out of 4390 here. The
    frame at s = 1 still recovers the observed perimeter exactly, which is
    what C2 asks for.
    """
    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid)
    seconds = evolution.arrival_seconds()
    inside = evolution.mask_end
    assert evolution.interval == pytest.approx(4390.0)
    assert np.nanmin(seconds[inside]) == pytest.approx(0.0)
    assert np.nanmax(seconds[inside]) == pytest.approx(4390.0, rel=1e-2)
    assert np.nanmax(seconds[inside]) <= 4390.0
    offset = evolution.arrival_seconds(t0=600.0)
    assert np.allclose(offset[inside] - seconds[inside], 600.0)


def test_vertex_source_choice_converges_as_the_grid_refines(pair):
    """
    Geometry-sampled and mask-sampled sources are two discretisations of the
    same boundary: mask vertices sit at cell centres, up to half a cell off
    the true boundary, while geometry vertices sit on it. So they disagree by
    a cell-scale amount that has to halve when the cell size halves.
    """
    initial, final, grid = pair

    def disagreement(on_grid):
        from_geometry = GaugeEvolution(initial, final, on_grid,
                                       vertex_source="geometry")
        from_mask = GaugeEvolution(initial, final, on_grid, vertex_source="mask")
        inside = from_geometry.mask_end
        return float(np.median(np.abs(from_geometry.progress[inside]
                                      - from_mask.progress[inside])))

    coarse = disagreement(grid)
    fine = disagreement(GridSpec.from_bounds(grid.bounds,
                                             cell_size=grid.cell_size / 2.0,
                                             crs=grid.crs))
    assert coarse < 2.5e-2
    assert fine < 0.7 * coarse


def test_fields_are_computed_once(pair):
    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid)
    assert evolution.T_f is evolution.T_f
    assert evolution.progress is evolution.progress


def test_contour_polygon_area_tracks_progress(pair):
    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid)
    areas = [evolution.contour_polygon(level).area for level in (0.0, 0.5, 1.0)]
    assert areas[0] < areas[1] < areas[2]
    assert areas[2] == pytest.approx(evolution.mask_end.sum() * grid.cell_area)


def test_summary_reports_provenance_and_diagnostics(pair):
    initial, final, grid = pair
    summary = GaugeEvolution(initial, final, grid, clock=TwoSidedClock()).summary()
    assert "Euclidean" in summary["metric"]
    assert "two-sided" in summary["clock"]
    assert summary["interval_s"] == pytest.approx(4390.0)
    assert summary["monotonicity_violations"] == 0
    assert summary["overshoot"] == 0.0
    assert summary["progress_min"] == 0.0 and summary["progress_max"] <= 1.0
    assert all(count > 0 for count in summary["source_vertices"])
