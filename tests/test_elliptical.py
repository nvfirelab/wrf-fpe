"""
Tests for the elliptical metric: the method's actual configuration.

Two things have to hold before anything downstream is worth running. The
solved front must be the analytic ellipse, and the anisotropy must actually
reach the output, measured as a head-to-back advance ratio rather than as
centroid drift.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from clock import TwoSidedClock
from evolution import GaugeEvolution, advance_ratio
from gauge import gauge_arrival
from grid import GridSpec
from metrics import (EllipticalMetric, EuclideanMetric, Source, infer_ros,
                     resolve_metric)
from wavelet import FireEllipse, lb_from_wind

UTM10N = "EPSG:32610"


def point_source_field(ellipse, span, cells=401, total_time=4000.0):
    """A single outward solve from a point ignition, and the grid it sits on."""
    grid = GridSpec.from_bounds((-span, -span, span, span),
                                cell_size=2 * span / (cells - 1), crs=UTM10N)
    X, Y = grid.meshgrid()
    T = gauge_arrival([0.0], [0.0], ellipse.uhat, ellipse.a, ellipse.b,
                      ellipse.c, X, Y)
    return grid, T


# ----------------------------------------------------------------------
# The analytic front
# ----------------------------------------------------------------------

@pytest.mark.parametrize("lb", [1.2, 3.0, 6.0])
def test_solved_front_matches_the_analytic_ellipse(lb):
    """
    Compared over the whole front, not just the axis rates - the failure mode
    of an iterative solver is axis rates that look perfect while the geometry
    between them is wrong.
    """
    ellipse = FireEllipse.from_ros((0.25 * np.cos(0.35), 0.25 * np.sin(0.35)), lb)
    t = 3000.0
    grid, T = point_source_field(ellipse, span=ellipse.head * 4000 * 1.15, cells=501)

    ox, oy = ellipse.outline(t, n=1440)
    analytic = grid.rasterize(Polygon(np.column_stack([ox, oy])))
    solved = T <= t

    intersection = np.count_nonzero(analytic & solved)
    union = np.count_nonzero(analytic | solved)
    assert intersection / union > 0.99


@pytest.mark.parametrize("lb", [1.2, 3.0, 6.0])
def test_radial_front_error_is_far_inside_two_percent(lb):
    """Mean radial error along rays, the write-up's 2% budget."""
    ellipse = FireEllipse.from_ros((0.3, 0.0), lb)
    t = 2000.0
    ox, oy = ellipse.outline(t, n=720)
    T = gauge_arrival([0.0], [0.0], ellipse.uhat, ellipse.a, ellipse.b,
                      ellipse.c, ox.reshape(1, -1), oy.reshape(1, -1))
    # Arrival time scales linearly along a ray, so a relative time error is a
    # relative radial error.
    assert np.mean(np.abs(T - t)) / t < 1e-12


# ----------------------------------------------------------------------
# C6: the anisotropy reaching the output
# ----------------------------------------------------------------------

@pytest.mark.parametrize("lb", [1.2, 1.5, 3.0])
def test_advance_ratio_recovers_hb_from_a_point_ignition(lb):
    ellipse = FireEllipse.from_ros((0.25 * np.cos(0.35), 0.25 * np.sin(0.35)), lb)
    grid, T = point_source_field(ellipse, span=ellipse.head * 4000 * 1.15, cells=701)
    steps = 20 if lb < 2.0 else 4
    frames = [T <= level for level in np.linspace(0.0, 4000.0, steps + 1)]

    ratio = advance_ratio(frames, grid, ellipse.uhat)
    assert ratio == pytest.approx(ellipse.hb, rel=0.03)


def test_advance_ratio_reports_nan_when_the_backing_is_unresolved():
    """
    Backing advance is the head advance divided by HB, so a strongly
    elongated ellipse needs a very fine grid to measure. Skipping those frame
    pairs is better than returning a quantised ratio that looks plausible.
    """
    ellipse = FireEllipse.from_ros((0.25, 0.0), 10.0)
    grid, T = point_source_field(ellipse, span=ellipse.head * 4000 * 1.15, cells=401)
    frames = [T <= level for level in np.linspace(0.0, 4000.0, 21)]
    assert np.isnan(advance_ratio(frames, grid, ellipse.uhat))


def test_isotropic_wavelet_advances_equally_in_both_directions():
    ellipse = FireEllipse.isotropic(0.25)
    grid, T = point_source_field(ellipse, span=0.25 * 4000 * 1.15, cells=401)
    frames = [T <= level for level in np.linspace(0.0, 4000.0, 21)]
    assert advance_ratio(frames, grid, (1.0, 0.0)) == pytest.approx(1.0, rel=0.02)


# ----------------------------------------------------------------------
# The metric object
# ----------------------------------------------------------------------

def test_metric_delegates_to_the_gauge(pair):
    initial, _, grid = pair
    ellipse = FireEllipse.from_ros((0.2, 0.15), 3.0)
    source = Source(*initial.boundary_vertices(grid), mask=initial.mask(grid))
    X, Y = grid.meshgrid()
    assert np.allclose(
        EllipticalMetric(ellipse).arrival(source, grid),
        gauge_arrival(source.x, source.y, ellipse.uhat, ellipse.a, ellipse.b,
                      ellipse.c, X, Y), equal_nan=True)


def test_metric_reversal_negates_the_heading():
    metric = EllipticalMetric.from_ros((0.2, 0.15), 3.0)
    assert np.allclose(metric.reversed().ellipse.uhat, -metric.ellipse.uhat)
    assert metric.reversed().reversed().ellipse == metric.ellipse
    assert metric.is_anisotropic


def test_metric_from_wind_uses_finney():
    metric = EllipticalMetric.from_wind((0.25, 0.0), 5.0)
    assert metric.ellipse.lb == pytest.approx(lb_from_wind(5.0), rel=1e-9)


def test_metric_rejects_a_non_ellipse():
    with pytest.raises(TypeError, match="needs a FireEllipse"):
        EllipticalMetric("an ellipse, honest")


# ----------------------------------------------------------------------
# The constraints, under the metric that matters
# ----------------------------------------------------------------------

@pytest.fixture
def elliptical(pair):
    initial, final, grid = pair
    return GaugeEvolution(initial, final, grid, lb=3.0), grid


def test_endpoints_stay_exact_under_the_ellipse(elliptical):
    """
    C2 is structural: T_f = 0 and T_b = 0 force s = 0 and s = 1 whatever the
    metric is. That is why changing the metric costs nothing in endpoint
    accuracy - the property everything else in the method is built on.
    """
    evolution, _ = elliptical
    assert evolution.metric.is_anisotropic
    frames = evolution.frames(num_steps=20)
    assert np.array_equal(frames[0], evolution.mask_start & evolution.mask_end)
    assert np.array_equal(frames[-1], evolution.mask_end)
    assert evolution.endpoint_jaccard() == (1.0, 1.0)
    assert evolution.overshoot() == 0.0


def test_monotone_and_bounded_under_the_ellipse(elliptical):
    evolution, _ = elliptical
    frames = evolution.frames(num_steps=20)
    for earlier, later in zip(frames[:-1], frames[1:]):
        assert not (earlier & ~later).any()
    assert evolution.monotonicity_violations == 0
    inside = evolution.progress[evolution.mask_end]
    assert inside.min() == 0.0 and inside.max() <= 1.0


def test_detached_components_are_still_reached(elliptical):
    """C5 does not depend on the metric: the gauge ignores connectivity."""
    from scipy.ndimage import label

    evolution, _ = elliptical
    labels, n_components = label(evolution.mask_end)
    assert n_components == 2
    assert np.isfinite(evolution.progress[labels == 2]).all()


def test_the_ellipse_changes_the_answer(elliptical, pair):
    """
    If the elliptical and Euclidean fields were indistinguishable the metric
    would not be reaching the output at all. How *much* they differ on real
    perimeters is the question M4 asks; that they differ is a precondition
    for asking it.
    """
    evolution, grid = elliptical
    initial, final, _ = pair
    euclidean = GaugeEvolution(initial, final, grid, metric=EuclideanMetric())
    inside = evolution.mask_end
    difference = np.abs(evolution.progress[inside] - euclidean.progress[inside])
    assert np.median(difference) > 1e-3
    assert difference.max() > 1e-2


def test_reversal_matters(elliptical, pair):
    """
    Getting the reversal wrong does not throw. Solving the inward leg under
    the forward metric gives a plausible-looking field with a different
    intermediate ordering, which is what makes it worth a test.
    """
    evolution, grid = elliptical
    initial, final, _ = pair
    metric = evolution.metric

    wrong = GaugeEvolution(initial, final, grid, metric=metric)
    wrong._cache["T_b"] = metric.arrival(
        Source(*final.boundary_vertices(grid), mask=final.mask(grid),
               direction="inward"), grid, where=wrong.mask_end)
    inside = evolution.mask_end
    assert not np.allclose(wrong.progress[inside], evolution.progress[inside])


# ----------------------------------------------------------------------
# Section 7: where the metric comes from
# ----------------------------------------------------------------------

def test_resolution_uses_the_final_perimeters_vector(pair):
    """
    A perimeter's vector describes the interval that ends at it.

    So the pair's vector is the final perimeter's, not the mean of the two:
    the initial's belongs to the previous interval, and averaging it in put
    the previous wind into every pair (found on the Mill Fire series, where
    a south-easterly 0.09 m/s interval came out as 0.71 m/s north-east).
    """
    initial, final, grid = pair
    metric, provenance = resolve_metric(initial, final, grid, lb=3.0)
    assert isinstance(metric, EllipticalMetric)
    assert "the final perimeter's ROS vector" in provenance
    assert metric.ellipse.head == pytest.approx(np.hypot(*final.ros), rel=1e-9)
    assert np.allclose(metric.ellipse.uhat, final.ros / np.hypot(*final.ros))
    # Not the mean, and not the initial's.
    assert metric.ellipse.head != pytest.approx(np.hypot(*(0.5 * (initial.ros + final.ros))))


def test_resolution_falls_back_to_the_initials_vector_and_says_so(pair):
    from dataclasses import replace

    initial, final, grid = pair
    metric, provenance = resolve_metric(initial, replace(final, ros=None), grid, lb=3.0)
    assert isinstance(metric, EllipticalMetric)
    assert "previous interval's vector standing in" in provenance
    assert metric.ellipse.head == pytest.approx(np.hypot(*initial.ros), rel=1e-9)


def test_resolution_falls_back_to_euclidean_without_an_lb(pair):
    """
    A ROS vector alone gives one of the ellipse's two free parameters. Rather
    than guess the elongation, the method drops to its floor - which is still
    endpoint-exact, monotone and continuous.
    """
    initial, final, grid = pair
    metric, provenance = resolve_metric(initial, final, grid)
    assert isinstance(metric, EuclideanMetric)
    assert "no LB is available" in provenance


def test_resolution_takes_lb_from_wind(pair):
    initial, final, grid = pair
    metric, provenance = resolve_metric(initial, final, grid, wind_mps=4.0)
    assert metric.ellipse.lb == pytest.approx(lb_from_wind(4.0), rel=1e-9)
    assert "Finney Eq. 13" in provenance


def test_resolution_reports_a_missing_vector(pair):
    initial, final, grid = pair
    stripped = [type(p)(geometry=p.geometry, crs=p.crs, time=p.time)
                for p in (initial, final)]
    metric, provenance = resolve_metric(*stripped, grid, lb=3.0)
    assert isinstance(metric, EuclideanMetric)
    assert "neither perimeter carries a ROS vector" in provenance


def test_geometric_inference_recovers_the_observed_bearing(pair):
    """
    The validation experiment section 7 sets up: infer the vector from the
    pair alone and compare it against the measured one. Agreement is what
    would extend the method to perimeters carrying no spread data at all.
    """
    initial, final, grid = pair
    inferred = infer_ros(initial.mask(grid), final.mask(grid), grid,
                         initial.interval_to(final))
    observed = 0.5 * (initial.ros + final.ros)

    def bearing(v):
        return np.degrees(np.arctan2(v[0], v[1])) % 360.0

    separation = abs((bearing(inferred) - bearing(observed) + 180.0) % 360.0 - 180.0)
    assert separation < 45.0
    assert 0.0 < np.hypot(*inferred) < 10.0


def test_geometric_inference_needs_growth(pair):
    initial, _, grid = pair
    mask = initial.mask(grid)
    with pytest.raises(ValueError, match="no growth"):
        infer_ros(mask, mask, grid, 4390.0)


def test_inference_is_opt_in(pair):
    """
    A vector inferred from the same pair being interpolated cannot tell you
    anything the endpoints did not already imply, so it should never happen
    silently.
    """
    initial, final, grid = pair
    stripped = [type(p)(geometry=p.geometry, crs=p.crs, time=p.time)
                for p in (initial, final)]
    off, _ = resolve_metric(*stripped, grid, lb=3.0)
    on, provenance = resolve_metric(*stripped, grid, lb=3.0,
                                    infer_from_geometry=True)
    assert isinstance(off, EuclideanMetric)
    assert isinstance(on, EllipticalMetric)
    assert "inferred from the perimeter pair" in provenance


def test_evolution_records_its_metric_provenance(pair):
    initial, final, grid = pair
    assert "no LB" in GaugeEvolution(initial, final, grid).metric_provenance
    assert "elliptical" in GaugeEvolution(initial, final, grid,
                                          lb=3.0).metric_provenance
    assert "as given" in GaugeEvolution(initial, final, grid,
                                        metric=EuclideanMetric()).metric_provenance
    summary = GaugeEvolution(initial, final, grid, wind_mps=4.0,
                             clock=TwoSidedClock()).summary()
    assert "Finney" in summary["metric_provenance"]
    assert "ellipse LB=" in summary["metric"]
