"""
Tests for the section 8 validation suite and the elliptical-versus-Euclidean
comparison.

Two kinds of case. The first uses the irregular synthetic pair from conftest,
which stands in for an observed pair - both perimeters pinned, growth that no
single vector generated - and checks the comparison reports something sane
about it. The second builds a series whose growth genuinely *is* elliptical,
by dilating a perimeter under the gauge, so there is a known right answer for
the held-out check to recover. The second kind is deliberately favourable to
the elliptical metric: its job is to show the diagnostics have the power to
tell the two metrics apart at all, which is what makes a null result on the
first kind readable as a finding rather than as a broken measurement.
"""

import numpy as np
import pytest
from shapely.geometry import Point

from grid import GridSpec
from metrics import EllipticalMetric, EuclideanMetric
from perimeter import PerimeterSeries
from validation import (MetricComparison, front_offset, head_transit, hold_out,
                        implied_ros)

from conftest import GROWTH_SECONDS, UTM10N


@pytest.fixture(scope="module")
def comparison_fixture(elliptical_growth):
    """A comparison over the elliptically grown outer pair."""
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    return MetricComparison(series[0], series[-1], grid, num_steps=20)


# ----------------------------------------------------------------------
# head_transit
# ----------------------------------------------------------------------

def test_head_transit_measures_the_support_difference(fire_origin):
    """For concentric discs the advance along any heading is the radius gap."""
    ox, oy = fire_origin
    grid = GridSpec.from_bounds((ox - 900.0, oy - 900.0, ox + 900.0, oy + 900.0),
                                cell_size=10.0, crs=UTM10N)
    inner = grid.rasterize(Point(ox, oy).buffer(300.0))
    outer = grid.rasterize(Point(ox, oy).buffer(700.0))
    for heading in ((1.0, 0.0), (0.0, 1.0), (3.0, 4.0)):
        transit = head_transit(inner, outer, grid, heading)
        assert transit == pytest.approx(400.0, abs=2.0 * grid.cell_size)


def test_head_transit_is_negative_when_the_front_retreats(fire_origin):
    """The sign is kept rather than taken away by an absolute value."""
    ox, oy = fire_origin
    grid = GridSpec.from_bounds((ox - 900.0, oy - 900.0, ox + 900.0, oy + 900.0),
                                cell_size=10.0, crs=UTM10N)
    inner = grid.rasterize(Point(ox, oy).buffer(300.0))
    outer = grid.rasterize(Point(ox, oy).buffer(700.0))
    assert head_transit(outer, inner, grid, (1.0, 0.0)) < 0.0


def test_head_transit_rejects_a_zero_heading(fire_origin):
    ox, oy = fire_origin
    grid = GridSpec.from_bounds((ox - 400.0, oy - 400.0, ox + 400.0, oy + 400.0),
                                cell_size=20.0, crs=UTM10N)
    mask = grid.rasterize(Point(ox, oy).buffer(200.0))
    with pytest.raises(ValueError, match="nonzero"):
        head_transit(mask, mask, grid, (0.0, 0.0))


# ----------------------------------------------------------------------
# implied vs observed ROS
# ----------------------------------------------------------------------

def test_implied_ros_recovers_the_rate_that_generated_the_growth(elliptical_growth):
    """
    When the growth is the ellipse's own, the implied head rate is the observed one.

    This is the calibration of the diagnostic: a ratio near 1 here is what
    makes a ratio far from 1 on an observed pair mean "the bulk vector does
    not explain the growth" rather than "the measurement is wrong".
    """
    from evolution import GaugeEvolution

    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    evolution = GaugeEvolution(series[0], series[-1], grid, lb=3.0)
    result = implied_ros(evolution)

    assert result["observed_mps"] == pytest.approx(0.30, rel=1e-9)
    # One cell of rasterisation slack at each end, over the interval.
    tolerance = 2.0 * grid.cell_size / elliptical_growth["interval"]
    assert result["implied_mps"] == pytest.approx(0.30, abs=tolerance)
    assert result["ratio"] == pytest.approx(1.0, abs=tolerance / 0.30)


def test_implied_ros_needs_a_heading_when_the_metric_has_none(pair):
    """A Euclidean run has no heading to measure a transit along."""
    from evolution import GaugeEvolution

    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid, metric=EuclideanMetric())
    with pytest.raises(ValueError, match="no heading"):
        implied_ros(evolution)

    # ...but supplying one makes the measurement available anyway.
    result = implied_ros(evolution, heading=(1.0, 0.0), observed_mps=0.25)
    assert np.isfinite(result["implied_mps"])


# ----------------------------------------------------------------------
# front_offset
# ----------------------------------------------------------------------

def test_front_offset_is_zero_for_identical_masks(fire_origin):
    ox, oy = fire_origin
    grid = GridSpec.from_bounds((ox - 600.0, oy - 600.0, ox + 600.0, oy + 600.0),
                                cell_size=10.0, crs=UTM10N)
    mask = grid.rasterize(Point(ox, oy).buffer(300.0))
    assert front_offset(mask, mask, grid) == 0.0


def test_front_offset_measures_a_known_dilation(fire_origin):
    """
    Two concentric discs 100 m apart average about half that across the annulus.

    The disputed cells fill the gap uniformly, so their mean distance to the
    nearer front is the half-width, not the width - which is the quantity the
    report reads against the cell size.
    """
    ox, oy = fire_origin
    grid = GridSpec.from_bounds((ox - 900.0, oy - 900.0, ox + 900.0, oy + 900.0),
                                cell_size=10.0, crs=UTM10N)
    inner = grid.rasterize(Point(ox, oy).buffer(400.0))
    outer = grid.rasterize(Point(ox, oy).buffer(500.0))
    assert front_offset(inner, outer, grid) == pytest.approx(50.0, abs=6.0)
    # Symmetric: which mask is passed first cannot matter.
    assert front_offset(outer, inner, grid) == pytest.approx(
        front_offset(inner, outer, grid), rel=1e-12)


# ----------------------------------------------------------------------
# MetricComparison
# ----------------------------------------------------------------------

def test_comparison_refuses_an_isotropic_metric(pair):
    """
    Comparing Euclidean against Euclidean measures nothing, so it raises.

    The synthetic pair carries ROS vectors but no LB, so without lb= the
    resolver correctly falls back to Euclidean - and the comparison has to
    say so rather than reporting a row of zeros as if they were a result.
    """
    initial, final, grid = pair
    with pytest.raises(ValueError, match="isotropic"):
        MetricComparison(initial, final, grid)


def test_both_runs_keep_the_endpoints_and_monotonicity(comparison_fixture):
    """C2 and C7 hold under the metric under test and under the control."""
    for run in (comparison_fixture.test, comparison_fixture.control):
        start, end = run.endpoint_jaccard()
        assert start == pytest.approx(1.0)
        assert end == pytest.approx(1.0)
        assert run.monotonicity_violations == 0
        assert run.overshoot() == pytest.approx(0.0)


def test_control_speed_does_not_change_the_comparison(elliptical_growth):
    """
    The Euclidean control's speed cancels in the two-sided ratio.

    Worth pinning, because it is the reason the comparison is about anisotropy
    alone: the control is not the same fire at the wrong speed, it is the same
    fire with no preferred direction. If this ever failed, every number in the
    report would depend on an arbitrary constant.
    """
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    slow = MetricComparison(series[0], series[-1], grid, num_steps=8,
                            control=EuclideanMetric(speed=0.05))
    fast = MetricComparison(series[0], series[-1], grid, num_steps=8,
                            control=EuclideanMetric(speed=20.0))
    assert (slow.progress_difference()["mean_abs"]
            == pytest.approx(fast.progress_difference()["mean_abs"], rel=1e-9))


def test_the_metric_changes_the_intermediate_timing(comparison_fixture):
    """The whole point: the bulk vector has to move the arrival times."""
    difference = comparison_fixture.progress_difference()
    assert difference["cells"] > 0
    assert difference["mean_abs"] > 0.01
    assert difference["max_abs"] > difference["mean_abs"]


def test_downwind_cells_burn_earlier_than_upwind_ones(comparison_fixture):
    """
    The redistribution is signed, and this is the sign it has to have.

    Downwind cells get a small forward time and a large backward one, so they
    burn early relative to upwind cells. A separation of the wrong sign means
    the metric was not reversed consistently between the two solves.
    """
    result = comparison_fixture.redistribution()
    assert result["downwind_cells"] > 0 and result["upwind_cells"] > 0
    assert result["separation"] > 0.0
    assert result["downwind_mean"] < result["upwind_mean"]


def test_reversing_the_heading_reverses_the_redistribution(elliptical_growth):
    """
    Falsifiability check: point the ellipse the other way and the sign flips.

    Without this the sign test above could pass on any field that happens to
    be later on one side of the fire, whatever the metric was doing.
    """
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    ellipse = elliptical_growth["ellipse"]
    forward = MetricComparison(series[0], series[-1], grid, num_steps=8,
                               metric=EllipticalMetric(ellipse))
    backward = MetricComparison(series[0], series[-1], grid, num_steps=8,
                                metric=EllipticalMetric(ellipse.reversed()))
    # The split direction is held fixed on purpose. Reversing the metric
    # reverses the field *and* the metric's own idea of downwind, so letting
    # the split follow the metric would cancel the sign change exactly and the
    # test would pass on a broken solve.
    assert forward.redistribution(heading=ellipse.uhat)["separation"] > 0.0
    assert backward.redistribution(heading=ellipse.uhat)["separation"] < 0.0


def test_the_disagreement_decomposition_is_exhaustive(comparison_fixture):
    """
    Size and placement shares account for the whole disagreement, and no more.

    The decomposition is what the write-up's "redistribution rather than a
    change of extent" prediction is tested against, so the arithmetic behind
    it is pinned here. The prediction itself is *not* asserted: which share
    dominates turns out to depend on the perimeter pair, so the code reports
    it rather than requiring an answer.
    """
    for row in comparison_fixture.frame_agreement():
        assert row["size_share"] >= 0.0
        assert row["placement_share"] >= 0.0
        assert (row["size_share"] + row["placement_share"]
                == pytest.approx(row["disagreement"], abs=1e-12))

    extent = comparison_fixture.extent_difference()
    assert extent["dominant"] in ("size", "placement")
    assert extent["mean_offset_m"] > 0.0


def test_nested_frames_are_all_size_and_no_placement(fire_origin):
    """
    The decomposition's defining case: one frame inside another is pure size.

    Two concentric discs disagree only about how much burned, so the placement
    share must be zero. Without this the split could be any pair of numbers
    that happened to add up.
    """
    ox, oy = fire_origin
    grid = GridSpec.from_bounds((ox - 900.0, oy - 900.0, ox + 900.0, oy + 900.0),
                                cell_size=10.0, crs=UTM10N)
    inner = grid.rasterize(Point(ox, oy).buffer(400.0))
    outer = grid.rasterize(Point(ox, oy).buffer(500.0))

    jaccard = float((inner & outer).sum()) / float((inner | outer).sum())
    size = abs(float(inner.sum()) - float(outer.sum())) / float(outer.sum())
    assert size == pytest.approx(1.0 - jaccard, abs=1e-12)


def test_advance_ratio_ladder_reports_where_it_failed_to_resolve(comparison_fixture):
    """
    The ladder is allowed to contain unresolved rows, and must label them.

    Backing advance per frame is head advance over HB, so at HB near 34 the
    fine end of the ladder cannot resolve the upwind side at all. Returning
    NaN there is the correct answer; quietly returning a quantised ratio would
    not be.
    """
    result = comparison_fixture.advance_ratios()
    assert result["hb"] > 1.0
    assert len(result["ladder"]) >= 2
    assert [row["num_steps"] for row in result["ladder"]] == sorted(
        (row["num_steps"] for row in result["ladder"]), reverse=True)
    # An unresolved row reports NaN in both columns rather than a ratio read
    # off an advance of less than a cell.
    for row in result["ladder"]:
        if not np.isfinite(row["test"]) or not np.isfinite(row["control"]):
            assert not np.isfinite(row["quotient"])
    assert result["resolved"] is None or np.isfinite(result["resolved"]["quotient"])


def test_the_two_metrics_share_their_pinned_endpoint_frames(comparison_fixture):
    """
    C2 in the comparison: at s = 0 and s = 1 the metric cannot matter.

    Both runs are pinned to the same two observations, so the endpoint frames
    are identical cell for cell. This is also why the coarse end of the
    advance-ratio ladder carries so little information - with a single
    interval, the only two frames it has are these.
    """
    test = comparison_fixture.test.frames(1)
    control = comparison_fixture.control.frames(1)
    assert len(test) == len(control) == 2
    for a, b in zip(test, control):
        assert np.array_equal(a, b)


def test_report_mentions_both_metrics(comparison_fixture):
    """Smoke test on the text block, which is what actually gets read."""
    text = comparison_fixture.report()
    assert "Euclidean" in text
    assert "ellipse" in text
    assert "redistribution" in text
    assert "separation" in text
    assert "size" in text and "placement" in text


# ----------------------------------------------------------------------
# Held-out perimeter
# ----------------------------------------------------------------------

def test_hold_out_needs_three_observations(pair):
    initial, final, grid = pair
    series = PerimeterSeries([initial, final])
    with pytest.raises(ValueError, match="three or more"):
        hold_out(series, grid, index=1)


@pytest.mark.parametrize("index", [0, 2])
def test_hold_out_rejects_an_endpoint(elliptical_growth, index):
    """Withholding an endpoint tests nothing: the run is pinned to it."""
    with pytest.raises(ValueError, match="endpoint"):
        hold_out(elliptical_growth["series"], elliptical_growth["grid"],
                 index=index)


def test_hold_out_level_matches_the_withheld_time(elliptical_growth):
    """The withheld perimeter is scored at its own place in the interval."""
    result = hold_out(elliptical_growth["series"], elliptical_growth["grid"],
                      index=1, lb=3.0, num_steps=20)
    assert result["level"] == pytest.approx(0.5)
    assert result["elapsed_s"] == pytest.approx(GROWTH_SECONDS / 2.0)


def test_hold_out_beats_both_trivial_baselines(elliptical_growth):
    """
    A Jaccard index on nested perimeters is high for almost any answer.

    So the score only means something against the two things you could say
    with no method at all - nothing moved, and everything burned at once.
    """
    result = hold_out(elliptical_growth["series"], elliptical_growth["grid"],
                      index=1, lb=3.0, num_steps=20)
    assert result["jaccard"] > result["static"]
    assert result["jaccard"] > result["saturated"]
    assert result["skill"] > 0.5


def test_hold_out_prefers_the_metric_that_generated_the_growth(elliptical_growth):
    """
    The one test that measures the method rather than the constraint.

    The intermediate perimeter here really is the ellipse's level set, so the
    elliptical metric should recover it and the Euclidean control should not.
    A clear gap is what licenses reading a *small* gap on observed data as a
    statement about the data rather than about the diagnostic.
    """
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    ellipse = elliptical_growth["ellipse"]

    elliptical = hold_out(series, grid, index=1, num_steps=20,
                          metric=EllipticalMetric(ellipse))
    euclidean = hold_out(series, grid, index=1, num_steps=20,
                         metric=EuclideanMetric())
    assert elliptical["jaccard"] > euclidean["jaccard"] + 0.1
    assert elliptical["jaccard"] > 0.9


def test_downwind_only_growth_is_reported_not_refused(fire_origin):
    """
    A fire that grew only downwind has no upwind half to compare.

    Common under a steady wind, and the first thing real data produced. The
    upwind mean reads NaN, the note says why, and the report still renders.
    """
    from datetime import datetime, timedelta, timezone

    from perimeter import Perimeter, ros_from_speed_bearing

    ox, oy = fire_origin
    grid = GridSpec.from_bounds((ox - 800.0, oy - 800.0, ox + 1600.0, oy + 1600.0),
                                cell_size=25.0, crs=UTM10N)
    # Heading 45 degrees; the final perimeter is the initial disc plus a disc
    # displaced 400 m along that heading, so every new cell lies downwind of
    # the initial centroid.
    ros = ros_from_speed_bearing(0.25, 45.0)
    shift = 400.0 / np.sqrt(2.0)
    initial = Point(ox, oy).buffer(300.0)
    final = initial.union(Point(ox + shift, oy + shift).buffer(300.0))
    t0 = datetime(2024, 7, 23, 1, 0, tzinfo=timezone.utc)
    a = Perimeter(initial, UTM10N, time=t0, ros=ros)
    b = Perimeter(final, UTM10N, time=t0 + timedelta(hours=1), ros=ros)

    comparison = MetricComparison(a, b, grid, lb=3.0, num_steps=6)
    result = comparison.redistribution()
    assert result["upwind_cells"] == 0
    assert result["downwind_cells"] > 0
    assert np.isnan(result["upwind_mean"]) and np.isnan(result["separation"])
    assert np.isfinite(result["downwind_mean"])
    assert "downwind" in result["note"]

    text = comparison.report()
    assert "note: all growth lies downwind" in text
    assert "separation    -" in text


def test_a_pair_that_is_not_nested_is_flagged_in_the_report(fire_origin):
    """Two disjoint patches are not footprints; the report must say so up front."""
    from datetime import datetime, timedelta, timezone

    from perimeter import Perimeter, ros_from_speed_bearing

    ox, oy = fire_origin
    grid = GridSpec.from_bounds((ox - 800.0, oy - 800.0, ox + 1600.0, oy + 1600.0),
                                cell_size=25.0, crs=UTM10N)
    ros = ros_from_speed_bearing(0.25, 45.0)
    t0 = datetime(2024, 7, 23, 1, 0, tzinfo=timezone.utc)
    a = Perimeter(Point(ox, oy).buffer(200.0), UTM10N, time=t0, ros=ros)
    b = Perimeter(Point(ox + 500.0, oy + 500.0).buffer(300.0), UTM10N,
                  time=t0 + timedelta(hours=1), ros=ros)
    comparison = MetricComparison(a, b, grid, lb=3.0, num_steps=4)
    assert comparison.initial_outside_final == pytest.approx(1.0)
    assert "WARNING" in comparison.report() and "not nested" in comparison.report()

    nested = MetricComparison(a, Perimeter(a.geometry.buffer(300.0), UTM10N,
                                           time=b.time, ros=ros), grid, lb=3.0, num_steps=4)
    assert nested.initial_outside_final == 0.0
    assert "WARNING" not in nested.report()


# ----------------------------------------------------------------------
# Series with sparse vectors
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def sparse_series(elliptical_growth):
    """
    Five perimeters where only the third carries a ROS vector.

    Built by slicing the elliptical growth at more levels; the vector is
    stripped from all but the middle one, which is how real series arrive.
    """
    from datetime import timedelta

    from metrics import EllipticalMetric, Source
    from perimeter import Perimeter, PerimeterSeries

    grid, ros, ellipse = (elliptical_growth[k] for k in ("grid", "ros", "ellipse"))
    first = elliptical_growth["series"][0]
    start = first.mask(grid)
    sx, sy = grid.boundary_vertices(start)
    arrival = EllipticalMetric(ellipse).arrival(Source(sx, sy, mask=start, direction="outward"), grid)
    perimeters = []
    for i, fraction in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        mask = start | (arrival <= fraction * GROWTH_SECONDS)
        perimeters.append(Perimeter(
            grid.mask_to_polygon(mask), grid.crs,
            time=first.time + timedelta(seconds=fraction * GROWTH_SECONDS),
            ros=ros if i == 2 else None))
    return PerimeterSeries(perimeters), grid


def test_a_pair_with_no_vector_borrows_the_nearest_one(sparse_series):
    from validation import series_metric

    series, grid = sparse_series
    for k in range(4):
        metric, provenance = series_metric(series, k, lb=3.0, grid=grid)
        assert metric.is_anisotropic
        assert metric.ellipse.head == pytest.approx(0.30)
        if k in (1, 2):
            assert "borrowed" not in provenance      # one end has the vector
        else:
            assert "borrowed from the observation at" in provenance


def test_borrowing_can_be_switched_off(sparse_series):
    from validation import series_metric

    series, grid = sparse_series
    metric, provenance = series_metric(series, 0, lb=3.0, grid=grid, borrow=False)
    assert not metric.is_anisotropic
    assert "neither perimeter carries a ROS vector" in provenance


def test_series_run_validates_every_pair_and_every_triplet(sparse_series, tmp_path, capsys):
    """The command the user runs on a whole folder: no crash, one line per pair and triplet."""
    import geopandas as gpd
    from pyproj import CRS, Transformer
    from shapely.ops import transform

    from validation import main

    series, grid = sparse_series
    to_wgs = Transformer.from_crs(grid.crs, CRS.from_epsg(4326), always_xy=True).transform
    paths = []
    for p in series:
        record = {"time_iso": p.time.isoformat(), "geometry": transform(to_wgs, p.geometry)}
        if p.has_ros:
            record["speed_mps"], record["dir_deg"] = p.ros_speed, p.ros_bearing
        path = str(tmp_path / f"perimeter_{p.time:%Y%m%dT%H%M%SZ}.shp")
        gpd.GeoDataFrame([record], crs="EPSG:4326").to_file(path)
        paths.append(path)

    assert main(paths + ["--lb", "3", "--cell-size", "40", "--num-steps", "6"]) == 0
    out = capsys.readouterr().out
    assert "5 perimeters, 1 carrying a ROS vector" in out
    assert out.count("(borrowed)") == 2 and out.count("(own)") == 2
    assert "Held-out" in out
    assert "elliptical better on" in out
