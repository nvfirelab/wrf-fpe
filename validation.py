"""
Module implementing the validation suite of section 8, and the
elliptical-versus-Euclidean comparison that suite exists to set up.

Five of the eight checks in that table already sit where they are measured:
the closed-form spot checks, the Euclidean regression and the analytic front
in the tests, the endpoint residual and the monotonicity count on
GaugeEvolution. Each of those scores one run against its own definition. The
three collected here need something a single run does not carry.

    implied vs observed ROS   the pair's geometry against the vector supplied
                              with it - how much of the observed growth the
                              bulk vector explains
    held-out perimeter        a third observation the run never saw, which is
                              the only check that measures the method rather
                              than the constraint
    metric comparison         a second run under a different metric

The last is the milestone question: does the bulk vector change the answer,
and by how much. The write-up predicts a specific shape for that answer, and
it is worth stating before measuring so the result can disagree with it.
Both perimeters are pinned as hard constraints, so extent at s = 0 and s = 1
is fixed whatever the metric is; and along a straight ray outward the metric
is a constant scale factor that cancels in the two-sided ratio entirely. So
the prediction is a *redistribution of timing* - downwind cells burning
earlier and upwind cells later - rather than a change of extent, with the
metric biting only where the geometry creates competing claims about which
part of the perimeter reaches a cell first.

`MetricComparison` measures both halves of that separately: `redistribution`
for the timing, `extent_difference` for the area. A run where the first is
large and the second is near zero confirms the prediction; a run where both
are near zero is the finding that a bulk vector carries little information
once the endpoints are pinned.
"""

import numpy as np

from evolution import GaugeEvolution, advance_ratio
from metrics import EuclideanMetric, resolve_metric


# ----------------------------------------------------------------------
# Implied vs observed ROS
# ----------------------------------------------------------------------

def head_transit(mask_start, mask_end, grid, heading):
    """
    Distance the front advanced along a heading between two rasterised masks.

    The support function difference: the furthest extent of each mask measured
    along the heading, subtracted. This is the same quantity `advance_ratio`
    takes between consecutive frames, and it is deliberately an extent rather
    than a mean normal advance - a mean over the front is dominated by flank
    behaviour and compresses towards the isotropic answer as LB grows.

    Parameters:
    -----------
    mask_start, mask_end : numpy.ndarray
        The two observed perimeters rasterised onto the grid
    grid : GridSpec
        The shared computational grid
    heading : array-like
        Direction to measure along; need not be normalised

    Returns:
    --------
    transit : float
        Advance along the heading in metres, negative if the front retreated
    """
    ux, uy = np.asarray(heading, dtype=float).reshape(2)
    norm = np.hypot(ux, uy)
    if norm <= 0:
        raise ValueError("heading must be a nonzero vector")
    mask_start = np.asarray(mask_start, dtype=bool)
    mask_end = np.asarray(mask_end, dtype=bool)
    if not mask_start.any() or not mask_end.any():
        raise ValueError("both masks must be non-empty to measure a transit")

    X, Y = grid.meshgrid()
    projection = (X * ux + Y * uy) / norm
    return float(projection[mask_end].max() - projection[mask_start].max())


def implied_ros(evolution, heading=None, observed_mps=None):
    """
    The head rate the perimeter pair implies, against the one that was observed.

    Head-direction transit distance over the interval, against |v|. The gap is
    a result worth reporting rather than a failure: the observed vector is a
    bulk estimate for the whole fire over the whole interval, and the transit
    is what the two perimeters actually did along that heading. A ratio far
    from 1 says the bulk vector explains only part of the observed growth -
    which bears directly on how much the metric can be expected to change,
    since a vector that does not describe the growth cannot redistribute its
    timing convincingly either.

    Parameters:
    -----------
    evolution : GaugeEvolution
        A completed or constructible run over the pair
    heading : array-like, optional
        Unit heading. Taken from the metric's ellipse when it has one.
    observed_mps : float, optional
        The observed head rate |v|. Taken from the metric's ellipse when it
        has one.

    Returns:
    --------
    result : dict
        transit_m, interval_s, implied_mps, observed_mps, ratio, heading
    """
    ellipse = getattr(evolution.metric, "ellipse", None)
    if heading is None:
        if ellipse is None:
            raise ValueError(
                "this metric has no heading; pass heading= to measure transit "
                "along a direction of your choosing")
        heading = ellipse.uhat
    if observed_mps is None:
        if ellipse is None:
            raise ValueError(
                "this metric has no head rate; pass observed_mps= to compare "
                "the implied rate against something")
        observed_mps = ellipse.head

    transit = head_transit(evolution.mask_start, evolution.mask_end,
                           evolution.grid, heading)
    interval = evolution.interval
    implied = transit / interval
    return {
        "transit_m": transit,
        "interval_s": interval,
        "implied_mps": implied,
        "observed_mps": float(observed_mps),
        "ratio": implied / float(observed_mps) if observed_mps else float("nan"),
        "heading": np.asarray(heading, dtype=float).reshape(2),
    }


# ----------------------------------------------------------------------
# Held-out perimeter
# ----------------------------------------------------------------------

def hold_out(series, grid, index=1, **kwargs):
    """
    Evolve across the outer pair of a series and score a withheld observation.

    The only check in the table that measures the method rather than the
    constraint. Every other score is helped by the endpoints being pinned;
    this one is not, because the withheld perimeter is not an endpoint of the
    run that predicts it.

    It is reported against two trivial baselines, because a Jaccard index on
    nested fire perimeters is high for nearly any answer and means nothing on
    its own. `static` is the score of predicting that nothing moved after the
    first observation; `saturated` is the score of predicting that everything
    burned immediately. A method that cannot beat both is not interpolating.

    Parameters:
    -----------
    series : PerimeterSeries
        Three or more observations in time order, already on the grid's CRS
    grid : GridSpec
        The shared computational grid
    index : int
        Which observation to withhold. Must be strictly between the ends.
    **kwargs
        Passed to GaugeEvolution - metric, lb, clock and so on

    Returns:
    --------
    result : dict
        level, jaccard, static, saturated, skill, metric, and the times used
    """
    if len(series) < 3:
        raise ValueError(
            f"a held-out test needs three or more observations; got {len(series)}")
    if not 0 < index < len(series) - 1:
        raise ValueError(
            f"index {index} is an endpoint of the run, so withholding it tests "
            f"nothing; choose 1..{len(series) - 2}")

    initial, final, withheld = series[0], series[-1], series[index]
    evolution = GaugeEvolution(initial, final, grid, **kwargs)

    # Where in the interval the withheld observation falls, as progress.
    elapsed = initial.interval_to(withheld)
    level = elapsed / initial.interval_to(final)

    predicted = (evolution.progress <= level) & evolution.mask_end
    truth = withheld.mask(grid)
    static = _jaccard(evolution.mask_start, truth)
    saturated = _jaccard(evolution.mask_end, truth)
    score = _jaccard(predicted, truth)

    # Skill against the better of the two things you could say without a
    # method at all. Zero means the run added nothing; one means it is exact.
    baseline = max(static, saturated)
    skill = (score - baseline) / (1.0 - baseline) if baseline < 1.0 else float("nan")

    return {
        "index": index,
        "level": level,
        "elapsed_s": elapsed,
        "interval_s": initial.interval_to(final),
        "jaccard": score,
        "static": static,
        "saturated": saturated,
        "skill": skill,
        "metric": evolution.metric.describe(),
        "metric_provenance": evolution.metric_provenance,
    }


# ----------------------------------------------------------------------
# Metrics for a series with sparse vectors
# ----------------------------------------------------------------------

def nearest_ros(series, when, exclude=()):
    """
    The ROS vector to borrow for an interval around `when`.

    A perimeter's vector describes the interval that ends at it, so the best
    stand-in for an interval with no vector of its own is the vector of the
    nearest *later* observation that has one - it covers the time just after
    ours, under the same wind more often than not. Only if no later
    observation carries one is the nearest earlier one used.

    Parameters:
    -----------
    series : PerimeterSeries
    when : datetime
        The instant to be near - normally the midpoint of a pair
    exclude : iterable of Perimeter
        Observations not to borrow from (the pair's own, already tried)

    Returns:
    --------
    ros, donor : numpy.ndarray or None, Perimeter or None
    """
    skip = {id(p) for p in exclude}
    donors = [p for p in series if p.has_ros and p.time is not None
              and id(p) not in skip]
    if not donors:
        return None, None
    later = [p for p in donors if p.time > when]
    pool = later if later else donors
    donor = min(pool, key=lambda p: abs((p.time - when).total_seconds()))
    return donor.ros, donor


def series_metric(series, k, lb=None, wind_mps=None, infer_from_geometry=False,
                  grid=None, borrow=True):
    """
    The metric for consecutive pair k of a series, and where it came from.

    Real series carry vectors on some observations and not others. The pair's
    own vector is preferred, exactly as `resolve_metric` does - the final
    perimeter's, since a vector describes the interval ending at it; when
    neither end has one, the vector of the nearest later observation is
    borrowed, because a bulk vector a few minutes away is a far better prior
    than pretending the fire had no direction. The provenance string names
    the donor and the time gap so nobody mistakes a borrowed vector for an
    observed one.

    Parameters:
    -----------
    series : PerimeterSeries
        On the grid's CRS
    k : int
        Index of the pair (series[k] -> series[k + 1])
    lb, wind_mps, infer_from_geometry, grid
        As for `resolve_metric`
    borrow : bool
        Allow borrowing from elsewhere in the series

    Returns:
    --------
    metric : Metric
    provenance : str
    """
    from metrics import EllipticalMetric, _resolve_lb, resolve_metric

    a, b = series[k], series[k + 1]
    if a.has_ros or b.has_ros or not borrow:
        return resolve_metric(a, b, grid, lb=lb, wind_mps=wind_mps,
                              infer_from_geometry=infer_from_geometry)

    midpoint = a.time + (b.time - a.time) / 2
    ros, donor = nearest_ros(series, midpoint, exclude=(a, b))
    if ros is None:
        return resolve_metric(a, b, grid, lb=lb, wind_mps=wind_mps,
                              infer_from_geometry=infer_from_geometry)
    lb_value, lb_note = _resolve_lb(a, b, lb, wind_mps)
    gap_min = (donor.time - midpoint).total_seconds() / 60.0
    note = (f"ROS vector borrowed from the observation at "
            f"{donor.time.strftime('%H:%M:%SZ')} ({gap_min:+.1f} min from the pair's "
            "midpoint; neither end carries one)")
    if lb_value is None:
        from metrics import EuclideanMetric
        return EuclideanMetric(), f"Euclidean: {lb_note}, so the ellipse is undetermined ({note})"
    return EllipticalMetric.from_ros(ros, lb_value), f"elliptical: {note}; {lb_note}"


# ----------------------------------------------------------------------
# The comparison
# ----------------------------------------------------------------------

class MetricComparison:
    """
    The same perimeter pair run under two metrics, and the gap between them.

    The test metric is the elliptical one built from the observed ROS vector;
    the control is Euclidean, which is what the method reduces to when no
    vector is available. Everything else - grid, clock, perimeters, frame
    count - is shared, so any difference in the output is the metric and
    nothing else.

    One consequence of the two-sided clock is worth knowing before reading any
    of these numbers: a uniform rescaling of the metric cancels in
    T_f / (T_f + T_b), so the control's speed does not affect the comparison
    at all. The control is not "the same fire moving at the wrong speed" - it
    is the same fire with no preferred direction. The comparison is therefore
    about anisotropy alone, which is the only thing a bulk vector adds.
    """

    def __init__(self, initial, final, grid, metric=None, control=None,
                 num_steps=20, lb=None, wind_mps=None,
                 infer_from_geometry=False, **kwargs):
        """
        Parameters:
        -----------
        initial, final : Perimeter
            The two observations
        grid : GridSpec
            The shared computational grid, in metres
        metric : Metric, optional
            The metric under test. Resolved from the observations when
            omitted, and required to be anisotropic - comparing a Euclidean
            metric against a Euclidean control measures nothing.
        control : Metric, optional
            What to compare against. Defaults to Euclidean.
        num_steps : int
            Frames to compare across
        lb, wind_mps, infer_from_geometry
            Passed to the metric resolution when `metric` is omitted
        **kwargs
            Passed to both GaugeEvolution constructions
        """
        if metric is None:
            metric, provenance = resolve_metric(
                initial.to_crs(grid.crs), final.to_crs(grid.crs), grid,
                lb=lb, wind_mps=wind_mps, infer_from_geometry=infer_from_geometry)
        else:
            provenance = f"as given: {metric.describe()}"
        if not metric.is_anisotropic:
            raise ValueError(
                f"the metric under test is isotropic ({metric.describe()}), so "
                f"the comparison would measure nothing. {provenance}. Supply "
                "lb= or wind_mps= to close the ellipse, or pass metric= "
                "explicitly.")

        self.metric_provenance = provenance
        self.control_metric = control if control is not None else EuclideanMetric()
        # How much of the initial perimeter lies outside the final one. The
        # method treats both as hard constraints, so a pair that is not
        # nested is either rasterisation noise (a fraction of a percent) or
        # two growth patches that were never footprints - and every number
        # below reads differently in that case, so the report says so.
        ma = initial.to_crs(grid.crs).mask(grid)
        mb = final.to_crs(grid.crs).mask(grid)
        self.initial_outside_final = (float((ma & ~mb).sum()) / float(ma.sum())
                                      if ma.any() else 0.0)
        self.num_steps = int(num_steps)
        self.test = GaugeEvolution(initial, final, grid, metric=metric,
                                   num_steps=num_steps, **kwargs)
        self.control = GaugeEvolution(initial, final, grid,
                                      metric=self.control_metric,
                                      num_steps=num_steps, **kwargs)

    # ------------------------------------------------------------------

    @property
    def grid(self):
        """The grid both runs share."""
        return self.test.grid

    @property
    def heading(self):
        """The test metric's heading, which defines downwind."""
        return self.test.metric.ellipse.uhat

    @property
    def band(self):
        """
        Cells the interpolation actually answers for.

        Inside the final perimeter and outside the initial one. Cells already
        burned at the first observation are forced to zero by both runs, and
        cells outside the final perimeter are NaN in both, so including either
        would dilute every difference below with cells that cannot differ.
        """
        return self.test.mask_end & ~self.test.mask_start

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def progress_difference(self):
        """
        How far apart the two progress fields are, over the band.

        Reported in progress and in seconds. The seconds are the number that
        matters for an export: TIGN_G is an arrival time, so an ignition that
        moves by less than the host model's timestep has not moved at all as
        far as the coupled run is concerned.

        Returns:
        --------
        result : dict
            mean_abs, rms, max_abs, and each as seconds, plus the signed mean
        """
        delta = (self.test.progress - self.control.progress)[self.band]
        delta = delta[np.isfinite(delta)]
        interval = self.test.interval
        if not delta.size:
            return {"cells": 0}
        return {
            "cells": int(delta.size),
            "mean_signed": float(delta.mean()),
            "mean_abs": float(np.abs(delta).mean()),
            "rms": float(np.sqrt(np.mean(delta ** 2))),
            "max_abs": float(np.abs(delta).max()),
            "mean_abs_s": float(np.abs(delta).mean() * interval),
            "max_abs_s": float(np.abs(delta).max() * interval),
            "interval_s": interval,
        }

    def redistribution(self, heading=None):
        """
        Whether downwind cells burn earlier under the ellipse, and by how much.

        The prediction, made mechanical: downwind cells receive a small T_f
        and a large T_b, so they burn early; upwind cells the reverse. Split
        the band by the sign of the displacement from the initial perimeter's
        centroid along the heading, and take the mean change in progress in
        each half.

        Read `separation`, not the two means on their own. Progress is a ratio
        of two solves and both endpoints are pinned, so the whole field can
        sit slightly later or earlier under one metric than the other without
        that meaning anything directional; `progress_difference` reports that
        offset as `mean_signed`. The difference between the halves is what
        survives it, and its sign is the falsifiable part - a separation of
        the wrong sign is a reversed metric, not a weak effect.

        Parameters:
        -----------
        heading : array-like, optional
            Direction defining downwind. Defaults to the test metric's own,
            which is what a report wants. Pass it explicitly when comparing
            two runs whose metrics point different ways - reversing a metric
            reverses both the field and its idea of downwind, so leaving this
            to default would hide the very sign change being looked for.

        Returns:
        --------
        result : dict
            downwind and upwind mean progress change, their separation, the
            same three in seconds, and the cell counts behind them
        """
        delta = self.test.progress - self.control.progress
        ux, uy = np.asarray(self.heading if heading is None else heading,
                            dtype=float).reshape(2)
        X, Y = self.grid.meshgrid()
        start = self.test.mask_start
        cx, cy = float(X[start].mean()), float(Y[start].mean())
        projection = (X - cx) * ux + (Y - cy) * uy

        band = self.band & np.isfinite(delta)
        downwind = band & (projection > 0)
        upwind = band & (projection < 0)

        # A fire that grew only downwind of where it started has no upwind
        # half to compare - common under a steady wind, and a result in its
        # own right rather than a failure. The empty side reads NaN and the
        # note says why.
        nan = float("nan")
        d_mean = float(delta[downwind].mean()) if downwind.any() else nan
        u_mean = float(delta[upwind].mean()) if upwind.any() else nan
        note = None
        if not downwind.any() and not upwind.any():
            note = "no growth between the perimeters"
        elif not upwind.any():
            note = ("all growth lies downwind of the initial perimeter's centroid; "
                    "there is no upwind advance for the metric to delay")
        elif not downwind.any():
            note = ("all growth lies upwind of the initial perimeter's centroid, "
                    "against the heading - check the ROS vector's direction")
        interval = self.test.interval
        return {
            "downwind_mean": d_mean,
            "upwind_mean": u_mean,
            "separation": u_mean - d_mean,
            "downwind_mean_s": d_mean * interval,
            "upwind_mean_s": u_mean * interval,
            "separation_s": (u_mean - d_mean) * interval,
            "downwind_cells": int(downwind.sum()),
            "upwind_cells": int(upwind.sum()),
            "note": note,
        }

    # ------------------------------------------------------------------
    # Extent
    # ------------------------------------------------------------------

    def frame_agreement(self, num_steps=None):
        """
        Per-frame agreement between the two runs' intermediate fronts.

        The Jaccard index answers "how much of the burned area do they agree
        on"; `offset_m` answers "how far apart are the fronts", which is the
        number a reader can judge against the grid resolution and against how
        far the fire moved. The first and last frames are pinned by C2 and
        agree by construction, so the interior frames are the measurement.

        Parameters:
        -----------
        num_steps : int, optional
            Number of intervals; num_steps + 1 frames are compared

        Returns:
        --------
        frames : list of dict
            level, jaccard, offset_m, and the two areas, one per frame
        """
        num_steps = int(num_steps if num_steps is not None else self.num_steps)
        levels = np.linspace(0.0, 1.0, num_steps + 1)
        test_frames = self.test.frames(num_steps)
        control_frames = self.control.frames(num_steps)
        cell_area = self.grid.cell_area

        rows = []
        for level, a, b in zip(levels, test_frames, control_frames):
            test_area, control_area = float(a.sum()), float(b.sum())
            jaccard = _jaccard(a, b)
            # The two runs can disagree by burning different amounts, or by
            # burning the same amount in different places. The first sets a
            # floor on the second: J <= min(area)/max(area) always, with
            # equality when one frame nests inside the other. So the size
            # share is forced and whatever disagreement is left over is
            # genuine displacement.
            largest = max(test_area, control_area)
            size = (abs(test_area - control_area) / largest) if largest else 0.0
            rows.append({
                "level": float(level),
                "jaccard": jaccard,
                "disagreement": 1.0 - jaccard,
                "size_share": size,
                "placement_share": max(1.0 - jaccard - size, 0.0),
                "offset_m": front_offset(a, b, self.grid),
                "test_area_m2": test_area * cell_area,
                "control_area_m2": control_area * cell_area,
            })
        return rows

    def extent_difference(self, num_steps=None):
        """
        Whether the two runs burn different cells, different amounts, or both.

        The write-up predicts a redistribution of timing rather than a change
        of extent: with both perimeters pinned, a bulk vector is said to have
        limited room to change how much has burned at a given progress level,
        only which cells those are. That prediction is a claim about how the
        disagreement between the two runs decomposes, so it is measured as a
        decomposition rather than asserted.

        At each frame the total disagreement, 1 - Jaccard, splits into

            size_share       |A_test - A_control| / max(A_test, A_control)
            placement_share  whatever is left over

        and the split is exhaustive, because J <= min(area)/max(area) always,
        with equality exactly when one frame nests inside the other. A run
        dominated by `placement_share` is redistributing timing, as predicted.
        A run dominated by `size_share` is not - the two runs are producing
        nested fronts and one has simply got further by that level, which is
        a change of extent between the pinned endpoints.

        Both outcomes occur in practice, so `dominant` says which this run is
        rather than the caller having to assume.

        Returns:
        --------
        result : dict
            frame counts, the mean and worst disagreement and its two shares,
            the dominant share, and the front offset in metres
        """
        rows = self.frame_agreement(num_steps)
        interior = [r for r in rows if 0.0 < r["level"] < 1.0]
        if not interior:
            raise ValueError("no interior frames to compare; increase num_steps")

        worst = max(interior, key=lambda r: r["disagreement"])
        mean_size = float(np.mean([r["size_share"] for r in interior]))
        mean_placement = float(np.mean([r["placement_share"] for r in interior]))
        return {
            "frames": len(interior),
            "mean_jaccard": float(np.mean([r["jaccard"] for r in interior])),
            "mean_disagreement": float(np.mean([r["disagreement"] for r in interior])),
            "mean_size_share": mean_size,
            "mean_placement_share": mean_placement,
            "dominant": "placement" if mean_placement > mean_size else "size",
            "min_jaccard": worst["jaccard"],
            "min_jaccard_level": worst["level"],
            "max_size_share": float(max(r["size_share"] for r in interior)),
            "max_placement_share": float(max(r["placement_share"] for r in interior)),
            "mean_offset_m": float(np.mean([r["offset_m"] for r in interior])),
            "max_offset_m": float(max(r["offset_m"] for r in interior)),
        }

    def advance_ratios(self, frame_counts=None, min_advance_cells=2.0):
        """
        The C6 head/back advance ratio for both runs, over a ladder of frame counts.

        A ladder rather than a number, for two reasons. The measurement has a
        resolution floor - backing advance per frame is head advance over HB,
        so at a large HB it takes few, large frames before the upwind side
        moves by more than a cell - and the number it produces on a *pinned*
        pair turns out not to be stable across the choice.

        That instability is the useful part. On a point ignition in free space
        this ratio is exactly HB, which is what makes it a check on the
        anisotropy. On an observed pair it is not: both endpoints are fixed
        observations, so the upwind and downwind extents at s = 0 and s = 1
        are the perimeters' own, and the ratio mostly measures how lopsidedly
        the fire happened to grow. The control column is the evidence for that
        reading - a Euclidean metric has no preferred direction at all, so any
        departure from 1 in that column is the perimeters and not the method.
        Read the test-over-control quotient, not the test column, and read the
        spread down the ladder as the error bar on it.

        Parameters:
        -----------
        frame_counts : sequence of int, optional
            Frame counts to measure at, coarsening. Defaults to the run's own
            count halved repeatedly down to a single interval.
        min_advance_cells : float
            Backing advance a frame pair must show, in cells, to be counted

        Returns:
        --------
        result : dict
            hb, the ladder of (num_steps, test, control, quotient) rows, and
            `resolved` - the coarsest row where both columns resolved, or None
        """
        if frame_counts is None:
            frame_counts, n = [], self.num_steps
            while n >= 1:
                frame_counts.append(int(n))
                n //= 2
        heading = self.heading

        ladder = []
        for n in dict.fromkeys(int(n) for n in frame_counts if int(n) >= 1):
            test = advance_ratio(self.test.frames(n), self.grid, heading,
                                 min_advance_cells=min_advance_cells)
            control = advance_ratio(self.control.frames(n), self.grid, heading,
                                    min_advance_cells=min_advance_cells)
            quotient = (test / control if np.isfinite(test) and np.isfinite(control)
                        and control != 0 else float("nan"))
            ladder.append({"num_steps": n, "test": test, "control": control,
                           "quotient": quotient})

        resolved = next((row for row in reversed(ladder)
                         if np.isfinite(row["quotient"])), None)
        return {"hb": self.test.metric.ellipse.hb, "ladder": ladder,
                "resolved": resolved}

    def summary(self, num_steps=None):
        """
        Every number the comparison produces, in one dict.

        Returns:
        --------
        summary : dict
            metric provenance, both runs' own diagnostics, and the gap between
            them in timing, extent and advance ratio
        """
        return {
            "metric": self.test.metric.describe(),
            "metric_provenance": self.metric_provenance,
            "control": self.control_metric.describe(),
            "grid": repr(self.grid),
            "initial_outside_final": self.initial_outside_final,
            "test_summary": self.test.summary(),
            "control_summary": self.control.summary(),
            "progress_difference": self.progress_difference(),
            "redistribution": self.redistribution(),
            "extent_difference": self.extent_difference(num_steps),
            "advance_ratios": self.advance_ratios(),
            "implied_ros": implied_ros(self.test),
        }

    def report(self, num_steps=None):
        """
        The summary as a block of text, for pasting into a run log.

        Returns:
        --------
        text : str
        """
        s = self.summary(num_steps)
        pd, rd = s["progress_difference"], s["redistribution"]
        ed, ar, ir = s["extent_difference"], s["advance_ratios"], s["implied_ros"]
        if pd.get("cells", 0) == 0:
            return ("Elliptical versus Euclidean, one perimeter pair\n"
                    f"  metric          {s['metric']}\n"
                    "  the perimeters show no growth between them; there is "
                    "nothing to compare")
        interval = pd.get("interval_s", float("nan"))

        lines = [
            "Elliptical versus Euclidean, one perimeter pair",
            "=" * 62,
            f"  grid            {s['grid']}",
            f"  metric          {s['metric']}",
            f"  provenance      {s['metric_provenance']}",
            f"  control         {s['control']}",
            f"  interval        {interval:.0f} s ({interval / 3600.0:.2f} h)",
            f"  band            {pd['cells']} cells being interpolated",
        ]
        if self.initial_outside_final > 0.01:
            lines += [
                f"  WARNING         {100 * self.initial_outside_final:.0f}% of the initial "
                "perimeter lies outside the final one.",
                "                  The pair is not nested. If these are growth patches rather",
                "                  than footprints, convert with kmz_to_shapefile --mode additive;",
                "                  every number below assumes footprints.",
            ]
        lines += [
            "",
            "Endpoints and monotonicity (C2, C7) - both runs",
            "-" * 62,
        ]
        for name, run in (("elliptical", s["test_summary"]),
                          ("euclidean ", s["control_summary"])):
            lines.append(
                f"  {name}      jaccard {run['endpoint_jaccard_start']:.4f} / "
                f"{run['endpoint_jaccard_end']:.4f}   "
                f"violations {run['monotonicity_violations']}   "
                f"overshoot {run['overshoot']:.3g}")

        lines += [
            "",
            "Timing: does the vector move the arrival times?",
            "-" * 62,
            f"  mean |change|   {pd['mean_abs']:.4f} progress"
            f"   = {pd['mean_abs_s'] / 60.0:.1f} min",
            f"  rms             {pd['rms']:.4f}",
            f"  max |change|    {pd['max_abs']:.4f} progress"
            f"   = {pd['max_abs_s'] / 60.0:.1f} min",
            f"  signed mean     {pd['mean_signed']:+.4f} progress"
            f"   (a uniform offset carries no direction)",
            "",
            "  redistribution (negative downwind = burns earlier)",
            f"    downwind      {_signed(rd['downwind_mean'])}"
            f"   = {_signed(rd['downwind_mean_s'] / 60.0, 1)} min"
            f"   ({rd['downwind_cells']} cells)",
            f"    upwind        {_signed(rd['upwind_mean'])}"
            f"   = {_signed(rd['upwind_mean_s'] / 60.0, 1)} min"
            f"   ({rd['upwind_cells']} cells)",
            f"    separation    {_signed(rd['separation'])}"
            f"   = {_signed(rd['separation_s'] / 60.0, 1)} min",
        ]
        if rd.get("note"):
            lines.append(f"    note: {rd['note']}")
        lines += [
            "",
            "Extent: does it move the fronts, or resize them?",
            "-" * 62,
            f"  interior frames {ed['frames']}",
            f"  mean jaccard    {ed['mean_jaccard']:.4f}",
            f"  min jaccard     {ed['min_jaccard']:.4f} at s = {ed['min_jaccard_level']:.2f}",
            f"  disagreement    {ed['mean_disagreement']:.4f} mean, splitting into",
            f"    size          {ed['mean_size_share']:.4f}"
            f"   (different amounts burned by the same level)",
            f"    placement     {ed['mean_placement_share']:.4f}"
            f"   (same amount, different cells)",
            f"  dominant        {ed['dominant']}"
            f"   ({'redistribution, as predicted' if ed['dominant'] == 'placement' else 'a change of extent, not the predicted redistribution'})",
            f"  front offset    {ed['mean_offset_m']:.1f} m mean, "
            f"{ed['max_offset_m']:.1f} m max"
            f"   ({ed['mean_offset_m'] / self.grid.cell_size:.1f} cells mean)",
            "",
            "Head/back advance ratio (C6)",
            "-" * 62,
            f"  ellipse HB      {ar['hb']:.3f}   (free-space value, not a target"
            f" on a pinned pair)",
            "  frames   elliptical   euclidean   quotient",
        ]
        for row in ar["ladder"]:
            lines.append(
                f"  {row['num_steps']:>6}   {_fmt(row['test']):>10}  "
                f" {_fmt(row['control']):>9}   {_fmt(row['quotient']):>8}")
        lines += [
            "  a euclidean column far from 1 means the ratio is reading the",
            "  perimeters' own lopsidedness, not the metric",
            "",
            "Implied versus observed ROS",
            "-" * 62,
            f"  head transit    {ir['transit_m']:.0f} m over {ir['interval_s']:.0f} s",
            f"  implied         {ir['implied_mps']:.4f} m/s",
            f"  observed |v|    {ir['observed_mps']:.4f} m/s",
            f"  ratio           {ir['ratio']:.3f}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (f"MetricComparison({self.test.metric.describe()} vs "
                f"{self.control_metric.describe()}, {self.grid!r})")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def front_offset(a, b, grid):
    """
    Mean distance in metres between the boundaries of two masks.

    Each cell the two masks disagree about is scored by how far past the other
    mask's front it lies, and the scores are averaged. Unlike one minus the
    Jaccard index this has units, so it can be read against the cell size and
    against how far the fire actually moved.

    The signed distance functions are taken with subcell=True, because the
    plain form inflates every magnitude by about half a cell and this is a
    measurement of small distances near a boundary - exactly where that bias
    is the largest share of the answer.

    Parameters:
    -----------
    a, b : numpy.ndarray
        Boolean masks of the grid's shape
    grid : GridSpec
        The shared computational grid

    Returns:
    --------
    offset : float
        Mean front separation in metres; zero when the masks are identical
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    only_a, only_b = a & ~b, b & ~a
    if not (only_a.any() or only_b.any()):
        return 0.0
    if not a.any() or a.all() or not b.any() or b.all():
        # One mask has no boundary, so there is no separation to measure.
        return float("nan")

    phi_a = np.abs(grid.sdf(a, subcell=True))
    phi_b = np.abs(grid.sdf(b, subcell=True))
    values = np.concatenate([phi_b[only_a], phi_a[only_b]])
    return float(values.mean())


def _fmt(value):
    """Three decimals, or a dash where the backing advance did not resolve."""
    return "-" if not np.isfinite(value) else f"{value:.3f}"


def _signed(value, decimals=4):
    """Signed fixed-point, or a dash where there was nothing to measure."""
    return "-" if not np.isfinite(value) else f"{value:+.{decimals}f}"


def _jaccard(a, b):
    """Intersection over union of two boolean masks."""
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def main(argv=None):
    """Run the comparison on a perimeter pair, or on every pair of a series."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare the elliptical metric against a Euclidean control "
                    "on real perimeters, and report by how much the bulk ROS "
                    "vector changes the answer. Two files: the full report for "
                    "that pair. More: one line per consecutive pair, plus the "
                    "held-out test on every triplet.")
    parser.add_argument("perimeters", nargs="+",
                        help="perimeter shapefiles in time order")
    parser.add_argument("--cell-size", type=float, default=25.0,
                        help="grid resolution in metres (default 25)")
    parser.add_argument("--padding", type=float, default=0.1,
                        help="fraction of the extent to pad the grid by")
    parser.add_argument("--lb", type=float, default=None,
                        help="length-to-breadth ratio, if the perimeters carry none")
    parser.add_argument("--wind-mps", type=float, default=None,
                        help="midflame wind speed, to supply LB via Finney Eq. 13")
    parser.add_argument("--infer-from-geometry", action="store_true",
                        help="estimate the ROS vector from the pair when neither "
                             "perimeter carries one and none can be borrowed")
    parser.add_argument("--no-borrow", action="store_true",
                        help="do not borrow a vector from the nearest observation "
                             "when a pair has none")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="frames to compare across (default 20)")
    parser.add_argument("--hold-out", type=int, default=None,
                        help="score only this observation's triplet instead of all")
    parser.add_argument("--full", action="store_true",
                        help="print the full report for every pair, not one line each")
    args = parser.parse_args(argv)

    from perimeter import PerimeterSeries

    series = PerimeterSeries.from_shapefiles(args.perimeters)
    grid = series.common_grid(cell_size=args.cell_size, padding=args.padding)
    series = series.on_grid(grid)
    n_vectors = sum(1 for p in series if p.has_ros)
    print(f"  {len(series)} perimeters, {n_vectors} carrying a ROS vector, "
          f"{grid!r}")

    def metric_for(k):
        return series_metric(series, k, lb=args.lb, wind_mps=args.wind_mps,
                             infer_from_geometry=args.infer_from_geometry,
                             grid=grid, borrow=not args.no_borrow)

    comparisons = []
    for k, (a, b) in enumerate(series.pairs()):
        metric, provenance = metric_for(k)
        if not metric.is_anisotropic:
            print(f"  pair {k} ({a.time:%H:%M:%SZ} -> {b.time:%H:%M:%SZ}): skipped, "
                  f"{provenance}")
            comparisons.append(None)
            continue
        comparison = MetricComparison(a, b, grid, metric=metric, num_steps=args.num_steps)
        comparison.metric_provenance = provenance
        comparisons.append(comparison)
        if len(series) == 2 or args.full:
            print("")
            print(comparison.report())

    if len(series) > 2:
        _print_pair_table(series, comparisons)

    if len(series) >= 3:
        indices = ([args.hold_out] if args.hold_out is not None
                   else range(1, len(series) - 1))
        _print_held_out(series, grid, indices, comparisons, metric_for, args.num_steps)
    return 0


def _print_pair_table(series, comparisons):
    """One line per consecutive pair: the numbers that summarise a report."""
    print("")
    print("Per pair: elliptical against Euclidean")
    print("-" * 96)
    print(f"  {'pair':>4}  {'from':>9} {'to':>9}  {'min':>5}  {'mean|ds|':>8}  "
          f"{'offset m':>8}  {'size':>6} {'place':>6}  {'impl/obs':>8}  metric")
    for k, comparison in enumerate(comparisons):
        a, b = series[k], series[k + 1]
        stamp = f"{a.time:%H:%M:%SZ} {b.time:%H:%M:%SZ}"
        if comparison is None:
            print(f"  {k:>4}  {stamp}  {a.interval_to(b) / 60:5.1f}  skipped (isotropic)")
            continue
        pd = comparison.progress_difference()
        ed = comparison.extent_difference()
        ir = implied_ros(comparison.test)
        borrowed = "borrowed" if "borrowed" in comparison.metric_provenance else "own"
        warn = (f"  NOT NESTED {100 * comparison.initial_outside_final:.0f}%"
                if comparison.initial_outside_final > 0.01 else "")
        print(f"  {k:>4}  {stamp}  {pd['interval_s'] / 60:5.1f}  {pd['mean_abs']:8.4f}  "
              f"{ed['mean_offset_m']:8.1f}  {ed['mean_size_share']:6.3f} "
              f"{ed['mean_placement_share']:6.3f}  {ir['ratio']:8.3f}  "
              f"{comparison.test.metric.ellipse.head:.2f} m/s @ "
              f"{comparison.test.metric.ellipse.bearing:.0f} ({borrowed}){warn}")
    print("  mean|ds| is the mean change in progress over the band; offset the mean "
          "front separation;")
    print("  size/place the disagreement split; impl/obs the implied over observed head rate.")


def _print_held_out(series, grid, indices, comparisons, metric_for, num_steps):
    """
    The held-out test on each triplet (i-1, i, i+1), both metrics.

    Evolving across the neighbours rather than the whole series keeps each
    test at the scale of a real gap - two intervals - and gives one score
    per interior observation instead of one for the series.
    """
    from metrics import EuclideanMetric, EllipticalMetric
    from perimeter import PerimeterSeries

    print("")
    print("Held-out: evolve i-1 -> i+1, score the withheld i (Jaccard, skill over the best trivial guess)")
    print("-" * 96)
    print(f"  {'i':>3}  {'withheld':>9}  {'span':>5}  {'elliptical':>18}  {'euclidean':>18}  "
          f"{'static':>6} {'satur.':>6}  metric")
    scores = []
    for i in indices:
        if not 0 < i < len(series) - 1:
            print(f"  {i:>3}  not an interior observation; nothing to withhold")
            continue
        triplet = PerimeterSeries([series[i - 1], series[i], series[i + 1]])
        # The metric for the outer pair i-1 -> i+1: the same rule as
        # everywhere else, the vector of the perimeter the interval ends at,
        # which is what the comparison for pair i already resolved (own or
        # borrowed). Only if that pair was isotropic is pair i-1's used.
        ellipse = next((c.test.metric.ellipse for c in (comparisons[i], comparisons[i - 1])
                        if c is not None), None)
        if ellipse is None:
            print(f"  {i:>3}  {series[i].time:%H:%M:%SZ}  skipped: no anisotropic metric either side")
            continue
        metric = EllipticalMetric(ellipse)
        span = series[i - 1].interval_to(series[i + 1]) / 60.0
        e = hold_out(triplet, grid, index=1, metric=metric, num_steps=num_steps)
        u = hold_out(triplet, grid, index=1, metric=EuclideanMetric(), num_steps=num_steps)
        scores.append((e["jaccard"], u["jaccard"], e["skill"], u["skill"]))
        print(f"  {i:>3}  {series[i].time:%H:%M:%SZ}  {span:5.1f}  "
              f"{e['jaccard']:7.4f} ({e['skill']:+.3f})  {u['jaccard']:7.4f} ({u['skill']:+.3f})  "
              f"{e['static']:6.3f} {e['saturated']:6.3f}  "
              f"{metric.ellipse.head:.2f} m/s @ {metric.ellipse.bearing:.0f}")
    if len(scores) > 1:
        arr = np.array(scores)
        wins = int((arr[:, 0] > arr[:, 1]).sum())
        print(f"  mean  elliptical {arr[:, 0].mean():.4f} ({arr[:, 2].mean():+.3f})   "
              f"euclidean {arr[:, 1].mean():.4f} ({arr[:, 3].mean():+.3f})   "
              f"elliptical better on {wins} of {len(scores)}")
    print("  skill = (jaccard - best trivial guess) / (1 - best trivial guess); "
          "static = nothing moved, satur. = everything burned at once")


if __name__ == "__main__":
    raise SystemExit(main())
