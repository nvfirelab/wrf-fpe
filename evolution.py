"""
Module evolving one observed perimeter into another under a chosen metric.

Four stages:

    A   rasterise both observed perimeters onto one grid in metres and convert
        each to a signed distance function                       (grid.py)
    B   build the wavelet from the observed ROS vector           (metrics.py)
    C   two arrival-time solves - outward from the initial perimeter under the
        metric, inward from the final one under the reversed metric, each a
        direct gauge minimisation rather than an integration      (gauge.py)
    D   blend and normalise                                      (clock.py)

Stage D is the whole method in one line:

    s(x) = T_f(x) / [ T_f(x) + T_b(x) ]

with the endpoints following structurally rather than by fitting. Converting
to seconds is then a single multiplication by the observed interval.
"""

import numpy as np

from clock import TwoSidedClock
from metrics import Source, resolve_metric


class GaugeEvolution:
    """
    Evolve an initial perimeter into a final one and report the progress field.

    The two knobs are constructor arguments. The metric decides whether the
    intermediate spread looks physical; the clock decides endpoint exactness.
    Setting both well is not a compromise between the two prototypes - they
    are independent.

    Fields are computed lazily and cached, so constructing the object is cheap
    and the first access to `progress` pays for the two solves.
    """

    def __init__(self, initial, final, grid, metric=None, clock=None,
                 num_steps=20, vertex_source="geometry", vertex_spacing=None,
                 restrict_to_final=True, lb=None, wind_mps=None,
                 infer_from_geometry=False):
        """
        Parameters:
        -----------
        initial, final : Perimeter
            The two observations. Both are projected onto the grid's CRS.
        grid : GridSpec
            The shared computational grid, in metres
        metric : Metric, optional
            What "distance" means. When omitted it is resolved from what the
            observations carry - the ellipse where a ROS vector and an LB are
            both available, Euclidean otherwise - and `metric_provenance`
            records which, and why.
        clock : Clock, optional
            How travel time becomes progress. Defaults to the two-sided clock,
            which is what makes the endpoints exact.
        num_steps : int
            Default number of frames requested from `frames()`
        vertex_source : str
            "geometry" or "mask", passed to Perimeter.boundary_vertices
        vertex_spacing : float, optional
            Spacing of geometry-sampled source vertices, in metres
        restrict_to_final : bool
            Solve only for cells inside the final perimeter. Cells outside it
            never ignite and are masked out of the progress field anyway, so
            this is exact rather than an approximation, and it is usually the
            largest single speed-up available.
        lb : float, optional
            Length-to-breadth ratio, when the metric is being resolved
        wind_mps : float, optional
            Midflame wind speed for Finney Eq. 13, used to supply LB
        infer_from_geometry : bool
            Allow a ROS vector to be inferred from the perimeter pair when
            neither observation carries one. Off by default: a vector inferred
            from the same pair being interpolated cannot tell you anything the
            endpoints did not already imply, so it should be an explicit
            choice rather than a silent one.
        """
        self.grid = grid
        self.initial = initial.to_crs(grid.crs)
        self.final = final.to_crs(grid.crs)
        if metric is not None:
            self.metric = metric
            self.metric_provenance = f"as given: {metric.describe()}"
        else:
            self.metric, self.metric_provenance = resolve_metric(
                self.initial, self.final, grid, lb=lb, wind_mps=wind_mps,
                infer_from_geometry=infer_from_geometry)
        self.clock = clock if clock is not None else TwoSidedClock()
        self.num_steps = int(num_steps)
        self.vertex_source = vertex_source
        self.vertex_spacing = vertex_spacing
        self.restrict_to_final = bool(restrict_to_final)

        self._cache = {}
        self._violations = None

    # ------------------------------------------------------------------
    # Stage A: the two observations on the grid
    # ------------------------------------------------------------------

    @property
    def mask_start(self):
        """The initial perimeter rasterised onto the grid."""
        return self.initial.mask(self.grid)

    @property
    def mask_end(self):
        """The final perimeter rasterised onto the grid."""
        return self.final.mask(self.grid)

    @property
    def interval(self):
        """Seconds between the two observations."""
        return self.initial.interval_to(self.final)

    def _source(self, perimeter, direction):
        """Boundary vertices and mask of one perimeter, as a solve's source."""
        sx, sy = perimeter.boundary_vertices(self.grid, source=self.vertex_source,
                                             spacing=self.vertex_spacing)
        return Source(sx, sy, mask=perimeter.mask(self.grid), direction=direction)

    # ------------------------------------------------------------------
    # Stage C: the two solves
    # ------------------------------------------------------------------

    @property
    def _where(self):
        """Cells the solves are computed for."""
        return self.mask_end if self.restrict_to_final else None

    @property
    def T_f(self):
        """
        Arrival time spreading outward from the initial perimeter, in seconds.

        The journey already travelled to reach each cell.
        """
        if "T_f" not in self._cache:
            self._cache["T_f"] = self.metric.arrival(
                self._source(self.initial, "outward"), self.grid, where=self._where)
        return self._cache["T_f"]

    @property
    def T_b(self):
        """
        Arrival time spreading inward from the final perimeter, in seconds.

        The journey remaining from each cell to the final perimeter, solved
        under the reversed metric because receding traverses the same paths
        backwards. Getting the reversal wrong does not throw - it produces a
        plausible-looking field with the wrong intermediate ordering, which is
        what the monotonicity count is there to catch.
        """
        if "T_b" not in self._cache:
            self._cache["T_b"] = self.metric.reversed().arrival(
                self._source(self.final, "inward"), self.grid, where=self._where)
        return self._cache["T_b"]

    # ------------------------------------------------------------------
    # Stage D: blend and normalise
    # ------------------------------------------------------------------

    @property
    def progress_raw(self):
        """
        The clock's progress field, before the endpoint overrides.

        Kept because it is what the clocks can be compared on: the two-sided
        clock cannot leave [0, 1], while the one-sided clock overshoots inside
        the final perimeter, which is exactly its failure.
        """
        if "progress_raw" not in self._cache:
            self._cache["progress_raw"] = self.clock.progress(
                self.T_f, self.T_b, self.grid, self.mask_start, self.mask_end)
        return self._cache["progress_raw"]

    @property
    def progress(self):
        """
        Normalised fire arrival time: 0 on the initial perimeter, 1 on the final.

        Dimensionless progress on [0, 1], NaN outside the final perimeter for
        cells that never ignite. Everything already burned at the first
        observation is set to 0 - it did not arrive during this interval.

        Returns:
        --------
        s : numpy.ndarray
            Progress field of shape grid.shape
        """
        if "progress" not in self._cache:
            s = np.array(self.progress_raw, dtype=float, copy=True)
            s[self.mask_start] = 0.0
            s[~self.mask_end] = np.nan
            self._cache["progress"] = s
        return self._cache["progress"]

    def arrival_seconds(self, t0=0.0):
        """
        Progress converted to seconds, ready for export.

        Parameters:
        -----------
        t0 : float
            Seconds from the simulation start to the initial observation

        Returns:
        --------
        t_ign : numpy.ndarray
            Ignition time in seconds, NaN outside the final perimeter
        """
        return t0 + self.progress * self.interval

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def frames(self, num_steps=None):
        """
        Burned-area masks at evenly spaced progress levels.

        The running union enforces monotonicity, but violations are counted
        before it rather than silently absorbed: a nonzero count is a signal
        that the two solves disagree, not a routine correction.

        Parameters:
        -----------
        num_steps : int, optional
            Number of intervals; num_steps + 1 frames are returned, from
            s = 0 to s = 1. Defaults to the constructor's num_steps.

        Returns:
        --------
        frames : list of numpy.ndarray
            Cumulative boolean burned-area masks
        """
        num_steps = int(num_steps if num_steps is not None else self.num_steps)
        s = self.progress
        burned = np.zeros(self.grid.shape, dtype=bool)
        frames, violations = [], 0
        for level in np.linspace(0.0, 1.0, num_steps + 1):
            reached = (s <= level) & self.mask_end
            violations += int((burned & ~reached).any())
            burned = burned | reached
            frames.append(burned.copy())
        self._violations = violations
        return frames

    @property
    def monotonicity_violations(self):
        """
        Frames that lost burned area before the running union restored it.

        Should be zero. Nonzero usually means the metric was not reversed
        consistently between the two solves.
        """
        if self._violations is None:
            self.frames()
        return self._violations

    def contour_polygon(self, level):
        """
        The burned area at one progress level, as a polygon.

        Parameters:
        -----------
        level : float
            Progress level on [0, 1]

        Returns:
        --------
        geometry : shapely geometry
            Burned area in the grid's CRS
        """
        return self.grid.mask_to_polygon((self.progress <= level) & self.mask_end)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def endpoint_jaccard(self):
        """
        Agreement of the s = 0 and s = 1 burned areas with the observations.

        The s = 0 value is exact by construction - everything already burned
        is forced to zero - so it checks the plumbing rather than the method.
        The s = 1 value is the one that discriminates: under the two-sided
        clock s cannot exceed 1, so the final frame recovers the observed
        perimeter exactly, while a clock that overshoots leaves cells inside
        the final perimeter unburned at s = 1.

        Returns:
        --------
        start, end : float
            Jaccard index of each endpoint frame against its observation
        """
        s = self.progress
        first = (s <= 0.0) & self.mask_end
        last = (s <= 1.0) & self.mask_end
        return (_jaccard(first, self.mask_start & self.mask_end),
                _jaccard(last, self.mask_end))

    def shortest_journey(self):
        """
        The shortest total travel time through any cell inside the final perimeter.

        T_f + T_b is the journey through a cell, and s is the fraction of it
        completed. Where the two perimeters nearly touch the journey is short,
        so s swings from 0 to 1 over a handful of cells: still continuous, but
        steep, and carrying little information because the fire barely moved
        there. A value of a few cell widths is a signal that part of the
        perimeter pair is under-determined, not that the solve went wrong.

        Returns:
        --------
        journey : float
            Shortest journey in the metric's own units (seconds, or metres for
            a Euclidean metric at 1 m/s)
        """
        total = (self.T_f + self.T_b)[self.mask_end]
        total = total[np.isfinite(total)]
        return float(total.min()) if total.size else float("nan")

    def overshoot(self):
        """
        How far past 1 the clock's progress runs in the region being interpolated.

        Measured between the two perimeters - inside the final one, outside
        the initial one - because that is the region the method is answering
        for. Cells already burned at the first observation are forced to zero
        regardless of what the clock said about them, so including them would
        report a number the output never shows.

        Zero for the two-sided clock, structurally: T_f and T_b are both
        non-negative, so the ratio cannot leave [0, 1]. Positive for the
        one-sided clock, which is why its s = 1 contour lands inside the
        observed perimeter rather than on it.
        """
        band = self.mask_end & ~self.mask_start
        values = self.progress_raw[band]
        values = values[np.isfinite(values)]
        return float(max(values.max() - 1.0, 0.0)) if values.size else 0.0

    def head_back_advance_ratio(self, num_steps=None, heading=None,
                                min_advance_cells=2.0):
        """
        Mean ratio of downwind to upwind advance across the frames.

        C6 made measurable. Between consecutive frames the front's extent is
        taken along the heading and against it, and the two increments are
        divided. For a point ignition in a homogeneous medium this is exactly
        the head-to-back ratio HB, which is what makes it a usable check on
        whether the anisotropy is actually reaching the output.

        Extent is used rather than mean normal advance per half-front, which
        sounds equivalent and is not: as LB grows the head becomes a nose that
        is a vanishing fraction of the front's length, so an area-over-length
        average is dominated by flank behaviour on both sides and compresses
        towards 1 - it reads 4.6 where HB is 34. Centroid drift is worse
        still, being near zero between nested fronts even when the anisotropy
        is working.

        The measurement has a resolution floor. Backing advance per frame is
        the head advance divided by HB, so resolving it needs cells smaller
        than that; frame pairs that do not clear `min_advance_cells` are
        skipped rather than contributing a quantised ratio, and the result is
        NaN if none of them do. Fewer, larger frames or a finer grid is the
        fix.

        Parameters:
        -----------
        num_steps : int, optional
            Number of frames to measure across
        heading : array-like, optional
            Unit heading. Taken from the metric's ellipse when it has one.
        min_advance_cells : float
            Backing advance a frame pair must show, in cells, to be counted

        Returns:
        --------
        ratio : float
            Mean downwind-to-upwind advance ratio, or NaN if none resolved
        """
        if heading is None:
            ellipse = getattr(self.metric, "ellipse", None)
            if ellipse is None:
                raise ValueError(
                    "this metric has no heading; pass heading= to measure "
                    "advance against a direction of your choosing")
            heading = ellipse.uhat
        return advance_ratio(self.frames(num_steps), self.grid, heading,
                             min_advance_cells=min_advance_cells)

    def summary(self):
        """
        Diagnostics worth reporting alongside a run.

        Returns:
        --------
        summary : dict
            Grid, metric and clock provenance, endpoint agreement, overshoot,
            and the monotonicity violation count
        """
        start_j, end_j = self.endpoint_jaccard()
        s = self.progress
        finite = s[np.isfinite(s)]
        return {
            "grid": repr(self.grid),
            "metric": self.metric.describe(),
            "metric_provenance": self.metric_provenance,
            "clock": self.clock.describe(),
            "interval_s": self.interval if (self.initial.time and self.final.time) else None,
            "cells_solved": int(self.mask_end.sum()) if self.restrict_to_final
                            else self.grid.size,
            "source_vertices": (self._source(self.initial, "outward").n_vertices,
                                self._source(self.final, "inward").n_vertices),
            "progress_min": float(finite.min()) if finite.size else None,
            "progress_max": float(finite.max()) if finite.size else None,
            "shortest_journey": self.shortest_journey(),
            "overshoot": self.overshoot(),
            "endpoint_jaccard_start": start_j,
            "endpoint_jaccard_end": end_j,
            "monotonicity_violations": self.monotonicity_violations,
        }

    def __repr__(self):
        return (f"GaugeEvolution({self.metric.describe()}, {self.clock!r}, "
                f"{self.grid!r})")


def advance_ratio(frames, grid, heading, min_advance_cells=2.0):
    """
    Mean ratio of downwind to upwind advance across a sequence of frames.

    Between consecutive frames the burned area's extent is taken along the
    heading and against it, and the two increments are divided.

    Parameters:
    -----------
    frames : sequence of numpy.ndarray
        Cumulative burned-area masks, in order
    grid : GridSpec
        The shared computational grid
    heading : array-like
        Direction to measure advance along; need not be normalised
    min_advance_cells : float
        Backing advance a frame pair must show, in cells, to be counted

    Returns:
    --------
    ratio : float
        Mean advance ratio, or NaN when no frame pair resolves the backing
    """
    ux, uy = np.asarray(heading, dtype=float).reshape(2)
    norm = np.hypot(ux, uy)
    if norm <= 0:
        raise ValueError("heading must be a nonzero vector")
    ux, uy = ux / norm, uy / norm

    X, Y = grid.meshgrid()
    projection = X * ux + Y * uy
    floor = min_advance_cells * grid.cell_size

    ratios = []
    for previous, current in zip(frames[:-1], frames[1:]):
        if not previous.any() or not current.any():
            continue
        head = projection[current].max() - projection[previous].max()
        back = (-projection[current]).max() - (-projection[previous]).max()
        if back > floor:
            ratios.append(head / back)
    return float(np.mean(ratios)) if ratios else float("nan")


def _jaccard(a, b):
    """Intersection over union of two boolean masks."""
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0
