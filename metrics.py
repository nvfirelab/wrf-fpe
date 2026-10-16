"""
Module defining the metric: what "distance" means between two points.

This is the first of the method's two knobs. The metric controls whether the
intermediate spread looks physical; it does not control endpoint exactness,
which is structural and holds under any metric. Because the metric is a
parameter rather than a fixed part of the algorithm, the method degrades
gracefully when no spread data is available - and reducing it to Euclidean
reproduces the linear prototype exactly, which is the regression test that
catches almost every plumbing bug.

Three kinds matter:

    Euclidean               time = length / speed, the same in every direction
                            and at every location. EuclideanMetric.
    Isotropic, varying      speed depends on where you are but not which way
                            you face. Not built: a modelled speed map is the
                            fuel physics C4 rules out.
    Anisotropic             speed depends on direction as well. This is what a
                            ROS vector needs, since a single number per cell
                            has nowhere to put a bearing. EllipticalMetric.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from gauge import gauge_arrival
from wavelet import FireEllipse, lb_from_wind


@dataclass
class Source:
    """
    Where a solve spreads from, and which way.

    The gauge only needs the boundary vertices; the distance-transform backend
    needs the rasterised mask and the direction of travel as well, because it
    works from the signed distance function rather than from points.

    Attributes:
    -----------
    x, y : numpy.ndarray
        1D arrays of boundary vertex coordinates in metres
    mask : numpy.ndarray or None
        The source region rasterised onto the grid
    direction : str
        "outward" for a solve leaving the source, "inward" for one travelling
        into it. The gauge is indifferent - it measures from the boundary
        either way - but the sign of the distance transform is not.
    """

    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray = None
    direction: str = "outward"

    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=float).ravel()
        self.y = np.ascontiguousarray(self.y, dtype=float).ravel()
        if self.direction not in ("outward", "inward"):
            raise ValueError(f"unknown direction {self.direction!r}")

    @property
    def n_vertices(self):
        """Number of boundary vertices, which sets the cost of a gauge solve."""
        return self.x.size


class Metric(ABC):
    """
    A rule assigning a travel time to every displacement.

    Subclasses supply two things: how to solve for arrival time from a source,
    and what the metric looks like travelled backwards. The second matters
    because the method runs two solves - outward from the initial perimeter
    and inward from the final one - and the inward solve traverses the same
    paths in reverse.
    """

    @abstractmethod
    def arrival(self, source, grid, where=None):
        """
        Arrival time in seconds at every cell, spreading from a source.

        Parameters:
        -----------
        source : Source
            Where the solve spreads from
        grid : GridSpec
            The shared computational grid
        where : numpy.ndarray, optional
            Boolean mask of cells to solve for; the rest come back NaN

        Returns:
        --------
        T : numpy.ndarray
            Arrival time in seconds, shape grid.shape
        """

    @abstractmethod
    def reversed(self):
        """The same metric traversed backwards, for the inward solve."""

    @property
    def is_anisotropic(self):
        """Whether travel time depends on direction as well as distance."""
        return False

    def describe(self):
        """One-line human-readable summary, for provenance in reports."""
        return repr(self)


class EuclideanMetric(Metric):
    """
    Time is distance over a single speed, the same in every direction.

    This is the floor the method degrades to when no ROS vector is available,
    and it reproduces the linear prototype: with Euclidean distances the
    two-sided blend is algebraically phi_i / (phi_i - phi_f), which is the
    prototype's zero level set written out.

    Two backends compute the same thing:

        "gauge"  minimises the gauge over the source's boundary vertices, the
                 same code path the elliptical metric uses. Costs cells x
                 vertices, and its accuracy is set by the vertex spacing
                 rather than by the grid.
        "edt"    reads the distance straight off the signed distance function.
                 Costs O(cells), and reproduces the prototype's distances
                 exactly because the prototype computes them the same way.

    They differ inside the source region: the distance transform reports zero
    everywhere behind the front, while the gauge reports distance back to the
    boundary. The progress field overrides that region anyway, so the two
    agree everywhere the answer is used.
    """

    def __init__(self, speed=1.0, backend="gauge"):
        """
        Parameters:
        -----------
        speed : float
            Spread rate in metres per second. Any positive value gives the
            same progress field - it cancels in the two-sided ratio - so the
            default of 1.0 means travel times come out as metres.
        backend : str
            "gauge" or "edt"
        """
        if speed <= 0:
            raise ValueError(f"speed must be positive; got {speed}")
        if backend not in ("gauge", "edt"):
            raise ValueError(f"unknown backend {backend!r}")
        self.speed = float(speed)
        self.backend = backend

    def arrival(self, source, grid, where=None):
        if self.backend == "edt":
            return self._arrival_edt(source, grid, where)
        X, Y = grid.meshgrid()
        return gauge_arrival(source.x, source.y, (1.0, 0.0),
                             self.speed, self.speed, 0.0, X, Y, where=where)

    def _arrival_edt(self, source, grid, where):
        """Read the distance off the SDF instead of minimising over vertices."""
        if source.mask is None:
            raise ValueError("the edt backend needs the source's rasterised mask")
        phi = grid.sdf(source.mask)
        distance = np.maximum(phi if source.direction == "outward" else -phi, 0.0)
        T = distance / self.speed
        if where is not None:
            T = np.where(np.asarray(where, dtype=bool), T, np.nan)
        return T

    def reversed(self):
        """Euclidean distance is symmetric, so travelling it backwards is free."""
        return self

    def describe(self):
        return f"Euclidean, {self.speed:g} m/s ({self.backend} backend)"

    def __repr__(self):
        return f"EuclideanMetric(speed={self.speed!r}, backend={self.backend!r})"


class EllipticalMetric(Metric):
    """
    Travel time under the fire ellipse built from an observed ROS vector.

    This is the metric the method is for. A scalar speed map cannot represent
    a ROS vector - a single number per cell has nowhere to put a bearing - so
    direction-dependence is not an embellishment here, it is the only way the
    observation can enter at all.

    The metric is a Randers metric: a Riemannian metric plus a uniform drift,
    which is what breaks the symmetry so that downwind is cheaper than upwind.
    It is homogeneous - the same everywhere - and that is what turns the
    Eikonal equation from a numerical PDE into arithmetic. Dropping the
    modelled fuel map is what buys the homogeneity.
    """

    def __init__(self, ellipse):
        """
        Parameters:
        -----------
        ellipse : FireEllipse
            The wavelet: what one ignition becomes in unit time
        """
        if not isinstance(ellipse, FireEllipse):
            raise TypeError("EllipticalMetric needs a FireEllipse")
        self.ellipse = ellipse

    @classmethod
    def from_ros(cls, ros, lb):
        """
        Build the metric straight from a ROS vector and a shape parameter.

        Parameters:
        -----------
        ros : array-like
            (u, v) in metres per second, in the grid's axes
        lb : float
            Length-to-breadth ratio

        Returns:
        --------
        metric : EllipticalMetric
        """
        return cls(FireEllipse.from_ros(ros, lb))

    @classmethod
    def from_wind(cls, ros, u_midflame):
        """
        Build the metric with LB taken from midflame wind speed.

        Parameters:
        -----------
        ros : array-like
            (u, v) in metres per second
        u_midflame : float
            Midflame wind speed in metres per second, for Finney Eq. 13

        Returns:
        --------
        metric : EllipticalMetric
        """
        return cls(FireEllipse.from_ros(ros, lb_from_wind(u_midflame)))

    def arrival(self, source, grid, where=None):
        X, Y = grid.meshgrid()
        e = self.ellipse
        return gauge_arrival(source.x, source.y, e.uhat, e.a, e.b, e.c, X, Y,
                             where=where)

    def reversed(self):
        """
        The same ellipse with the heading negated.

        Negating the heading reverses the offset while leaving the shape
        unchanged, which is what "the reversed metric" means concretely.
        """
        return EllipticalMetric(self.ellipse.reversed())

    @property
    def is_anisotropic(self):
        return not self.ellipse.is_isotropic

    def describe(self):
        return self.ellipse.describe()

    def __repr__(self):
        return f"EllipticalMetric({self.ellipse!r})"


def infer_ros(mask_start, mask_end, grid, interval, quantile=0.99):
    """
    Estimate a bulk ROS vector from the perimeter pair alone.

    Direction comes from the displacement between the perimeters, taken at the
    cells of greatest normal advance rather than from the centroid shift,
    which is the more robust of the two. Magnitude is that advance divided by
    the interval.

    This makes the method applicable to any perimeter pair whatsoever, but it
    is worth being honest about what it can show: a vector inferred from the
    same pair being interpolated cannot tell you anything the endpoints did
    not already imply. It is a legitimate prior extracted from the data rather
    than circular fitting, which is why it is opt-in rather than the default.

    Parameters:
    -----------
    mask_start, mask_end : numpy.ndarray
        The two observed perimeters rasterised onto the grid
    grid : GridSpec
        The shared computational grid
    interval : float
        Seconds between the two observations
    quantile : float
        Advance quantile defining "greatest normal advance". 1.0 uses the
        single furthest cell, as the write-up specifies; the default of 0.99
        averages the furthest one per cent, which is the same quantity read
        off more than one cell.

    Returns:
    --------
    ros : numpy.ndarray
        (u, v) in metres per second, in the grid's axes
    """
    return infer_ros_detail(mask_start, mask_end, grid, interval, quantile)["ros"]


def infer_ros_detail(mask_start, mask_end, grid, interval, quantile=0.99):
    """
    `infer_ros`, with the working that locates the vector on the ground.

    Parameters:
    -----------
    As for `infer_ros`.

    Returns:
    --------
    detail : dict
        ros         (u, v) in m/s in the grid's axes - exactly `infer_ros`
        advance_m   mean advance of the selected cells (speed * interval)
        tail        (x, y): advance-weighted mean of the initial-perimeter
                    cells nearest the selected cells, where the growth left
                    the old edge
        front       (x, y): advance-weighted mean of the selected cells
        n_cells     how many cells were selected
    """
    from scipy.ndimage import distance_transform_edt

    mask_start = np.asarray(mask_start, dtype=bool)
    mask_end = np.asarray(mask_end, dtype=bool)
    band = mask_end & ~mask_start
    if not band.any():
        raise ValueError("the perimeters show no growth to infer a vector from")
    if interval <= 0:
        raise ValueError(f"interval must be positive; got {interval}")

    advance = grid.sdf(mask_start)
    threshold = np.quantile(advance[band], quantile)
    selected = band & (advance >= threshold)
    if not selected.any():
        selected = band & (advance >= advance[band].max())

    # Displacement of each selected cell from its nearest initial-perimeter cell.
    _, indices = distance_transform_edt(~mask_start, return_indices=True)
    X, Y = grid.meshgrid()
    dx = X[selected] - X[tuple(indices)][selected]
    dy = Y[selected] - Y[tuple(indices)][selected]

    weights = advance[selected]
    ux, uy = float(np.sum(weights * dx)), float(np.sum(weights * dy))
    norm = np.hypot(ux, uy)
    if norm <= 0:
        raise ValueError("the growth has no net direction to infer")

    speed = float(np.mean(weights)) / float(interval)
    x0 = X[tuple(indices)][selected]
    y0 = Y[tuple(indices)][selected]
    wsum = float(np.sum(weights))
    return {
        "ros": np.array([ux / norm, uy / norm]) * speed,
        "advance_m": float(np.mean(weights)),
        "tail": (float(np.sum(weights * x0)) / wsum, float(np.sum(weights * y0)) / wsum),
        "front": (float(np.sum(weights * X[selected])) / wsum,
                  float(np.sum(weights * Y[selected])) / wsum),
        "n_cells": int(selected.sum()),
    }


def resolve_metric(initial, final, grid=None, lb=None, wind_mps=None,
                   infer_from_geometry=False):
    """
    Choose a metric for a perimeter pair, and say why.

    The metric is a parameter rather than a fixed part of the algorithm, so
    the method degrades gracefully when spread data is unavailable - which
    matters, since most archived perimeter data carries no rate-of-spread
    information at all. The order of preference:

        1. The final perimeter carries a ROS vector    -> that one
        2. Only the initial does                       -> that one, flagged
        3. infer_from_geometry=True                    -> inferred from the pair
        4. otherwise                                   -> Euclidean

    The final's vector, not the mean of the two. In the data this was built
    for, the vector attached to a perimeter describes the interval that ends
    at it - which is why the first observation in a series never carries one
    - so the final perimeter's vector is the one for this pair, and the
    initial's belongs to the interval before. Averaging them smeared the
    previous interval's wind into every pair and did the most damage exactly
    where the wind had shifted.

    and independently, for the shape parameter:

        1. lb=                                         -> as given
        2. either perimeter's lb attribute             -> as recorded
        3. wind_mps=                                   -> Finney Eq. 13
        4. otherwise                                   -> no ellipse is defined

    Without LB the ellipse has only one of its two free parameters, so the
    result falls back to Euclidean rather than guessing an elongation. That
    fallback is the floor, not a failure: it is endpoint-exact, monotone,
    continuous, and still a valid arrival-time grid.

    Parameters:
    -----------
    initial, final : Perimeter
        The two observations, already in the grid's CRS
    grid : GridSpec, optional
        Required only for geometric inference
    lb : float, optional
        Length-to-breadth ratio, overriding anything the perimeters carry
    wind_mps : float, optional
        Midflame wind speed for Finney Eq. 13, used when no LB is available
    infer_from_geometry : bool
        Allow inferring the ROS vector from the perimeter pair itself

    Returns:
    --------
    metric : Metric
    provenance : str
        What was used and what was missing, for the run's report
    """
    ros, ros_note = _resolve_ros(initial, final, grid, infer_from_geometry)
    lb_value, lb_note = _resolve_lb(initial, final, lb, wind_mps)

    if ros is None:
        return EuclideanMetric(), f"Euclidean: {ros_note}"
    if lb_value is None:
        return EuclideanMetric(), (f"Euclidean: {lb_note}, so the ellipse is "
                                   f"undetermined even though a ROS vector is "
                                   f"available ({ros_note})")
    metric = EllipticalMetric.from_ros(ros, lb_value)
    return metric, f"elliptical: {ros_note}; {lb_note}"


def _resolve_ros(initial, final, grid, infer_from_geometry):
    """Pick the ROS vector for a pair, and describe where it came from."""
    if final.has_ros:
        return final.ros, "the final perimeter's ROS vector (the interval ending at it)"
    if initial.has_ros:
        return initial.ros, ("the initial perimeter's ROS vector - the final has none, "
                             "so this is the previous interval's vector standing in")
    if infer_from_geometry:
        if grid is None:
            raise ValueError("geometric inference needs a grid")
        ros = infer_ros(initial.mask(grid), final.mask(grid), grid,
                        initial.interval_to(final))
        return ros, "inferred from the perimeter pair's maximum normal advance"
    return None, ("neither perimeter carries a ROS vector; pass "
                  "infer_from_geometry=True to estimate one from the pair")


def _resolve_lb(initial, final, lb, wind_mps):
    """Pick the length-to-breadth ratio for a pair, and describe its source."""
    if lb is not None:
        return float(lb), f"LB={float(lb):.2f} as given"
    recorded = [p.lb for p in (initial, final) if p.lb is not None]
    if recorded:
        value = float(np.mean(recorded))
        return value, f"LB={value:.2f} from the perimeter attributes"
    if wind_mps is not None:
        value = lb_from_wind(wind_mps)
        return value, f"LB={value:.2f} from {wind_mps:g} m/s midflame wind (Finney Eq. 13)"
    return None, ("no LB is available - supply lb= or wind_mps=")
