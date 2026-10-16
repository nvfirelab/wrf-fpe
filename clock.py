"""
Module defining the clock: how raw travel time becomes a 0-to-1 progress value.

This is the second of the method's two knobs, and it is the one that controls
endpoint exactness. The two prototypes differed in both knobs at once, which
is why endpoint exactness and realistic spread looked welded together when
they are in fact independent.
"""

from abc import ABC, abstractmethod

import numpy as np


class Clock(ABC):
    """Turns a pair of arrival-time fields into normalised progress on [0, 1]."""

    @abstractmethod
    def progress(self, T_f, T_b, grid, mask_start, mask_end):
        """
        Normalised fire arrival time.

        Parameters:
        -----------
        T_f : numpy.ndarray
            Arrival time spreading outward from the initial perimeter
        T_b : numpy.ndarray
            Arrival time spreading inward from the final perimeter, under the
            reversed metric
        grid : GridSpec
            The shared computational grid
        mask_start, mask_end : numpy.ndarray
            The two observed perimeters rasterised onto the grid

        Returns:
        --------
        s : numpy.ndarray
            Progress field, before the endpoint overrides are applied
        """

    def describe(self):
        """One-line human-readable summary, for provenance in reports."""
        return repr(self)


class TwoSidedClock(Clock):
    """
    Progress as the fraction of the journey through a cell that is complete.

        s = T_f / ( T_f + T_b )

    T_f + T_b is the total travel time of the path through x, so s is how much
    of that journey had been made when the fire arrived. The endpoints follow
    structurally: on the initial perimeter T_f = 0 so s = 0, on the final
    perimeter T_b = 0 so s = 1. Neither is fitted, and both hold under any
    metric - which is why changing the metric costs nothing in endpoint
    accuracy.

    The denominator is per point, not global. Every cell has its own total
    journey time, so a cell on a fast downwind path and one on a slow backing
    path can both read s = 0.5 despite very different absolute times. That is
    the correct behaviour for interpolating between two fixed observations.

    Where the two perimeters run close together the journey through a cell is
    short, and s traverses 0 to 1 over very few cells. The field is still
    continuous - moving one cell changes T_f by at most one cell, so a step in
    s is bounded by cell_size / (T_f + T_b) - but it is steep, and the
    interval is genuinely under-determined there: the fire barely moved, so
    there is little journey to take a fraction of. GaugeEvolution reports the
    shortest journey on the grid so this stays visible rather than implicit.
    """

    def progress(self, T_f, T_b, grid, mask_start, mask_end):
        total = T_f + T_b
        return T_f / np.maximum(total, 1e-12)

    def describe(self):
        return "two-sided (journey travelled over journey travelled plus remaining)"

    def __repr__(self):
        return "TwoSidedClock()"


class OneSidedClock(Clock):
    """
    Progress as arrival time over an estimate of the total, from a lookup.

        s = T_f / T_f(nearest point of the final boundary)

    This is the physics-guided prototype's normalisation, and it is here to be
    measured, not to be used. The lookup is constant across whole regions and
    jumps wherever the nearest boundary point changes - that is, across the
    medial axis of the final perimeter. The ratio inherits those jumps, which
    breaks the continuity the arrival grid needs, and the level set where the
    ratio equals one is not the observed perimeter, so the final contour lands
    somewhere the observation contradicts.

    Keeping it costs a few lines and makes the failure reproducible on the
    same data as the fix, which is worth more than a description of it.
    """

    def progress(self, T_f, T_b, grid, mask_start, mask_end):
        from scipy.ndimage import distance_transform_edt

        boundary = grid.boundary_cells(mask_end)
        if not boundary.any():
            raise ValueError("the final perimeter has no boundary to look up")
        # Nearest final-boundary cell to every cell, then that cell's arrival
        # time. This is the medial-axis discontinuity, reproduced exactly.
        _, indices = distance_transform_edt(~boundary, return_indices=True)
        total = T_f[tuple(indices)]
        return T_f / np.maximum(total, 1e-12)

    def describe(self):
        return "one-sided (arrival time over a nearest-boundary total; discontinuous)"

    def __repr__(self):
        return "OneSidedClock()"
