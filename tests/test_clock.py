"""
Tests for the clock knob.

The two-sided clock is what makes the endpoints exact. The one-sided clock is
kept only so its failure stays reproducible on the same data as the fix.
"""

import numpy as np
import pytest

from clock import OneSidedClock, TwoSidedClock
from evolution import GaugeEvolution


def test_two_sided_endpoints_are_structural():
    """T_f = 0 forces s = 0 and T_b = 0 forces s = 1, whatever the metric was."""
    T_f = np.array([[0.0, 1.0, 3.0, 5.0]])
    T_b = np.array([[5.0, 4.0, 2.0, 0.0]])
    s = TwoSidedClock().progress(T_f, T_b, None, None, None)
    assert s[0, 0] == 0.0
    assert s[0, -1] == 1.0
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_two_sided_denominator_is_per_point():
    """
    A cell on a fast path and one on a slow path can both read 0.5 despite
    very different absolute times. That is the point of the two-sided clock.
    """
    T_f = np.array([[10.0, 1000.0]])
    T_b = np.array([[10.0, 1000.0]])
    s = TwoSidedClock().progress(T_f, T_b, None, None, None)
    assert s[0, 0] == pytest.approx(0.5)
    assert s[0, 1] == pytest.approx(0.5)


def test_two_sided_never_overshoots(pair):
    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid, clock=TwoSidedClock())
    assert evolution.overshoot() == 0.0
    inside = evolution.progress[evolution.mask_end]
    assert np.nanmax(inside) <= 1.0
    assert np.nanmin(inside) >= 0.0


def test_one_sided_overshoots_inside_the_final_perimeter(pair):
    """
    The nearest-boundary lookup makes the s = 1 level set land somewhere the
    observation contradicts, which is C2 broken rather than merely loosened.
    """
    initial, final, grid = pair
    one = GaugeEvolution(initial, final, grid, clock=OneSidedClock())
    two = GaugeEvolution(initial, final, grid, clock=TwoSidedClock())

    assert one.overshoot() > 0.05
    assert two.overshoot() == 0.0
    assert one.endpoint_jaccard()[1] < two.endpoint_jaccard()[1] == 1.0


def test_one_sided_jumps_across_the_medial_axis(pair):
    """
    The lookup is constant across whole regions and jumps where the nearest
    boundary point changes.

    Measured as a step between neighbouring cells divided by the largest step
    a continuous field could take there, cell_size / (T_f + T_b). The
    two-sided field is Lipschitz by construction - moving one cell changes
    T_f by at most one cell - so its worst normalised step is 1. Anything
    much above that is a discontinuity rather than a steep gradient, and the
    distinction matters because s is legitimately steep wherever the two
    perimeters run close together.
    """
    initial, final, grid = pair
    two = GaugeEvolution(initial, final, grid, clock=TwoSidedClock())
    one = GaugeEvolution(initial, final, grid, clock=OneSidedClock())
    inside = two.mask_end
    scale = np.where(inside, grid.cell_size / np.maximum(two.T_f + two.T_b, 1e-9),
                     np.nan)

    def worst_normalised_step(evolution):
        field = np.where(inside, evolution.progress_raw, np.nan)
        worst = 0.0
        for axis in (0, 1):
            step = np.abs(np.diff(field, axis=axis))
            allowed = np.minimum(np.take(scale, range(scale.shape[axis] - 1), axis),
                                 np.take(scale, range(1, scale.shape[axis]), axis))
            worst = max(worst, float(np.nanmax(step / allowed)))
        return worst

    assert worst_normalised_step(two) <= 1.0 + 1e-9
    assert worst_normalised_step(one) > 50.0


def test_one_sided_needs_a_final_boundary(pair):
    initial, final, grid = pair
    evolution = GaugeEvolution(initial, final, grid, clock=OneSidedClock())
    empty = np.zeros(grid.shape, dtype=bool)
    with pytest.raises(ValueError, match="no boundary"):
        OneSidedClock().progress(evolution.T_f, evolution.T_b, grid,
                                 evolution.mask_start, empty)


def test_clocks_describe_themselves():
    assert "two-sided" in TwoSidedClock().describe()
    assert "discontinuous" in OneSidedClock().describe()
