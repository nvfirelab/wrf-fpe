"""
Tests for the fire ellipse.

The parameterisation has one property that everything else rests on: the
ignition point sits at the ellipse's rear focus, so a, b and c are not
independent and the wavelet has two free parameters rather than three.
"""

import numpy as np
import pytest

from gauge import gauge_arrival
from wavelet import FireEllipse, hb_from_lb, lb_from_wind

ALL_LB = [1.0, 1.2, 1.5, 2.0, 3.0, 6.0, 10.0]


# ----------------------------------------------------------------------
# The published relations
# ----------------------------------------------------------------------

def test_finney_eq13_gives_a_circle_at_zero_wind():
    """0.936 + 0.461 - 0.397 = 1, so still air gives a round fire."""
    assert lb_from_wind(0.0) == pytest.approx(1.0, abs=1e-12)


def test_lb_grows_with_wind():
    winds = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    lb = lb_from_wind(winds)
    assert np.all(np.diff(lb) > 0)
    assert lb_from_wind(5.0) == pytest.approx(
        0.936 * np.exp(0.2566 * 5.0) + 0.461 * np.exp(-0.1548 * 5.0) - 0.397)


def test_wind_speed_cannot_be_negative():
    with pytest.raises(ValueError, match="cannot be negative"):
        lb_from_wind(-1.0)


def test_hb_is_one_at_lb_one_and_grows_from_there():
    assert hb_from_lb(1.0) == pytest.approx(1.0)
    values = [hb_from_lb(lb) for lb in (1.2, 1.5, 3.0, 6.0)]
    assert np.all(np.diff(values) > 0)
    with pytest.raises(ValueError, match="LB must be >= 1"):
        hb_from_lb(0.9)


# ----------------------------------------------------------------------
# The focus condition
# ----------------------------------------------------------------------

@pytest.mark.parametrize("lb", ALL_LB)
def test_ignition_point_sits_at_the_rear_focus(lb):
    """
    b = a/LB and b = sqrt(a^2 - c^2) have to agree, or the parameterisation is
    not describing an ellipse with the ignition point at its focus. Verified
    across LB from 1.2 to 10 in the write-up; this pins it.
    """
    ellipse = FireEllipse.from_ros((0.25, 0.0), lb)
    assert ellipse.c == pytest.approx(np.sqrt(ellipse.a ** 2 - ellipse.b ** 2),
                                      rel=1e-12, abs=1e-15)
    assert ellipse.lb == pytest.approx(lb, rel=1e-12)


@pytest.mark.parametrize("lb", [1.0, 1.5, 10.0, 100.0, 10_000.0])
def test_c_is_below_a_for_every_finite_lb(lb):
    """
    c/a is the eccentricity, so this holds for any wind whatsoever. It is not
    a limit of applicability, and no wind violates it. What the limit marks is
    the degenerate case: as LB grows the backing rate goes to zero.
    """
    ellipse = FireEllipse.from_ros((0.25, 0.0), lb)
    assert ellipse.c < ellipse.a
    assert ellipse.eccentricity == pytest.approx(ellipse.c / ellipse.a)
    assert ellipse.backing > 0.0


def test_backing_vanishes_in_the_degenerate_limit():
    huge = FireEllipse.from_ros((0.25, 0.0), 1e6)
    assert huge.backing == pytest.approx(0.0, abs=1e-6)
    assert huge.head == pytest.approx(0.25, rel=1e-6)


# ----------------------------------------------------------------------
# The three rates
# ----------------------------------------------------------------------

@pytest.mark.parametrize("lb", ALL_LB)
def test_the_three_rates_follow_from_the_axes(lb):
    ellipse = FireEllipse.from_ros((0.25, 0.0), lb)
    assert ellipse.head == pytest.approx(ellipse.a + ellipse.c)
    assert ellipse.backing == pytest.approx(ellipse.a - ellipse.c)
    assert ellipse.head == pytest.approx(0.25, rel=1e-12)
    assert ellipse.hb == pytest.approx(hb_from_lb(lb), rel=1e-9)


def test_flanking_is_not_the_semi_minor_axis():
    """
    b is the wavelet's half-width, but a displacement perpendicular to the
    heading from the ignition point leaves the ellipse before its widest
    point, so the transit rate is smaller.
    """
    ellipse = FireEllipse.from_ros((0.25, 0.0), 3.0)
    assert ellipse.flanking < ellipse.b
    assert ellipse.flanking == pytest.approx(
        ellipse.b * np.sqrt(1 - ellipse.eccentricity ** 2))
    assert ellipse.backing < ellipse.flanking < ellipse.head


# ----------------------------------------------------------------------
# The support function
# ----------------------------------------------------------------------

@pytest.mark.parametrize("lb", [1.2, 3.0, 6.0])
def test_support_function_endpoints(lb):
    """F(0) is the head rate, F(pi) the backing rate, F(pi/2) the semi-minor."""
    ellipse = FireEllipse.from_ros((0.25, 0.0), lb)
    assert ellipse.support(0.0) == pytest.approx(ellipse.head, rel=1e-12)
    assert ellipse.support(np.pi) == pytest.approx(ellipse.backing, rel=1e-12)
    assert ellipse.support(np.pi / 2) == pytest.approx(ellipse.b, rel=1e-12)


def test_support_function_is_positive_all_the_way_round():
    ellipse = FireEllipse.from_ros((0.25, 0.0), 6.0)
    psi = np.linspace(0.0, 2 * np.pi, 721)
    assert np.all(ellipse.support(psi) > 0.0)


# ----------------------------------------------------------------------
# The outline, and its agreement with the gauge
# ----------------------------------------------------------------------

@pytest.mark.parametrize("lb", [1.2, 3.0, 6.0])
@pytest.mark.parametrize("t", [600.0, 4000.0])
def test_the_outline_is_the_solved_front(lb, t):
    """
    A point ignition in a homogeneous medium grows into the wavelet, scaled.
    Every point on that analytic front must have arrival time exactly t - and
    because the gauge is closed form, "exactly" means to machine precision
    rather than to a front-solver's tolerance.
    """
    ellipse = FireEllipse.from_ros((0.25 * np.cos(0.35), 0.25 * np.sin(0.35)), lb)
    ox, oy = ellipse.outline(t, n=720)
    T = gauge_arrival([0.0], [0.0], ellipse.uhat, ellipse.a, ellipse.b, ellipse.c,
                      ox.reshape(1, -1), oy.reshape(1, -1))
    assert np.abs(T - t).max() / t < 1e-12


def test_the_outline_extends_by_the_axis_rates():
    ellipse = FireEllipse.from_ros((0.25, 0.0), 3.0)
    ox, oy = ellipse.outline(1000.0, n=2000)
    assert ox.max() == pytest.approx(1000.0 * ellipse.head, rel=1e-6)
    assert ox.min() == pytest.approx(-1000.0 * ellipse.backing, rel=1e-6)
    assert oy.max() == pytest.approx(1000.0 * ellipse.b, rel=1e-6)


# ----------------------------------------------------------------------
# Orientation and reversal
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bearing", [0.0, 45.0, 137.0, 294.0])
def test_bearing_round_trips(bearing):
    ellipse = FireEllipse.from_speed_bearing(0.25, bearing, 3.0)
    assert ellipse.bearing == pytest.approx(bearing, abs=1e-9)
    assert ellipse.head == pytest.approx(0.25, rel=1e-12)


def test_reversing_flips_the_heading_and_swaps_the_rates():
    ellipse = FireEllipse.from_ros((0.2, 0.15), 3.0)
    reverse = ellipse.reversed()
    assert np.allclose(reverse.uhat, -ellipse.uhat)
    assert (reverse.a, reverse.b, reverse.c) == (ellipse.a, ellipse.b, ellipse.c)
    assert reverse.head == pytest.approx(ellipse.head)
    assert reverse.reversed().bearing == pytest.approx(ellipse.bearing)
    # The shape is unchanged; what reverses is which way it points.
    assert abs(reverse.bearing - ellipse.bearing) == pytest.approx(180.0, abs=1e-9)


def test_isotropic_wavelet_is_a_circle():
    ellipse = FireEllipse.isotropic(0.3)
    assert ellipse.is_isotropic
    assert ellipse.lb == pytest.approx(1.0)
    assert ellipse.hb == pytest.approx(1.0)
    assert ellipse.head == ellipse.backing == pytest.approx(0.3)


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------

def test_a_zero_vector_carries_no_heading():
    with pytest.raises(ValueError, match="zero ROS vector"):
        FireEllipse.from_ros((0.0, 0.0), 3.0)


def test_lb_below_one_is_rejected():
    with pytest.raises(ValueError, match="LB must be >= 1"):
        FireEllipse.from_ros((0.25, 0.0), 0.5)


def test_axes_set_independently_are_rejected():
    """The focus condition is not optional: it is what makes this an ellipse."""
    with pytest.raises(ValueError, match="not at the rear focus"):
        FireEllipse(uhat=(1.0, 0.0), a=1.0, b=0.5, c=0.2)
    with pytest.raises(ValueError, match="semi-minor axis"):
        FireEllipse(uhat=(1.0, 0.0), a=0.5, b=1.0, c=0.0)
    with pytest.raises(ValueError, match="backing rate is non-positive"):
        FireEllipse(uhat=(1.0, 0.0), a=1.0, b=0.001, c=1.5)
