"""
Module building the fire ellipse - the wavelet - from an observed ROS vector.

Treat every point on the burning front as a new ignition. After a short time
each has grown into a small shape, and the new front is the outer envelope of
all of them. The entire spread model is therefore encoded in one shape: what a
single ignition becomes in unit time. Choose the wavelet and you have chosen
the metric.

The word is borrowed from Huygens' optics and has nothing to do with wavelet
transforms. The fire literature's "elliptical template" or "fire shape" mean
the same thing, and the wavelet is a local law rather than a prediction of a
fire's shape: a real perimeter is the envelope of thousands of these, and is
not itself an ellipse.

Empirically a point ignition under steady wind grows into an ellipse whose
ignition point sits at the rear focus, not at the centre. That offset is
exactly what makes heading spread faster than backing.
"""

from dataclasses import dataclass

import numpy as np


def lb_from_wind(u_midflame):
    """
    Length-to-breadth ratio of the fire ellipse from midflame wind speed.

    Finney (1998, rev. 2004) Eq. 13, which is Anderson's (1983) relation with
    Finney's trailing -0.397 modification. Take it in this form rather than
    reconstructing it from the 1983 paper; U is midflame windspeed in metres
    per second, and the expression gives LB = 1 at zero wind.

    Parameters:
    -----------
    u_midflame : float or array-like
        Midflame wind speed in metres per second

    Returns:
    --------
    LB : float or numpy.ndarray
        Length-to-breadth ratio, at least 1
    """
    u = np.asarray(u_midflame, dtype=float)
    if np.any(u < 0):
        raise ValueError("midflame wind speed cannot be negative")
    lb = 0.936 * np.exp(0.2566 * u) + 0.461 * np.exp(-0.1548 * u) - 0.397
    lb = np.maximum(lb, 1.0)
    return float(lb) if lb.ndim == 0 else lb


def hb_from_lb(lb):
    """
    Head-to-back ratio from the length-to-breadth ratio.

    Finney (1998) Eq. 14. This is not an independent empirical fit: putting
    the ignition point at the ellipse's rear focus fixes the offset at the
    focal distance, and substituting that into (a + c)/(a - c) gives this
    expression directly.

    Parameters:
    -----------
    lb : float
        Length-to-breadth ratio, at least 1

    Returns:
    --------
    HB : float
        Head-to-back ratio. 1 at LB = 1, growing without bound as LB does.
    """
    lb = float(lb)
    if lb < 1.0:
        raise ValueError(f"LB must be >= 1; got {lb}")
    root = np.sqrt(lb * lb - 1.0)
    if lb - root <= 0.0:
        return np.inf
    return float((lb + root) / (lb - root))


@dataclass(frozen=True, eq=False)
class FireEllipse:
    """
    The shape a single point ignition grows into in unit time.

    Three quantities describe it, all in metres per second because the wavelet
    is a unit-time shape:

        head = c + a ,  backing = a - c ,  flanking = b

    a is the semi-major axis, b the semi-minor, and c the offset of the
    ignition point behind the ellipse centre. They are not independent: the
    ignition point sits at the rear focus, which fixes c = sqrt(a^2 - b^2), so
    c/a is the eccentricity and the wavelet has two free parameters rather
    than three.

    Because c/a is the eccentricity, c < a holds for every finite LB. That is
    not a physical constraint on wind speed and no wind violates it; the check
    in code is a guard against a corrupted parameterisation. What the limit
    does mark is the degenerate case: as LB grows without bound the backing
    rate goes to zero.

    Attributes:
    -----------
    uhat : numpy.ndarray
        Unit heading (ux, uy) in the grid's axes
    a, b, c : float
        Semi-major axis, semi-minor axis and rear-focus offset, in m/s
    """

    uhat: np.ndarray
    a: float
    b: float
    c: float

    def __post_init__(self):
        uhat = np.asarray(self.uhat, dtype=float).reshape(2)
        norm = float(np.hypot(*uhat))
        if norm <= 0:
            raise ValueError("heading must be a nonzero vector")
        object.__setattr__(self, "uhat", uhat / norm)
        object.__setattr__(self, "a", float(self.a))
        object.__setattr__(self, "b", float(self.b))
        object.__setattr__(self, "c", float(self.c))

        if self.a <= 0 or self.b <= 0:
            raise ValueError(f"axes must be positive; got a={self.a}, b={self.b}")
        if self.b > self.a * (1.0 + 1e-9):
            raise ValueError(f"semi-minor axis {self.b} exceeds semi-major {self.a}")
        if self.c < 0:
            raise ValueError(f"offset must not be negative; got c={self.c}")
        if self.c >= self.a:
            raise ValueError(
                f"c >= a (c={self.c}, a={self.a}): backing rate is non-positive. "
                "c/a is the eccentricity, so this cannot arise from any finite "
                "LB - the parameterisation is corrupted."
            )
        focal = np.sqrt(max(self.a ** 2 - self.b ** 2, 0.0))
        if abs(self.c - focal) > 1e-6 * max(self.a, 1.0):
            raise ValueError(
                f"the ignition point is not at the rear focus: c={self.c} but "
                f"sqrt(a^2 - b^2)={focal}. Build the ellipse with from_ros() "
                "rather than setting the axes independently."
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_ros(cls, ros, lb):
        """
        Build the wavelet from one observed ROS vector and a shape parameter.

        You observe one vector, which gives the heading and the head rate. One
        further number - the length-to-breadth ratio - closes the system.

        Parameters:
        -----------
        ros : array-like
            (u, v) rate of spread in metres per second, in the grid's axes
        lb : float
            Length-to-breadth ratio, at least 1

        Returns:
        --------
        ellipse : FireEllipse
        """
        w = np.asarray(ros, dtype=float).reshape(2)
        head = float(np.hypot(*w))
        if head < 1e-9:
            raise ValueError("zero ROS vector: it carries no heading")
        lb = float(lb)
        if lb < 1.0:
            raise ValueError(f"LB must be >= 1; got {lb}")

        hb = hb_from_lb(lb)
        back = 0.0 if np.isinf(hb) else head / hb
        a = 0.5 * (head + back)
        c = 0.5 * (head - back)
        b = a / lb
        return cls(uhat=w / head, a=a, b=b, c=c)

    @classmethod
    def from_speed_bearing(cls, speed, bearing_deg, lb):
        """
        Build the wavelet from a speed and a compass bearing.

        Parameters:
        -----------
        speed : float
            Head rate of spread in metres per second
        bearing_deg : float
            Direction of travel, degrees clockwise from the grid's north
        lb : float
            Length-to-breadth ratio

        Returns:
        --------
        ellipse : FireEllipse
        """
        theta = np.deg2rad(float(bearing_deg))
        return cls.from_ros((speed * np.sin(theta), speed * np.cos(theta)), lb)

    @classmethod
    def isotropic(cls, speed):
        """A circular wavelet: no heading, LB = 1, every direction equal."""
        return cls(uhat=(1.0, 0.0), a=speed, b=speed, c=0.0)

    # ------------------------------------------------------------------
    # The three rates
    # ------------------------------------------------------------------

    @property
    def head(self):
        """Spread rate along the heading, in m/s."""
        return self.a + self.c

    @property
    def backing(self):
        """Spread rate against the heading, in m/s."""
        return self.a - self.c

    @property
    def flanking(self):
        """
        Spread rate perpendicular to the heading as the gauge measures it, in m/s.

        Not the semi-minor axis. b is the wavelet's half-width, but a
        displacement perpendicular to the heading *from the ignition point*
        leaves the ellipse before its widest point, so the transit rate is
        b*sqrt(1 - c^2/a^2).
        """
        return self.b * np.sqrt(1.0 - self.eccentricity ** 2)

    @property
    def eccentricity(self):
        """c/a, which is both the offset ratio and the ellipse's eccentricity."""
        return self.c / self.a

    @property
    def lb(self):
        """Length-to-breadth ratio."""
        return self.a / self.b

    @property
    def hb(self):
        """Head-to-back ratio."""
        return self.head / self.backing

    @property
    def bearing(self):
        """Heading in degrees clockwise from the grid's north."""
        return float(np.degrees(np.arctan2(self.uhat[0], self.uhat[1])) % 360.0)

    @property
    def is_isotropic(self):
        """Whether the wavelet is a circle, making the metric Euclidean."""
        return self.c == 0.0 and self.a == self.b

    # ------------------------------------------------------------------
    # The two views of the wavelet
    # ------------------------------------------------------------------

    def support(self, psi):
        """
        Normal speed of a front facing at angle psi to the heading.

            F(psi) = c*cos(psi) + sqrt( a^2 cos^2(psi) + b^2 sin^2(psi) )

        This is the support function: the perpendicular distance from the
        ignition point to the wavelet's tangent line in that direction. It is
        the form the level-set and Eikonal literature uses, because a PDE
        needs a normal speed. The gauge - the dual question, how long until
        the fire covers a given displacement - is what this method solves
        instead, and it lives in gauge.py.

        Parameters:
        -----------
        psi : float or array-like
            Angle in radians between the front normal and the heading

        Returns:
        --------
        F : float or numpy.ndarray
            Normal spread rate in metres per second
        """
        psi = np.asarray(psi, dtype=float)
        F = (self.c * np.cos(psi)
             + np.sqrt((self.a * np.cos(psi)) ** 2 + (self.b * np.sin(psi)) ** 2))
        return float(F) if F.ndim == 0 else F

    def outline(self, t=1.0, n=360):
        """
        The wavelet's boundary after time t, as coordinates about the ignition point.

        The analytic front from a point ignition in a homogeneous medium, and
        therefore what a solved arrival-time field has to reproduce.

        Parameters:
        -----------
        t : float
            Elapsed time in seconds
        n : int
            Number of points around the ellipse

        Returns:
        --------
        x, y : numpy.ndarray
            Offsets from the ignition point in metres
        """
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        # Ellipse centred c*t ahead of the ignition point, in the heading frame.
        ex = self.c * t + self.a * t * np.cos(theta)
        ey = self.b * t * np.sin(theta)
        ux, uy = self.uhat
        return ex * ux - ey * uy, ex * uy + ey * ux

    def reversed(self):
        """
        The same ellipse with the heading negated, for the inward solve.

        Receding traverses the same paths backwards, so the shape is unchanged
        and only the offset flips. Getting this wrong does not throw: it
        produces a plausible-looking field with the wrong intermediate
        ordering.
        """
        return FireEllipse(uhat=-self.uhat, a=self.a, b=self.b, c=self.c)

    def __eq__(self, other):
        """
        Value equality. Defined by hand because the default dataclass version
        compares uhat with ==, which returns an array rather than a bool.
        """
        if not isinstance(other, FireEllipse):
            return NotImplemented
        return (np.array_equal(self.uhat, other.uhat)
                and (self.a, self.b, self.c) == (other.a, other.b, other.c))

    def __hash__(self):
        return hash((float(self.uhat[0]), float(self.uhat[1]),
                     self.a, self.b, self.c))

    def describe(self):
        """One-line human-readable summary, for provenance in reports."""
        return (f"ellipse LB={self.lb:.2f} HB={self.hb:.2f}, head {self.head:.4f} m/s "
                f"@ {self.bearing:.0f} deg, flank {self.flanking:.4f}, "
                f"back {self.backing:.4f} m/s")

    def __repr__(self):
        return (f"FireEllipse(uhat=({self.uhat[0]:.4f}, {self.uhat[1]:.4f}), "
                f"a={self.a:.6g}, b={self.b:.6g}, c={self.c:.6g})")
