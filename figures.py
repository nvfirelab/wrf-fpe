"""
Module drawing the figures: the filmstrip row, the arrival-time field, and
the elliptical-versus-Euclidean comparison.

The poster carried two filmstrip rows because it had two prototypes, each
getting one thing right: the linear row met the final perimeter exactly and
spread unrealistically, the physics-guided row spread realistically and
missed the perimeter. Section 3 shows those two properties come from two
independent knobs, so one row from this method - plausible intermediate
fronts and an exact final match - replaces both. That row is `filmstrip`.

Conventions are kept from the prototype's figure so the replacement reads
in the same language: the initial perimeter dashed in blue, the final dashed
in green, the evolved front solid in red, one panel per progress level. What
is added is the evidence the row is making its claim on: the burned area at
each level filled, the wavelet drawn in the corner so the anisotropy is
visible rather than asserted, a scale bar because the grid is in metres, and
the endpoint Jaccard printed on the last panel because "exact" should be a
number.

The product is a continuous time-of-arrival field, not a set of frames, and
`arrival_map` draws that field itself - the thing SFIRE actually receives.
`comparison_filmstrip` puts the elliptical and Euclidean rows one above the
other for the same pair, which is the §8 figure that justifies the method or
reports that a bulk vector carries little information; M4 says which.

`chain_filmstrip` and `chain_arrival_map` are the same two pictures for a
whole chained series (an `ArrivalTime`): panels at absolute times rather than
progress levels, every observed perimeter drawn, and the front landing on
each observation in turn.

`animate` and `chain_animate` are the filmstrips as GIFs, one frame per
level or per instant. For a long series the animation is the readable form
- forty segments make a filmstrip of a hundred postage stamps - so it is
what chain mode writes by default.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# The prototype's palette, kept so the new row reads like the ones it replaces.
INITIAL_COLOUR = "#1f5fbf"
FINAL_COLOUR = "#2a8f3c"
FRONT_COLOUR = "#d62728"
BURNED_COLOUR = "#f2b8a2"
BURNED_AT_START_COLOUR = "#c9c9c9"

DEFAULT_LEVELS = (0.25, 0.50, 0.75, 1.00)


# ----------------------------------------------------------------------
# The filmstrip row
# ----------------------------------------------------------------------

def filmstrip(evolution, levels=DEFAULT_LEVELS, path=None, title=None,
              show_wavelet=True, panel_size=4.6, dpi=300):
    """
    One row: the burned area and front at each progress level.

    Parameters:
    -----------
    evolution : GaugeEvolution
        The evolved pair
    levels : sequence of float
        Progress levels on (0, 1], one panel each
    path : str, optional
        Where to save the figure; not saved if omitted
    title : str, optional
        Figure title. Defaults to the observation times and the metric.
    show_wavelet : bool
        Draw the fire ellipse and heading in the first panel's corner
    panel_size : float
        Panel width in inches
    dpi : int
        Resolution when saving

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    levels = [float(v) for v in levels]
    if not levels or min(levels) <= 0.0 or max(levels) > 1.0:
        raise ValueError("levels must lie in (0, 1]")

    fig, axes = plt.subplots(1, len(levels), figsize=(panel_size * len(levels), panel_size),
                             squeeze=False)
    start_j, end_j = evolution.endpoint_jaccard()
    for k, (ax, level) in enumerate(zip(axes[0], levels)):
        draw_panel(ax, evolution, level, wavelet=(show_wavelet and k == 0),
                   scale_bar=(k == 0))
        ax.set_title(f"s = {level:.2f}   {_minutes(evolution, level)}", fontsize=11)
        if level >= 1.0:
            ax.text(0.03, 0.97, f"final match\nJaccard {end_j:.3f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=FINAL_COLOUR, linewidth=1.0), zorder=7)

    _legend(fig, evolution)
    fig.suptitle(title if title is not None else _default_title(evolution),
                 fontsize=12, y=1.04)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def draw_panel(ax, evolution, level, wavelet=False, scale_bar=False,
               observed=True):
    """
    One panel of the filmstrip onto an existing axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
    evolution : GaugeEvolution
    level : float
        Progress level to draw the front at
    wavelet : bool
        Draw the fire ellipse glyph in the corner
    scale_bar : bool
        Draw a scale bar
    observed : bool
        Draw the two observed perimeters dashed
    """
    grid = evolution.grid
    X, Y = grid.meshgrid()
    field = _contourable(evolution)
    burned = (evolution.progress <= level) & evolution.mask_end

    # Burned area: grey for what was already burned at the first observation,
    # the fire colour for what this interval added up to this level.
    fill = np.full(grid.shape, np.nan)
    fill[burned] = 1.0
    fill[evolution.mask_start & evolution.mask_end] = 0.0
    cmap = matplotlib.colors.ListedColormap([BURNED_AT_START_COLOUR, BURNED_COLOUR])
    ax.imshow(fill, origin="lower", extent=grid.extent, cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest", zorder=1)

    if observed:
        _draw_geometry(ax, evolution.initial.geometry, color=INITIAL_COLOUR,
                       linestyle="--", linewidth=1.4, zorder=3)
        _draw_geometry(ax, evolution.final.geometry, color=FINAL_COLOUR,
                       linestyle="--", linewidth=1.4, zorder=3)
    ax.contour(X, Y, field, levels=[level], colors=[FRONT_COLOUR], linewidths=2.2,
               zorder=4)

    if wavelet:
        _draw_wavelet(ax, evolution, grid)
    if scale_bar:
        _scale_bar(ax, grid)

    ax.set_xlim(grid.extent[0], grid.extent[1])
    ax.set_ylim(grid.extent[2], grid.extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# ----------------------------------------------------------------------
# The arrival-time field
# ----------------------------------------------------------------------

def arrival_map(evolution, path=None, contour_minutes=None, title=None,
                size=6.5, dpi=300, t0=0.0):
    """
    The continuous field itself: ignition time with iso-arrival contours.

    Parameters:
    -----------
    evolution : GaugeEvolution
        The evolved pair
    path : str, optional
        Where to save
    contour_minutes : float, optional
        Spacing of the iso-arrival contours in minutes. Defaults to a round
        number giving eight to twelve contours over the interval.
    title : str, optional
    size : float
        Figure width in inches
    dpi : int
    t0 : float
        Seconds from the simulation start to the initial observation, added
        to every value so the colourbar matches what is exported

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    grid = evolution.grid
    X, Y = grid.meshgrid()
    minutes = evolution.arrival_seconds(t0) / 60.0
    span = evolution.interval / 60.0
    if contour_minutes is None:
        contour_minutes = _round_step(span / 10.0)

    fig, ax = plt.subplots(figsize=(size, size * grid.ny / grid.nx))
    image = ax.imshow(minutes, origin="lower", extent=grid.extent, cmap="magma_r",
                      interpolation="nearest", zorder=1)
    lo, hi = t0 / 60.0, t0 / 60.0 + span
    first = np.ceil(lo / contour_minutes) * contour_minutes
    levels = np.arange(first, hi + 1e-9, contour_minutes)
    if levels.size:
        cs = ax.contour(X, Y, _contourable(evolution) * span + t0 / 60.0,
                        levels=levels, colors="white", linewidths=0.7, alpha=0.9,
                        zorder=2)
        ax.clabel(cs, fmt=lambda v: f"{v:.0f}", fontsize=7, inline=True)
    _draw_geometry(ax, evolution.initial.geometry, color=INITIAL_COLOUR,
                   linestyle="--", linewidth=1.4, zorder=3)
    _draw_geometry(ax, evolution.final.geometry, color=FINAL_COLOUR,
                   linestyle="--", linewidth=1.4, zorder=3)
    _draw_wavelet(ax, evolution, grid)
    _scale_bar(ax, grid)

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label("ignition time (minutes)")
    ax.set_xlim(grid.extent[0], grid.extent[1])
    ax.set_ylim(grid.extent[2], grid.extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(title if title is not None else
                 f"Arrival time, contours every {contour_minutes:g} min\n"
                 f"{_default_title(evolution)}", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Elliptical against Euclidean
# ----------------------------------------------------------------------

def comparison_filmstrip(comparison, levels=DEFAULT_LEVELS, path=None,
                         title=None, panel_size=4.2, dpi=300):
    """
    Two rows for one pair: the metric under test above the Euclidean control.

    Both rows meet the final perimeter exactly - that is the clock, not the
    metric - so the figure is read across the interior panels. Each control
    panel is labelled with the Jaccard index against the test panel above
    it, so the visual difference has a number attached.

    Parameters:
    -----------
    comparison : MetricComparison
    levels : sequence of float
    path : str, optional
    title : str, optional
    panel_size : float
    dpi : int

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    levels = [float(v) for v in levels]
    rows = (("elliptical", comparison.test), ("Euclidean control", comparison.control))
    fig, axes = plt.subplots(2, len(levels),
                             figsize=(panel_size * len(levels), 2 * panel_size + 0.6),
                             squeeze=False)
    for r, (name, run) in enumerate(rows):
        for k, (ax, level) in enumerate(zip(axes[r], levels)):
            draw_panel(ax, run, level, wavelet=(r == 0 and k == 0),
                       scale_bar=(r == 0 and k == 0))
            label = f"s = {level:.2f}"
            if r == 1:
                a = (comparison.test.progress <= level) & comparison.test.mask_end
                b = (run.progress <= level) & run.mask_end
                label += f"   agreement {_jaccard(a, b):.3f}"
            ax.set_title(label, fontsize=10)
        axes[r][0].set_ylabel(name, fontsize=11)
        axes[r][0].yaxis.set_visible(True)
        axes[r][0].set_yticks([])

    _legend(fig, comparison.test)
    fig.suptitle(title if title is not None else
                 f"{comparison.test.metric.describe()}\nagainst {comparison.control_metric.describe()}",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# A chained series
# ----------------------------------------------------------------------

def chain_filmstrip(arrival, times=None, panels_per_segment=3, max_columns=4,
                    path=None, title=None, panel_size=4.4, dpi=300):
    """
    The filmstrip for a whole chained series, one panel per absolute time.

    Where `filmstrip` walks one pair from s = 0 to 1, this walks the chained
    field in seconds from the simulation start, so the same picture covers
    every segment and the front is seen landing on each observed perimeter
    in turn. Panels that fall on an observation time carry that match as a
    Jaccard index, which is C2 shown holding at every join and not only at
    the end.

    Parameters:
    -----------
    arrival : ArrivalTime
        The chained field, with its perimeters
    times : sequence, optional
        Panel times, as seconds from the simulation start or as timezone-
        aware datetimes / ISO strings. Defaults to `panels_per_segment`
        evenly spaced times through each segment, ending on its observation.
    panels_per_segment : int
        Used when `times` is omitted
    max_columns : int
        Panels per row before wrapping
    path : str, optional
    title : str, optional
    panel_size : float
    dpi : int

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    seconds_list = _panel_times(arrival, times, panels_per_segment)
    n = len(seconds_list)
    ncols = max(1, min(int(max_columns), n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows),
                             squeeze=False)
    colours = _observation_colours(len(arrival.perimeters))
    obs_times = arrival.observation_times

    for k, (ax, t) in enumerate(zip(axes.flat, seconds_list)):
        draw_chain_panel(ax, arrival, t, colours=colours, wavelet=(k == 0),
                         scale_bar=(k == 0))
        when = arrival.simulation_start + _timedelta(t)
        label = f"{when.strftime('%H:%MZ')}   +{(t - arrival.t_first) / 60:.0f} min"
        ax.set_title(label, fontsize=10)
        hit = [i for i, ot in enumerate(obs_times) if abs(ot - t) <= 1e-6 * max(t, 1.0)]
        if hit:
            i = hit[0]
            burned = np.isfinite(arrival.seconds) & (arrival.seconds <= t)
            observed = arrival.perimeters[i].mask(arrival.grid)
            ax.text(0.03, 0.97, f"observed\nJaccard {_jaccard(burned, observed):.3f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=colours[i], linewidth=1.0), zorder=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    _chain_legend(fig, arrival, colours)
    fig.suptitle(title if title is not None else _chain_title(arrival), fontsize=12,
                 y=1.0 + 0.04 / nrows)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def draw_chain_panel(ax, arrival, t, colours=None, wavelet=False, scale_bar=False):
    """
    One panel of the chain filmstrip: the burned area and front at time t.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
    arrival : ArrivalTime
    t : float
        Seconds from the simulation start
    colours : sequence, optional
        One colour per observed perimeter
    wavelet, scale_bar : bool
    """
    grid = arrival.grid
    X, Y = grid.meshgrid()
    seconds = arrival.seconds
    finite = np.isfinite(seconds)
    burned = finite & (seconds <= t)
    colours = colours if colours is not None else _observation_colours(len(arrival.perimeters))

    fill = np.full(grid.shape, np.nan)
    fill[burned] = 1.0
    fill[arrival.first_mask] = 0.0
    cmap = matplotlib.colors.ListedColormap([BURNED_AT_START_COLOUR, BURNED_COLOUR])
    ax.imshow(fill, origin="lower", extent=grid.extent, cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest", zorder=1)

    for perimeter, colour in zip(arrival.perimeters, colours):
        _draw_geometry(ax, perimeter.geometry, color=colour, linestyle="--",
                       linewidth=1.3, zorder=3)

    # Outside the last perimeter is "never", so the level set at any t closes
    # on the last observation's edge rather than running open into the NaNs.
    field = np.where(finite, seconds, arrival.t_last + 1.0)
    if np.nanmin(field) < t < np.nanmax(field):
        ax.contour(X, Y, field, levels=[t + 1e-6 * max(t, 1.0)], colors=[FRONT_COLOUR],
                   linewidths=2.2, zorder=4)

    if wavelet:
        _draw_ellipse_glyph(ax, _segment_ellipse(arrival, t), grid, arrival.last_mask)
    if scale_bar:
        _scale_bar(ax, grid)
    ax.set_xlim(grid.extent[0], grid.extent[1])
    ax.set_ylim(grid.extent[2], grid.extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


ARRIVAL_SCALES = ("segments", "area", "linear")


def _arrival_scale(arrival, scale, max_knots=33):
    """
    The colour mapping for a chained arrival-time field, in minutes.

    A chain's intervals routinely span three orders of magnitude - minutes
    between the first observations, days across the last - and a colour axis
    linear in time then spends almost its whole range on the longest
    interval, rendering everything else as one flat shade. The remedy is a
    piecewise-linear axis; the colourbar is ticked at real times, so the
    nonlinearity is on display rather than hidden.

    scale : str
        "segments" gives every evolution an equal share of the ramp, so a
        three-minute interval and a three-day one are equally readable, and
        colour reads as "which interval did this cell ignite in".
        "area" spaces the knots at the field's own quantiles, so equal
        colour covers equal burned area and the resolution follows the fire.
        "linear" is linear in time: honest about rate, unreadable on a
        ragged chain.

    Returns:
    --------
    norm : matplotlib.colors.Normalize
    ticks : numpy.ndarray or None
        Colourbar ticks in minutes; None to let matplotlib choose
    """
    if scale not in ARRIVAL_SCALES:
        raise ValueError(f"scale must be one of {ARRIVAL_SCALES}, not {scale!r}")
    lo, hi = arrival.t_first / 60.0, arrival.t_last / 60.0
    linear = matplotlib.colors.Normalize(vmin=lo, vmax=hi)
    if scale == "linear" or not hi > lo:
        return linear, None

    if scale == "segments":
        knots = np.asarray(arrival.observation_times, dtype=float) / 60.0
    else:
        finite = arrival.seconds[np.isfinite(arrival.seconds)] / 60.0
        if finite.size < 2:
            return linear, None
        knots = np.quantile(finite, np.linspace(0.0, 1.0, max_knots))
        knots[0], knots[-1] = lo, hi

    # np.interp needs strictly increasing knots: a zero-length interval, or a
    # quantile repeated because many cells share an ignition time, would
    # otherwise make the mapping non-invertible.
    span = hi - lo
    kept = [float(knots[0])]
    for value in knots[1:]:
        if value > kept[-1] + 1e-9 * span:
            kept.append(float(value))
    knots = np.array(kept)
    if knots.size < 3:
        return linear, None

    targets = np.linspace(0.0, 1.0, knots.size)
    norm = matplotlib.colors.FuncNorm(
        (lambda v: np.interp(v, knots, targets),
         lambda u: np.interp(u, targets, knots)), vmin=lo, vmax=hi)
    step = max(1, int(np.ceil((knots.size - 1) / 8.0)))
    ticks = np.unique(np.concatenate([knots[::step], knots[-1:]]))
    return norm, ticks


def _elapsed_format(values, span_minutes):
    """
    A unit and format for elapsed-time labels that keeps them distinct.

    The unit follows the span - minutes for a short chain, hours for a long
    one - but the precision follows the *values*, because the labelled times
    on a stretched colour axis bunch up at the active end and a fixed one
    decimal would print several of them identically.

    Returns:
    --------
    unit : str
    divisor : float
    fmt : callable
    """
    unit, divisor = ("hours", 60.0) if span_minutes >= 360.0 else ("minutes", 1.0)
    scaled = np.asarray(values, dtype=float) / divisor
    for decimals in (0, 1, 2, 3):
        labels = [f"{v:.{decimals}f}" for v in scaled]
        if len(set(labels)) == len(labels):
            break
    return unit, divisor, (lambda v, d=decimals: f"{v / divisor:.{d}f}")


def _clock_labels(arrival, minutes):
    """Observation-style UTC labels for times given in minutes from the start."""
    stamps = [arrival.simulation_start + _timedelta(60.0 * float(m)) for m in minutes]
    spans_days = (max(stamps) - min(stamps)).total_seconds() > 86400.0
    return [t.strftime("%d %H:%MZ" if spans_days else "%H:%MZ") for t in stamps]


def chain_arrival_map(arrival, path=None, contour_minutes=None, title=None,
                      scale="segments", size=6.5, dpi=300):
    """
    The chained arrival-time field with iso-arrival contours and every observation.

    The field is the whole chain's - one raster, every evolution in it - and
    by default the colour axis is stretched so each evolution gets an equal
    share of the ramp. See `_arrival_scale` for why, and for the
    alternatives.

    Parameters:
    -----------
    arrival : ArrivalTime
    path : str, optional
    contour_minutes : float, optional
        Contour spacing in minutes. The default places roughly ten contours
        evenly along the colour axis instead, which on a ragged chain is not
        even spacing in time
    title : str, optional
    scale : str
        "segments" (default), "area" or "linear"; see `_arrival_scale`
    size : float
    dpi : int

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    grid = arrival.grid
    X, Y = grid.meshgrid()
    minutes = arrival.seconds / 60.0
    lo, hi = arrival.t_first / 60.0, arrival.t_last / 60.0
    norm, ticks = _arrival_scale(arrival, scale)
    if ticks is None:
        ticks = np.linspace(lo, hi, 9)

    fig, ax = plt.subplots(figsize=(size, size * grid.ny / grid.nx))
    image = ax.imshow(minutes, origin="lower", extent=grid.extent, cmap="magma_r",
                      norm=norm, interpolation="nearest", zorder=1)
    field = np.where(np.isfinite(minutes), minutes, hi + 1.0)
    if contour_minutes is None:
        # Evenly spaced in colour, so the contours land where the colour
        # axis resolves rather than bunching inside the longest interval.
        levels = np.unique(np.asarray(norm.inverse(np.linspace(0.0, 1.0, 12)[1:-1]),
                                      dtype=float))
        spacing = "even in colour"
    else:
        first = np.ceil(lo / contour_minutes) * contour_minutes
        levels = np.arange(first, hi + 1e-9, contour_minutes)
        spacing = f"every {contour_minutes:g} min"
    levels = levels[(levels > lo) & (levels < hi)]
    if levels.size:
        unit, _, fmt = _elapsed_format(levels, hi - lo)
        cs = ax.contour(X, Y, field, levels=levels, colors="white", linewidths=0.7,
                        alpha=0.9, zorder=2)
        ax.clabel(cs, fmt=lambda v: fmt(v), fontsize=7, inline=True)
        contour_unit = f", labelled in {unit} elapsed"
    else:
        contour_unit = ""
    colours = _observation_colours(len(arrival.perimeters))
    # On a long chain the intermediate outlines are a thicket over the field
    # they are meant to annotate, so they are drawn back and the two that
    # bound the chain are left prominent.
    crowded = len(arrival.perimeters) > 6
    for k, (perimeter, colour) in enumerate(zip(arrival.perimeters, colours)):
        ends = k in (0, len(arrival.perimeters) - 1)
        _draw_geometry(ax, perimeter.geometry, color=colour, linestyle="--",
                       linewidth=1.3 if ends or not crowded else 0.55,
                       alpha=1.0 if ends or not crowded else 0.5, zorder=3)
    _draw_ellipse_glyph(ax, _segment_ellipse(arrival, arrival.t_first), grid,
                        arrival.last_mask,
                        note=f"segment 0 of {len(arrival.segments)}"
                             if len(arrival.segments) > 1 else None)
    _scale_bar(ax, grid)

    # Labelled as clock times rather than elapsed minutes: on a stretched
    # axis the tick spacing is not uniform, and a wall clock is what the
    # observations are quoted in anyway. The exported array stays in seconds.
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, ticks=ticks)
    bar.ax.set_yticklabels(_clock_labels(arrival, ticks), fontsize=7)
    bar.set_label("ignition time, UTC (TIGN_G is seconds from "
                  f"{arrival.simulation_start.strftime('%Y-%m-%d %H:%MZ')})")
    ax.set_xlim(grid.extent[0], grid.extent[1])
    ax.set_ylim(grid.extent[2], grid.extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _chain_legend(fig, arrival, colours, fronts=False)
    described = {"segments": "colour stretched so every evolution gets an equal share",
                 "area": "colour stretched so equal colour is equal burned area",
                 "linear": "colour linear in time"}[scale]
    fig.suptitle(title if title is not None else
                 f"Arrival time, {described}\ncontours {spacing}{contour_unit}"
                 f"\n{_chain_title(arrival)}", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def _panel_times(arrival, times, panels_per_segment):
    """Panel times in seconds from the simulation start."""
    if times is not None:
        out = []
        for t in times:
            if isinstance(t, (int, float, np.integer, np.floating)):
                out.append(float(t))
            else:
                from tign import _aware
                out.append((_aware(t) - arrival.simulation_start).total_seconds())
        if not out:
            raise ValueError("times is empty")
        return out
    n = int(panels_per_segment)
    if n < 1:
        raise ValueError("panels_per_segment must be at least 1")
    out = []
    for seg in arrival.segments:
        for k in range(1, n + 1):
            out.append(seg["t0_s"] + seg["interval_s"] * k / n)
    return out


def _segment_ellipse(arrival, t):
    """The ellipse of the segment containing time t, or the nearest one."""
    for seg in arrival.segments:
        if seg["t0_s"] <= t <= seg["t0_s"] + seg["interval_s"]:
            return seg.get("ellipse")
    seg = arrival.segments[0] if t < arrival.t_first else arrival.segments[-1]
    return seg.get("ellipse")


def _observation_colours(n):
    """The prototype's blue and green for a pair; a viridis ramp for more."""
    if n <= 2:
        return [INITIAL_COLOUR, FINAL_COLOUR][:n]
    cmap = plt.get_cmap("viridis")
    return [cmap(v) for v in np.linspace(0.05, 0.9, n)]


def _chain_legend(fig, arrival, colours, fronts=True, y=None, ncol=None, max_listed=6):
    """
    One legend for a chained figure.

    Up to `max_listed` observations are named individually. Beyond that a
    forty-perimeter series would need a legend taller than the figure, so
    only the earliest and latest are named and the rest are described as
    shaded between them - which is what the colour ramp already shows.
    """
    perimeters = arrival.perimeters
    if len(perimeters) <= max_listed:
        handles = [Line2D([], [], color=c, linestyle="--", linewidth=1.3,
                          label=f"observed {_stamp(p)}")
                   for p, c in zip(perimeters, colours)]
    else:
        handles = [
            Line2D([], [], color=colours[0], linestyle="--", linewidth=1.3,
                   label=f"observed {_stamp(perimeters[0])} (earliest)"),
            Line2D([], [], color=colours[len(colours) // 2], linestyle="--", linewidth=1.3,
                   label=f"{len(perimeters) - 2} intermediate observations, shaded earliest to latest"),
            Line2D([], [], color=colours[-1], linestyle="--", linewidth=1.3,
                   label=f"observed {_stamp(perimeters[-1])} (latest)"),
        ]
    if fronts:
        handles += [
            Line2D([], [], color=FRONT_COLOUR, linewidth=2.2, label="evolved front"),
            Patch(facecolor=BURNED_COLOUR, label="burned since first observation"),
            Patch(facecolor=BURNED_AT_START_COLOUR, label="burned before"),
        ]
    if ncol is None:
        ncol = min(len(handles), 4)
    if y is None:
        y = -0.02 * np.ceil(len(handles) / ncol) - 0.02
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, y))


def _chain_title(arrival):
    metrics = {seg["metric"] for seg in arrival.segments}
    metric = next(iter(metrics)) if len(metrics) == 1 else f"{len(metrics)} metrics across segments"
    return (f"{_stamp(arrival.perimeters[0])} → {_stamp(arrival.perimeters[-1])}, "
            f"{len(arrival.segments)} segment(s), "
            f"{(arrival.t_last - arrival.t_first) / 3600:.1f} h\n{metric}")


def _timedelta(seconds):
    from datetime import timedelta
    return timedelta(seconds=float(seconds))


# ----------------------------------------------------------------------
# Animations
# ----------------------------------------------------------------------

def animate(evolution, path, num_frames=40, fps=8, hold_frames=0, size=6.5,
            dpi=100, title=None):
    """
    The filmstrip as a GIF: one frame per progress level from 0 to 1.

    Parameters:
    -----------
    evolution : GaugeEvolution
    path : str
        Output .gif
    num_frames : int
        Frames from s = 0 to s = 1 inclusive
    fps : int
        Playback rate
    hold_frames : int
        Extra copies of the first and last frame, so the endpoints can be
        read before the loop restarts. Zero by default: the loop then runs
        at a constant rate, which is what the interpolation itself does
    size : float
        Figure width in inches
    dpi : int
    title : str, optional

    Returns:
    --------
    path : str
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    levels = np.linspace(0.0, 1.0, int(num_frames))
    sequence = ([0] * int(hold_frames) + list(range(len(levels)))
                + [len(levels) - 1] * int(hold_frames))
    grid = evolution.grid
    fig, ax = plt.subplots(figsize=(size, size * grid.ny / grid.nx + 1.6))
    # A GIF is saved without bbox_inches="tight", so the title and legend
    # have to fit inside the canvas: reserve margins and anchor within them.
    fig.subplots_adjust(top=0.85, bottom=0.13, left=0.03, right=0.97)
    _legend(fig, evolution, y=0.01, ncol=3)
    fig.suptitle(title if title is not None else _default_title(evolution), fontsize=9)
    start_j, end_j = evolution.endpoint_jaccard()

    def draw(index):
        level = levels[index]
        ax.clear()
        draw_panel(ax, evolution, max(level, 1e-9), wavelet=True, scale_bar=True)
        label = f"s = {level:.2f}   {_minutes(evolution, level)}"
        if index == len(levels) - 1:
            label += f"   final match: Jaccard {end_j:.3f}"
        ax.set_title(label, fontsize=11)
        return []

    anim = FuncAnimation(fig, draw, frames=sequence, blit=False)
    anim.save(path, writer=PillowWriter(fps=int(fps)), dpi=dpi)
    plt.close(fig)
    return path


def chain_animate(arrival, path, frames_per_segment=8, fps=6, hold_frames=0,
                  times=None, size=6.5, dpi=100, title=None):
    """
    The chained series as a GIF: one frame per instant, the clock in the title.

    Frames are spaced evenly *within* each segment rather than uniformly in
    time, because real intervals range from minutes to days and a uniform
    clock would spend most of the animation watching nothing happen. The
    time label carries the jumps.

    The arrival-time field itself is continuous across the joins - one
    raster, with each segment's final instant equal to the next one's
    first - so nothing in the animation should suggest the front stops at
    an observation. `hold_frames` is therefore 0 by default; raise it to
    dwell on each observation and read the Jaccard in the title.

    Parameters:
    -----------
    arrival : ArrivalTime
    path : str
        Output .gif
    frames_per_segment : int
        Frames between consecutive observations (the observation itself is
        the last of them)
    fps : int
    hold_frames : int
        Extra copies of every observation frame. Zero by default, so the
        front never appears to pause at a join; a few frames make each
        observation legible at the cost of that illusion
    times : sequence, optional
        Explicit frame times instead - seconds from the simulation start, or
        timezone-aware datetimes / ISO strings
    size : float
    dpi : int
    title : str, optional

    Returns:
    --------
    path : str
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    obs_times = arrival.observation_times
    if times is not None:
        seconds = _panel_times(arrival, times, 1)
    else:
        seconds = [arrival.t_first]
        for seg in arrival.segments:
            n = max(int(frames_per_segment), 1)
            seconds += [seg["t0_s"] + seg["interval_s"] * k / n for k in range(1, n + 1)]

    def is_observation(t):
        return next((i for i, ot in enumerate(obs_times)
                     if abs(ot - t) <= 1e-6 * max(abs(t), 1.0)), None)

    sequence = []
    for index, t in enumerate(seconds):
        sequence.append(index)
        if is_observation(t) is not None:
            sequence += [index] * int(hold_frames)

    grid = arrival.grid
    colours = _observation_colours(len(arrival.perimeters))
    n_legend = min(len(arrival.perimeters), 3) + 3
    legend_rows = int(np.ceil(n_legend / 2))
    fig, ax = plt.subplots(figsize=(size, size * grid.ny / grid.nx + 1.2 + 0.22 * legend_rows))
    fig.subplots_adjust(top=0.86, bottom=0.04 + 0.038 * legend_rows, left=0.03, right=0.97)
    _chain_legend(fig, arrival, colours, y=0.01, ncol=2)
    fig.suptitle(title if title is not None else _chain_title(arrival), fontsize=9)
    masks = {}

    def draw(index):
        t = seconds[index]
        ax.clear()
        draw_chain_panel(ax, arrival, t, colours=colours, wavelet=True, scale_bar=True)
        when = arrival.simulation_start + _timedelta(t)
        label = f"{when:%Y-%m-%d %H:%M:%SZ}   +{(t - arrival.t_first) / 60:.0f} min"
        i = is_observation(t)
        if i is not None:
            if i not in masks:
                masks[i] = arrival.perimeters[i].mask(grid)
            burned = np.isfinite(arrival.seconds) & (arrival.seconds <= t)
            label += f"   observed: Jaccard {_jaccard(burned, masks[i]):.3f}"
        ax.set_title(label, fontsize=10)
        return []

    anim = FuncAnimation(fig, draw, frames=sequence, blit=False)
    anim.save(path, writer=PillowWriter(fps=int(fps)), dpi=dpi)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Pieces
# ----------------------------------------------------------------------

def _contourable(evolution):
    """
    The progress field with the outside filled, so its level sets close.

    NaN outside the final perimeter would leave the s = 1 contour open along
    the perimeter. Setting the outside just above 1 puts that contour on the
    final perimeter's rasterised edge, which is what "exact final match"
    looks like when drawn.
    """
    s = np.array(evolution.progress, dtype=float, copy=True)
    s[~evolution.mask_end] = 1.0 + 1e-6
    return s


def _draw_geometry(ax, geometry, **kwargs):
    """Plot the rings of a (Multi)Polygon."""
    polygons = getattr(geometry, "geoms", [geometry])
    for polygon in polygons:
        if polygon.is_empty:
            continue
        for ring in (polygon.exterior, *polygon.interiors):
            x, y = ring.xy
            ax.plot(x, y, **kwargs)


def _draw_wavelet(ax, evolution, grid, fraction=0.16):
    """The glyph for one evolution; see `_draw_ellipse_glyph`."""
    _draw_ellipse_glyph(ax, getattr(evolution.metric, "ellipse", None), grid,
                        evolution.mask_end, fraction=fraction)


def _draw_ellipse_glyph(ax, ellipse, grid, occupied, fraction=0.16, note=None):
    """
    The fire ellipse glyph in whichever corner has the least fire in it.

    Drawn at the size it would reach in a time chosen so the head advance is
    `fraction` of the panel width, with that time printed, so the glyph
    carries a magnitude and not just a shape. The ellipse is asymmetric about
    its ignition point, so it is positioned by its bounding box rather than
    by that point, and the caption sits on the side facing the panel's
    interior so nothing leaves the axes. Isotropic metrics get no glyph -
    there is no heading to show. `note` is appended to the caption, to say
    which segment's ellipse this is when the figure covers several.
    """
    if ellipse is None or ellipse.is_isotropic:
        return
    left, right, bottom, top = grid.extent
    width, height = right - left, top - bottom
    t = fraction * width / ellipse.head
    ex, ey = ellipse.outline(t=t, n=180)
    box_w, box_h = ex.max() - ex.min(), ey.max() - ey.min()
    pad_x, pad_y = 0.05 * width, 0.05 * height
    caption_h = 0.11 * height

    # Candidate corners as the bounding box's lower-left, with the caption's
    # space reserved on the interior side; pick the emptiest. The lower right
    # is the scale bar's, so it is not a candidate.
    corners = {
        "upper-left": (left + pad_x, top - pad_y - box_h),
        "lower-left": (left + pad_x, bottom + pad_y + caption_h),
        "upper-right": (right - pad_x - box_w, top - pad_y - box_h),
    }
    # The caption always sits below the glyph, so a lower corner reserves
    # room for it against the edge and an upper corner takes it from the
    # interior; either way it stays inside the axes.
    X, Y = grid.meshgrid()
    occupancy = {}
    for name, (bx, by) in corners.items():
        region = ((X >= bx) & (X <= bx + box_w) &
                  (Y >= by - caption_h) & (Y <= by + box_h))
        occupancy[name] = int((region & occupied).sum())
    name = min(corners, key=lambda n: (occupancy[n], list(corners).index(n)))
    bx, by = corners[name]
    x0, y0 = bx - ex.min(), by - ey.min()

    ax.fill(x0 + ex, y0 + ey, color=FRONT_COLOUR, alpha=0.18, zorder=5)
    ax.plot(x0 + ex, y0 + ey, color=FRONT_COLOUR, linewidth=1.0, zorder=5)
    ux, uy = ellipse.uhat
    ax.annotate("", xy=(x0 + ux * ellipse.head * t, y0 + uy * ellipse.head * t),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=FRONT_COLOUR, lw=1.4), zorder=6)
    ax.plot([x0], [y0], marker="o", markersize=3, color=FRONT_COLOUR, zorder=6)
    # Aligned to the panel edge the glyph sits against, so a caption wider
    # than the glyph grows inward rather than out of the axes.
    on_left = name.endswith("left")
    caption = (f"{ellipse.head:.2f} m/s @ {ellipse.bearing:.0f}°, LB {ellipse.lb:.1f}\n"
               f"wavelet after {t / 60:.0f} min")
    if note:
        caption += f"\n{note}"
    ax.text(bx if on_left else bx + box_w, by - 0.015 * height, caption,
            fontsize=7, ha="left" if on_left else "right", va="top",
            color=FRONT_COLOUR, zorder=6)


def _scale_bar(ax, grid):
    """A round-number scale bar in the lower-right corner."""
    width = grid.extent[1] - grid.extent[0]
    height = grid.extent[3] - grid.extent[2]
    length = _round_step(width / 5.0)
    x1 = grid.extent[1] - 0.05 * width
    x0 = x1 - length
    y = grid.extent[2] + 0.06 * height
    ax.plot([x0, x1], [y, y], color="black", linewidth=2.5, solid_capstyle="butt",
            zorder=6)
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    ax.text(0.5 * (x0 + x1), y + 0.015 * height, label, ha="center", va="bottom",
            fontsize=8, zorder=6)


def _legend(fig, evolution, y=-0.04, ncol=None):
    """One legend for the whole figure, below the panels."""
    handles = [
        Line2D([], [], color=INITIAL_COLOUR, linestyle="--", linewidth=1.4,
               label=f"observed {_stamp(evolution.initial)}"),
        Line2D([], [], color=FINAL_COLOUR, linestyle="--", linewidth=1.4,
               label=f"observed {_stamp(evolution.final)}"),
        Line2D([], [], color=FRONT_COLOUR, linewidth=2.2, label="evolved front"),
        Patch(facecolor=BURNED_COLOUR, label="burned this interval"),
        Patch(facecolor=BURNED_AT_START_COLOUR, label="burned before"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles) if ncol is None else ncol, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, y))


def _default_title(evolution):
    return (f"{_stamp(evolution.initial)} → {_stamp(evolution.final)} "
            f"({evolution.interval / 60:.0f} min)\n{evolution.metric.describe()}")


def _stamp(perimeter):
    return perimeter.time.strftime("%Y-%m-%d %H:%MZ") if perimeter.time else "untimed"


def _minutes(evolution, level):
    try:
        return f"+{level * evolution.interval / 60:.0f} min"
    except ValueError:
        return ""


def _round_step(value):
    """The nearest 1, 2 or 5 x 10^n at or below a value."""
    if value <= 0:
        return 1.0
    power = 10.0 ** np.floor(np.log10(value))
    for m in (5.0, 2.0, 1.0):
        if m * power <= value:
            return m * power
    return power


def _jaccard(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def main(argv=None):
    """Draw the filmstrip for a perimeter pair given as shapefiles."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw the filmstrip row, and optionally the arrival-time "
                    "field and the elliptical-versus-Euclidean comparison, for "
                    "a perimeter pair.")
    parser.add_argument("initial")
    parser.add_argument("final")
    parser.add_argument("--out", default="filmstrip.png", help="filmstrip PNG")
    parser.add_argument("--arrival", default=None, help="arrival-time map PNG")
    parser.add_argument("--comparison", default=None, help="comparison filmstrip PNG")
    parser.add_argument("--gif", default=None, help="animated GIF of the evolution")
    parser.add_argument("--frames", type=int, default=40, help="frames in the GIF")
    parser.add_argument("--fps", type=int, default=8, help="GIF playback rate")
    parser.add_argument("--hold-frames", type=int, default=0,
                        help="extra copies of the first and last GIF frame "
                             "(default 0, a constant-rate loop)")
    parser.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument("--cell-size", type=float, default=25.0)
    parser.add_argument("--padding", type=float, default=0.1)
    parser.add_argument("--lb", type=float, default=None)
    parser.add_argument("--wind-mps", type=float, default=None)
    parser.add_argument("--infer-from-geometry", action="store_true")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    from evolution import GaugeEvolution
    from perimeter import PerimeterSeries

    series = PerimeterSeries.from_shapefiles([args.initial, args.final])
    grid = series.common_grid(cell_size=args.cell_size, padding=args.padding)
    series = series.on_grid(grid)
    shared = dict(lb=args.lb, wind_mps=args.wind_mps,
                  infer_from_geometry=args.infer_from_geometry)
    evolution = GaugeEvolution(series[0], series[1], grid, **shared)
    print("  metric:", evolution.metric_provenance)

    filmstrip(evolution, levels=args.levels, path=args.out, title=args.title, dpi=args.dpi)
    print("  wrote", args.out)
    if args.arrival:
        arrival_map(evolution, path=args.arrival, dpi=args.dpi)
        print("  wrote", args.arrival)
    if args.gif:
        animate(evolution, args.gif, num_frames=args.frames, fps=args.fps,
                hold_frames=args.hold_frames, title=args.title)
        print("  wrote", args.gif)
    if args.comparison:
        from validation import MetricComparison
        comparison = MetricComparison(series[0], series[1], grid, **shared)
        comparison_filmstrip(comparison, levels=args.levels, path=args.comparison,
                             dpi=args.dpi)
        print("  wrote", args.comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
