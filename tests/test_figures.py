"""
Tests for the figures: they render headlessly, save, and draw what they claim.

Figures are checked structurally - panel counts, what is drawn on which
axes, that files come out - not pixel by pixel. The one substantive claim,
that the last panel's front is the final perimeter, is checked on the
contour path the panel actually drew.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from shapely.geometry import Point, Polygon

from evolution import GaugeEvolution
from figures import (DEFAULT_LEVELS, arrival_map, comparison_filmstrip, draw_panel,
                     filmstrip, main)
from metrics import EuclideanMetric
from validation import MetricComparison


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def evolution(pair):
    initial, final, grid = pair
    return GaugeEvolution(initial, final, grid, lb=3.0, num_steps=8)


def _front_paths(ax):
    """Vertices of every contour path drawn on an axes, in data coordinates."""
    paths = []
    for artist in ax.collections:
        for path in artist.get_paths():
            if len(path.vertices) > 2:
                paths.append(path.vertices)
    for artist in getattr(ax, "_children", []):
        # Newer matplotlib returns contours as a single ContourSet child.
        if hasattr(artist, "get_paths") and artist not in ax.collections:
            for path in artist.get_paths():
                if len(path.vertices) > 2:
                    paths.append(path.vertices)
    return paths


def test_filmstrip_has_one_panel_per_level(evolution, tmp_path):
    levels = (0.2, 0.5, 0.8, 1.0)
    out = tmp_path / "strip.png"
    fig = filmstrip(evolution, levels=levels, path=str(out), dpi=50)
    assert len(fig.axes) == len(levels)
    assert out.exists() and out.stat().st_size > 1000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_filmstrip_default_levels_end_on_the_final_perimeter(evolution):
    fig = filmstrip(evolution)
    assert DEFAULT_LEVELS[-1] == 1.0
    assert len(fig.axes) == len(DEFAULT_LEVELS)
    last = fig.axes[-1]
    assert any("Jaccard" in t.get_text() for t in last.texts)


def test_last_panel_front_lies_on_the_final_perimeter(evolution):
    """
    The red s = 1 contour is the final perimeter, to within a cell.

    Sampled at every vertex of the drawn contour: each must be within one
    cell diagonal of the observed final geometry's boundary.
    """
    fig, ax = plt.subplots()
    draw_panel(ax, evolution, 1.0, observed=False)
    paths = _front_paths(ax)
    assert paths, "no contour was drawn"
    boundary = evolution.final.geometry.boundary
    tolerance = np.sqrt(2.0) * evolution.grid.cell_size
    far = 0
    for vertices in paths:
        for x, y in vertices[::5]:
            if boundary.distance(Point(x, y)) > tolerance:
                far += 1
    assert far == 0


def test_intermediate_front_sits_between_the_observations(evolution):
    """At s = 0.5 the front is outside the initial perimeter and inside the final."""
    fig, ax = plt.subplots()
    draw_panel(ax, evolution, 0.5, observed=False)
    initial = evolution.initial.geometry
    final = evolution.final.geometry.buffer(evolution.grid.cell_size)
    for vertices in _front_paths(ax):
        for x, y in vertices[::7]:
            p = Point(x, y)
            assert final.covers(p)
            assert not initial.buffer(-evolution.grid.cell_size).covers(p)


def test_panel_draws_the_burned_area_and_both_observations(evolution):
    fig, ax = plt.subplots()
    draw_panel(ax, evolution, 0.5, wavelet=True, scale_bar=True)
    assert any(isinstance(a, AxesImage) for a in ax.images)
    # Observed perimeters are Line2D rings: the initial is one ring, the
    # final is a MultiPolygon so at least two, plus the scale bar's line.
    assert len(ax.lines) >= 4
    assert ax.get_aspect() == 1.0
    assert ax.get_xticks().size == 0


def test_wavelet_glyph_is_skipped_for_an_isotropic_metric(pair):
    initial, final, grid = pair
    euclid = GaugeEvolution(initial, final, grid, metric=EuclideanMetric())
    fig, ax = plt.subplots()
    draw_panel(ax, euclid, 0.5, wavelet=True, observed=False)
    assert not any("wavelet" in t.get_text() for t in ax.texts)
    fig2, ax2 = plt.subplots()
    draw_panel(ax2, GaugeEvolution(initial, final, grid, lb=3.0), 0.5, wavelet=True,
               observed=False)
    assert any("wavelet" in t.get_text() for t in ax2.texts)


def test_wavelet_glyph_stays_inside_the_axes(evolution):
    """The glyph and its caption are placed by bounding box, so nothing leaves the panel."""
    fig, ax = plt.subplots()
    draw_panel(ax, evolution, 0.25, wavelet=True, scale_bar=True)
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    caption = next(t for t in ax.texts if "wavelet" in t.get_text())
    x, y = caption.get_position()
    assert left <= x <= right and bottom <= y <= top
    fig.canvas.draw()
    bbox = caption.get_window_extent().transformed(ax.transData.inverted())
    assert bbox.x0 >= left - 1e-6 and bbox.x1 <= right + 1e-6
    assert bbox.y0 >= bottom - 1e-6


def test_filmstrip_rejects_levels_outside_the_unit_interval(evolution):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        filmstrip(evolution, levels=(0.0, 0.5))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        filmstrip(evolution, levels=(0.5, 1.2))


def test_arrival_map_draws_the_field_with_a_colourbar(evolution, tmp_path):
    out = tmp_path / "arrival.png"
    fig = arrival_map(evolution, path=str(out), dpi=50, t0=3600.0)
    assert out.exists()
    ax = fig.axes[0]
    image = next(a for a in ax.images)
    data = image.get_array()
    finite = data[np.isfinite(data)]
    # Minutes from the simulation start: the first observation is at +60.
    assert finite.min() == pytest.approx(60.0)
    assert finite.max() == pytest.approx(60.0 + evolution.interval / 60.0, abs=0.5)
    assert len(fig.axes) == 2      # map and colourbar


def test_comparison_filmstrip_has_two_rows_and_labels_agreement(pair, tmp_path):
    initial, final, grid = pair
    comparison = MetricComparison(initial, final, grid, lb=3.0, num_steps=4)
    out = tmp_path / "comparison.png"
    fig = comparison_filmstrip(comparison, levels=(0.5, 1.0), path=str(out), dpi=50)
    assert out.exists()
    assert len(fig.axes) == 4
    titles = [ax.get_title() for ax in fig.axes]
    assert sum("agreement" in t for t in titles) == 2
    # The pinned endpoint agrees exactly; the interior does not.
    assert "agreement 1.000" in titles[-1]
    assert "agreement 1.000" not in titles[-2]


def test_cli_writes_all_three_figures(perimeter_files, tmp_path, capsys):
    strip = tmp_path / "strip.png"
    arrival = tmp_path / "arrival.png"
    comparison = tmp_path / "comparison.png"
    code = main([perimeter_files["initial"], perimeter_files["final"],
                 "--out", str(strip), "--arrival", str(arrival),
                 "--comparison", str(comparison), "--lb", "3", "--cell-size", "50",
                 "--levels", "0.5", "1.0", "--dpi", "50"])
    assert code == 0
    assert strip.exists() and arrival.exists() and comparison.exists()
    assert "elliptical" in capsys.readouterr().out


# ----------------------------------------------------------------------
# Animations
# ----------------------------------------------------------------------

def _gif_info(path):
    from PIL import Image

    with Image.open(path) as im:
        assert im.format == "GIF"
        durations = []
        for i in range(im.n_frames):
            im.seek(i)
            durations.append(im.info.get("duration", 0))
        return im.n_frames, durations, im.size


def test_animate_writes_a_gif_that_holds_the_endpoints(evolution, tmp_path):
    from figures import animate

    out = tmp_path / "pair.gif"
    assert animate(evolution, str(out), num_frames=6, fps=5, hold_frames=3, dpi=40) == str(out)
    n_frames, durations, size = _gif_info(out)
    # Pillow merges identical consecutive frames into one longer frame, so the
    # holds show up as duration rather than count: first and last frames long.
    assert n_frames == 6
    assert durations[0] > durations[1] and durations[-1] > durations[-2]
    assert durations[0] == pytest.approx(4 * 200, abs=1)     # (1 + 3 holds) at 5 fps
    assert size[0] > 100 and size[1] > 100


def test_chain_animate_frames_per_segment_and_holds_observations(tmp_path, elliptical_growth):
    from figures import chain_animate
    from tign import ArrivalTime

    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    arrival = ArrivalTime.from_series(series, grid, "2024-07-23T00:00:00Z", num_steps=4)
    out = tmp_path / "chain.gif"
    chain_animate(arrival, str(out), frames_per_segment=4, fps=4, hold_frames=2, dpi=40)
    n_frames, durations, _ = _gif_info(out)
    # 1 + 2 segments x 4 frames = 9 distinct frames; the 3 observation frames
    # (first, join, last) are held, i.e. three times longer than the rest.
    assert n_frames == 9
    held = [d for d in durations if d > 1.5 * min(durations)]
    assert len(held) == 3
    assert durations[0] in held and durations[4] in held and durations[8] in held


class _RaggedArrival:
    """
    Just the attributes `_arrival_scale` reads, with a ragged set of intervals.

    The shape is the real chain's: dozens of short intervals covering most of
    the burned area, then one that is three days long and covers little.
    """

    MINUTES = [3, 6, 12, 14, 7, 12, 23, 53, 21, 9, 51, 8, 3, 9, 15, 5133]

    def __init__(self, minutes=None):
        minutes = np.asarray(self.MINUTES if minutes is None else minutes, dtype=float)
        edges = np.concatenate([[0.0], np.cumsum(minutes)])
        self.observation_times = list(60.0 * edges)
        self.t_first, self.t_last = 60.0 * edges[0], 60.0 * edges[-1]
        # Equal burned area per segment, spread evenly inside each one.
        self.seconds = np.concatenate(
            [np.linspace(60.0 * edges[k], 60.0 * edges[k + 1], 50, endpoint=False)
             for k in range(len(minutes))]).reshape(-1, 1)


def test_arrival_scale_gives_every_evolution_an_equal_share():
    """The complaint the scale exists to answer: one long interval ate the ramp."""
    from figures import _arrival_scale

    arrival = _RaggedArrival()
    n = len(arrival.observation_times) - 1

    linear, _ = _arrival_scale(arrival, "linear")
    segments, ticks = _arrival_scale(arrival, "segments")
    # The last interval is 5133 of 5379 minutes: linearly it owns 95% of the
    # ramp and the other fifteen share what is left.
    minutes = [t / 60.0 for t in arrival.observation_times]
    assert float(linear(minutes[-2])) < 0.06
    assert float(segments(minutes[-2])) == pytest.approx((n - 1) / n, abs=1e-6)
    for k, t in enumerate(minutes):
        assert float(segments(t)) == pytest.approx(k / n, abs=1e-6)
    assert ticks is not None and len(ticks) <= 10


def test_arrival_scale_area_spends_equal_colour_on_equal_area():
    from figures import _arrival_scale

    arrival = _RaggedArrival()
    norm, _ = _arrival_scale(arrival, "area")
    finite = arrival.seconds[np.isfinite(arrival.seconds)] / 60.0
    for quantile in (0.25, 0.5, 0.75):
        assert float(norm(np.quantile(finite, quantile))) == pytest.approx(
            quantile, abs=0.02)


@pytest.mark.parametrize("scale", ["segments", "area", "linear"])
def test_arrival_scale_is_monotone_and_invertible(scale):
    from figures import _arrival_scale

    arrival = _RaggedArrival()
    norm, _ = _arrival_scale(arrival, scale)
    probe = np.linspace(arrival.t_first / 60.0, arrival.t_last / 60.0, 200)
    mapped = np.asarray(norm(probe), dtype=float)
    assert mapped[0] == pytest.approx(0.0) and mapped[-1] == pytest.approx(1.0)
    assert np.all(np.diff(mapped) >= -1e-12)
    back = np.asarray(norm.inverse(np.linspace(0.05, 0.95, 40)), dtype=float)
    assert np.all(np.diff(back) > 0)


def test_arrival_scale_survives_repeated_and_degenerate_knots():
    """Zero-length intervals must not make the mapping non-invertible."""
    from figures import _arrival_scale

    norm, ticks = _arrival_scale(_RaggedArrival([5.0, 0.0, 0.0, 7.0]), "segments")
    probe = np.linspace(0.0, 12.0, 50)
    assert np.all(np.diff(np.asarray(norm(probe), dtype=float)) >= -1e-12)
    # A chain with no span at all falls back to the linear ramp.
    flat = _RaggedArrival([0.0, 0.0])
    assert _arrival_scale(flat, "segments")[1] is None


def test_arrival_scale_rejects_an_unknown_name():
    from figures import _arrival_scale

    with pytest.raises(ValueError, match="scale must be one of"):
        _arrival_scale(_RaggedArrival(), "rainbow")


def test_chain_arrival_map_labels_its_colourbar_with_distinct_clock_times(
        tmp_path, elliptical_growth):
    from figures import chain_arrival_map
    from tign import ArrivalTime

    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    arrival = ArrivalTime.from_series(series, grid, "2024-07-23T00:00:00Z", num_steps=4)
    for scale in ("segments", "area", "linear"):
        out = tmp_path / f"arrival_{scale}.png"
        fig = chain_arrival_map(arrival, path=str(out), scale=scale, dpi=40)
        assert out.exists()
        labels = [t.get_text() for t in fig.axes[-1].get_yticklabels()]
        labels = [t for t in labels if t]
        assert labels and len(set(labels)) == len(labels)
        assert all(label.endswith("Z") for label in labels)
        plt.close(fig)


def test_chain_animate_does_not_hold_observations_by_default(tmp_path, elliptical_growth):
    """The default GIF runs at a constant rate: the field has no pause at a join."""
    from figures import chain_animate
    from tign import ArrivalTime

    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    arrival = ArrivalTime.from_series(series, grid, "2024-07-23T00:00:00Z", num_steps=4)
    out = tmp_path / "chain.gif"
    chain_animate(arrival, str(out), frames_per_segment=4, fps=4, dpi=40)
    n_frames, durations, _ = _gif_info(out)
    assert n_frames == 9
    assert max(durations) == pytest.approx(min(durations), abs=1)


def test_chain_animate_accepts_explicit_times(tmp_path, elliptical_growth):
    from figures import chain_animate
    from tign import ArrivalTime

    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    arrival = ArrivalTime.from_series(series, grid, "2024-07-23T00:00:00Z", num_steps=4)
    out = tmp_path / "chain.gif"
    chain_animate(arrival, str(out), times=[3900.0, 4500.0, "2024-07-23T01:45:00Z"], fps=4,
                  dpi=40)
    n_frames, _, _ = _gif_info(out)
    assert n_frames == 3


def test_long_series_legend_names_only_the_ends(elliptical_growth):
    """Forty observations must not become forty legend rows."""
    from types import SimpleNamespace

    from figures import _chain_legend, _observation_colours

    many = SimpleNamespace(perimeters=list(elliptical_growth["series"]) * 5)
    fig = plt.figure()
    _chain_legend(fig, many, _observation_colours(len(many.perimeters)))
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert len(labels) == 6
    assert any("earliest" in l for l in labels) and any("latest" in l for l in labels)
    assert any("13 intermediate" in l for l in labels)


def test_cli_writes_a_gif(perimeter_files, tmp_path):
    gif = tmp_path / "pair.gif"
    code = main([perimeter_files["initial"], perimeter_files["final"],
                 "--out", str(tmp_path / "strip.png"), "--gif", str(gif),
                 "--frames", "5", "--lb", "3", "--cell-size", "50", "--dpi", "40"])
    assert code == 0
    assert _gif_info(gif)[0] == 5
