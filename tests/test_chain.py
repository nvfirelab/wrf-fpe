"""
Tests for chain mode: a YAML of evolutions, run in order, combined into one
field and one image.

The perimeters are the elliptically grown series written out as shapefiles
in the kmz_to_shapefile schema, so the YAML is exercised exactly as a user
would write it - relative paths, ISO times, per-evolution overrides.
"""

import os
from datetime import timedelta

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pytest

from chain import load_config, main, run
from figures import chain_arrival_map, chain_filmstrip
from tign import ArrivalTime, sfire_init_check


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def chain_files(tmp_path_factory, elliptical_growth):
    """
    The three elliptical perimeters as WGS84 shapefiles, plus their times.

    Returns:
    --------
    info : dict
        dir, paths (in order), times (in order)
    """
    import geopandas as gpd
    from pyproj import CRS, Transformer
    from shapely.ops import transform

    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    to_wgs = Transformer.from_crs(grid.crs, CRS.from_epsg(4326), always_xy=True).transform
    out = tmp_path_factory.mktemp("chain")
    paths, times = [], []
    for p in series:
        name = str(out / f"perimeter_{p.time:%Y%m%dT%H%M%SZ}.shp")
        gpd.GeoDataFrame([{
            "time_iso": p.time.isoformat(), "speed_mps": 0.30, "dir_deg": 60.0,
            "geometry": transform(to_wgs, p.geometry),
        }], crs="EPSG:4326").to_file(name)
        paths.append(name)
        times.append(p.time)
    return {"dir": str(out), "paths": paths, "times": times}


def _yaml(chain_files, body, name="chain.yaml"):
    """
    Write a YAML body, stripping the test function's own indentation.

    The evolutions block is generated at column 0 and spliced in, so a plain
    textwrap.dedent would find no common indent; instead the first non-blank
    line sets the indent to strip and lines with less keep what they have.
    """
    lines = body.splitlines()
    first = next(l for l in lines if l.strip())
    indent = len(first) - len(first.lstrip())
    out = [l[indent:] if l[:indent].strip() == "" else l for l in lines]
    path = os.path.join(chain_files["dir"], name)
    with open(path, "w") as handle:
        handle.write("\n".join(out).strip("\n") + "\n")
    return path


def _evolutions_block(chain_files, indices=((0, 1), (1, 2)), extra=""):
    lines = []
    for a, b in indices:
        pa, pb = (os.path.basename(chain_files["paths"][i]) for i in (a, b))
        ta, tb = (chain_files["times"][i].strftime("%Y-%m-%dT%H:%M:%SZ") for i in (a, b))
        lines.append(f"  - initial: {{path: {pa}, time: {ta}}}\n"
                     f"    final:   {{path: {pb}, time: {tb}}}{extra}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def full_config(chain_files):
    path = _yaml(chain_files, f"""
        grid: {{cell_size: 25}}
        metric: {{lb: 3.0}}
        simulation_start: 2024-07-23T00:00:00Z
        evolutions:
        {_evolutions_block(chain_files)}
        outputs:
          filmstrip: out_filmstrip.png
          arrival_map: out_arrival.png
          report: out_report.txt
          netcdf: out_tign.nc
          dt: 2.0
        figure: {{panels_per_segment: 2, dpi: 40}}
        validate: true
        """, name="full.yaml")
    return load_config(path)


@pytest.fixture(scope="module")
def full_result(full_config):
    return run(full_config, log=lambda *_: None)


# ----------------------------------------------------------------------
# Reading the file
# ----------------------------------------------------------------------

def test_config_resolves_paths_and_parses_times(full_config, chain_files):
    evs = full_config["evolutions"]
    assert len(evs) == 2
    assert evs[0]["initial"]["path"] == chain_files["paths"][0]
    assert evs[1]["final"]["path"] == chain_files["paths"][2]
    assert evs[0]["initial"]["time"] == chain_files["times"][0]
    assert evs[0]["initial"]["time"].tzinfo is not None
    assert full_config["simulation_start"].isoformat() == "2024-07-23T00:00:00+00:00"
    assert full_config["outputs"]["filmstrip"] == os.path.join(chain_files["dir"], "out_filmstrip.png")
    assert full_config["evolutions"][0]["options"] == {"lb": 3.0}


def test_per_evolution_options_override_the_defaults(chain_files):
    path = _yaml(chain_files, f"""
        metric: {{lb: 3.0}}
        evolutions:
        {_evolutions_block(chain_files, indices=((0, 1),))}
        {_evolutions_block(chain_files, indices=((1, 2),), extra=chr(10) + "    lb: 5.0")}
        """, name="override.yaml")
    config = load_config(path)
    assert config["evolutions"][0]["options"]["lb"] == 3.0
    assert config["evolutions"][1]["options"]["lb"] == 5.0


def test_a_gap_between_evolutions_is_refused(chain_files):
    path = _yaml(chain_files, f"""
        evolutions:
        {_evolutions_block(chain_files, indices=((0, 1),))}
          - initial: {{path: {os.path.basename(chain_files["paths"][1])}, time: 2024-07-23T01:45:00Z}}
            final:   {{path: {os.path.basename(chain_files["paths"][2])}, time: 2024-07-23T02:00:00Z}}
        """, name="gap.yaml")
    with pytest.raises(ValueError, match="has to be continuous"):
        load_config(path)


def test_missing_evolutions_or_paths_are_refused(chain_files):
    with pytest.raises(ValueError, match="'evolutions'"):
        load_config(_yaml(chain_files, "grid: {cell_size: 25}\n", name="empty.yaml"))
    with pytest.raises(ValueError, match="needs 'initial' and 'final'"):
        load_config(_yaml(chain_files, "evolutions:\n  - {initial: {path: a.shp}}\n",
                          name="half.yaml"))
    with pytest.raises(ValueError, match="unknown metric option"):
        load_config(_yaml(chain_files, f"""
            metric: {{ellipse: 3}}
            evolutions:
            {_evolutions_block(chain_files, indices=((0, 1),))}
            """, name="badopt.yaml"))


def test_bare_timestamps_are_read_as_utc(chain_files):
    """PyYAML hands back a naive datetime for a timestamp without a zone."""
    p0, p1 = (os.path.basename(p) for p in chain_files["paths"][:2])
    path = _yaml(chain_files, f"""
        evolutions:
          - initial: {{path: {p0}, time: 2024-07-23 01:00:00}}
            final:   {{path: {p1}, time: 2024-07-23 01:30:00}}
        """, name="naive.yaml")
    config = load_config(path)
    assert config["evolutions"][0]["initial"]["time"] == chain_files["times"][0]


# ----------------------------------------------------------------------
# Running it
# ----------------------------------------------------------------------

def test_run_evolves_every_pair_and_chains_them(full_result, chain_files):
    assert len(full_result["evolutions"]) == 2
    arrival = full_result["arrival"]
    assert isinstance(arrival, ArrivalTime)
    assert len(arrival.segments) == 2
    assert len(arrival.perimeters) == 3
    assert arrival.t_first == pytest.approx(3600.0)
    assert arrival.t_last == pytest.approx(7200.0)
    for seg in arrival.segments:
        assert seg["endpoint_jaccard"][1] == pytest.approx(1.0)
        assert seg["monotonicity_violations"] == 0


def test_run_writes_every_requested_output(full_result, full_config):
    written = full_result["written"]
    for key in ("filmstrip", "arrival_map", "report", "netcdf"):
        assert key in written
        assert os.path.exists(written[key])
        assert os.path.getsize(written[key]) > 0
    with netCDF4.Dataset(written["netcdf"]) as ds:
        tign = np.asarray(ds.variables["TIGN_G"][0])
    assert sfire_init_check(tign, 2.0)["passes"]
    report = open(written["report"]).read()
    assert "segment 0" in report and "segment 1" in report
    assert "Elliptical versus Euclidean" in report      # validate: true
    assert "init_no_fire check passes" in report


def test_validate_runs_a_comparison_per_pair(full_result):
    assert len(full_result["comparisons"]) == 2
    for comparison in full_result["comparisons"]:
        assert comparison.test.metric.is_anisotropic


def test_yaml_times_override_the_shapefile_times(chain_files):
    """The file says 01:00 and 01:30; the YAML says an hour later, and wins."""
    p0, p1 = (os.path.basename(p) for p in chain_files["paths"][:2])
    path = _yaml(chain_files, f"""
        metric: {{lb: 3.0}}
        evolutions:
          - initial: {{path: {p0}, time: 2024-07-23T02:00:00Z}}
            final:   {{path: {p1}, time: 2024-07-23T02:30:00Z}}
        """, name="shift.yaml")
    result = run(load_config(path), log=lambda *_: None)
    ev = result["evolutions"][0]
    assert ev.initial.time == chain_files["times"][0] + timedelta(hours=1)
    assert ev.interval == pytest.approx(1800.0)


def test_a_single_evolution_is_the_pair_case(chain_files):
    path = _yaml(chain_files, f"""
        metric: {{lb: 3.0}}
        evolutions:
        {_evolutions_block(chain_files, indices=((0, 1),))}
        outputs: {{filmstrip: single.png}}
        figure: {{dpi: 40}}
        """, name="single.yaml")
    result = run(load_config(path), log=lambda *_: None)
    assert len(result["arrival"].segments) == 1
    assert os.path.exists(result["written"]["filmstrip"])


def test_simulation_start_defaults_to_the_first_observation(chain_files):
    path = _yaml(chain_files, f"""
        metric: {{lb: 3.0}}
        evolutions:
        {_evolutions_block(chain_files, indices=((0, 1),))}
        """, name="nostart.yaml")
    result = run(load_config(path), log=lambda *_: None)
    assert result["arrival"].simulation_start == chain_files["times"][0]
    assert result["arrival"].t_first == 0.0


def test_array_outputs_need_dt(chain_files):
    path = _yaml(chain_files, f"""
        metric: {{lb: 3.0}}
        evolutions:
        {_evolutions_block(chain_files, indices=((0, 1),))}
        outputs: {{netcdf: x.nc}}
        """, name="nodt.yaml")
    with pytest.raises(ValueError, match="outputs.dt"):
        run(load_config(path), log=lambda *_: None)


def test_euclidean_metric_can_be_requested_per_evolution(chain_files):
    path = _yaml(chain_files, f"""
        metric: {{metric: euclidean}}
        evolutions:
        {_evolutions_block(chain_files, indices=((0, 1),))}
        validate: true
        """, name="euclid.yaml")
    result = run(load_config(path), log=lambda *_: None)
    assert not result["evolutions"][0].metric.is_anisotropic
    assert result["comparisons"] == []
    assert "validation skipped" in result["report"]


def test_cli_runs_a_file(full_config, chain_files, capsys):
    code = main([os.path.join(chain_files["dir"], "full.yaml")])
    assert code == 0
    out = capsys.readouterr().out
    assert "segment 0" in out and "wrote filmstrip" in out


# ----------------------------------------------------------------------
# The chain figures
# ----------------------------------------------------------------------

def test_chain_filmstrip_lands_on_each_observation(full_result):
    arrival = full_result["arrival"]
    fig = chain_filmstrip(arrival, panels_per_segment=2, dpi=40)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 4
    # Panels 2 and 4 fall on observations and say so with a Jaccard of 1.
    boxed = [t.get_text() for ax in visible for t in ax.texts if "Jaccard" in t.get_text()]
    assert len(boxed) == 2
    assert all("1.000" in text for text in boxed)


def test_chain_filmstrip_accepts_explicit_times_and_wraps(full_result):
    arrival = full_result["arrival"]
    times = ["2024-07-23T01:10:00Z", 4200.0, "2024-07-23T01:40:00Z",
             "2024-07-23T01:50:00Z", "2024-07-23T02:00:00Z"]
    fig = chain_filmstrip(arrival, times=times, max_columns=3, dpi=40)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 5
    assert len(fig.axes) == 6            # 2 rows x 3 columns, one blank
    with pytest.raises(ValueError, match="empty"):
        chain_filmstrip(arrival, times=[])


def test_chain_arrival_map_covers_the_whole_series(full_result, tmp_path):
    arrival = full_result["arrival"]
    out = tmp_path / "chain_arrival.png"
    fig = chain_arrival_map(arrival, path=str(out), dpi=40)
    assert out.exists()
    data = fig.axes[0].images[0].get_array()
    finite = data[np.isfinite(data)]
    assert finite.min() == pytest.approx(arrival.t_first / 60.0)
    assert finite.max() == pytest.approx(arrival.t_last / 60.0, abs=1.0)


def test_chain_rejects_an_unknown_arrival_scale_before_solving(chain_files):
    """The figures are written last; a typo should not cost a whole solve."""
    path = _yaml(chain_files, f"""
        metric: {{lb: 3.0}}
        evolutions:
        {_evolutions_block(chain_files)}
        outputs: {{arrival_map: chain_arrival.png}}
        figure: {{arrival_scale: rainbow}}
        """)
    with pytest.raises(ValueError, match="figure.arrival_scale must be one of"):
        load_config(path)


def test_chain_passes_the_arrival_scale_through(chain_files, tmp_path):
    path = _yaml(chain_files, f"""
        metric: {{lb: 3.0}}
        evolutions:
        {_evolutions_block(chain_files)}
        outputs: {{arrival_map: chain_arrival.png}}
        figure: {{arrival_scale: area, dpi: 40}}
        """)
    config = load_config(path)
    assert config["figure"]["arrival_scale"] == "area"
    result = run(config, log=lambda *_: None)
    assert os.path.exists(result["written"]["arrival_map"])
    plt.close("all")


def test_chain_writes_the_animation_when_asked(chain_files):
    from PIL import Image

    path = _yaml(chain_files, f"""
        metric: {{lb: 3.0}}
        evolutions:
        {_evolutions_block(chain_files)}
        outputs: {{animation: chain.gif}}
        figure: {{frames_per_segment: 3, fps: 4, hold_frames: 1, gif_dpi: 40}}
        """, name="gif.yaml")
    result = run(load_config(path), log=lambda *_: None)
    gif = result["written"]["animation"]
    assert os.path.exists(gif)
    with Image.open(gif) as im:
        assert im.format == "GIF"
        assert im.n_frames == 1 + 2 * 3
