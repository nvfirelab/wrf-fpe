"""
Tests for TIGN_G export and pair chaining.

The milestone's exit criterion is that SFIRE initialises without rejecting
the array. SFIRE is not available here, so the test is against a
reimplementation of init_no_fire's acceptance loop - the one check that
routine makes on tign - and a wrfinput-shaped NetCDF built to the fire-mesh
layout the registry declares. Everything else is arithmetic that can be
checked exactly: the seconds are t_i + s (t_f - t_i), the chain is a minimum,
and the join between segments is continuous.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pyproj import CRS, Transformer

from evolution import GaugeEvolution
from metrics import EuclideanMetric
from perimeter import Perimeter, PerimeterSeries
from tign import ArrivalTime, sfire_init_check, main

from conftest import GROWTH_SECONDS

SIM_START = datetime(2024, 7, 23, 0, 0, tzinfo=timezone.utc)   # an hour before the first observation
DT = 2.0


@pytest.fixture(scope="module")
def arrival(elliptical_growth):
    """The three-perimeter elliptical series, evolved and chained."""
    return ArrivalTime.from_series(elliptical_growth["series"],
                                   elliptical_growth["grid"], SIM_START, num_steps=8)


@pytest.fixture
def fake_wrfinput(tmp_path, elliptical_growth):
    """
    A wrfinput-shaped NetCDF with a fire mesh over the synthetic fire.

    Ten-fold refinement of a 200 m atmospheric grid, with the trailing sr_x
    columns and sr_y rows zeroed the way WRF leaves its padding, DT as a
    global attribute, and TIGN_G present but blank.

    Returns:
    --------
    info : dict
        path, the mesh's projected coordinates and the refinement
    """
    import netCDF4

    grid = elliptical_growth["grid"]
    sr, dx, n_atm = 10, 200.0, 22
    x0, y0 = grid.minx - 300.0, grid.miny - 300.0
    fx = x0 + (np.arange((n_atm + 1) * sr) + 0.5) * dx / sr
    fy = y0 + (np.arange((n_atm + 1) * sr) + 0.5) * dx / sr
    FX, FY = np.meshgrid(fx, fy)
    to_wgs = Transformer.from_crs(grid.crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = (np.asarray(a) for a in to_wgs.transform(FX, FY))
    for a in (lon, lat):
        a[:, -sr:] = 0.0
        a[-sr:, :] = 0.0

    path = str(tmp_path / "wrfinput_d01")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north_subgrid", lon.shape[0])
        ds.createDimension("west_east_subgrid", lon.shape[1])
        ds.DT = DT
        dims = ("Time", "south_north_subgrid", "west_east_subgrid")
        for name, data in (("FXLONG", lon), ("FXLAT", lat),
                           ("TIGN_G", np.zeros_like(lon))):
            ds.createVariable(name, "f4", dims)[0] = data
    return {"path": path, "FX": FX, "FY": FY, "sr": sr}


# ----------------------------------------------------------------------
# init_no_fire's loop
# ----------------------------------------------------------------------

def test_init_check_is_strict_at_the_end_of_the_first_step():
    """`.not. lfn > 0.` fails at equality, so tign == time_start + dt is rejected."""
    time_now = DT
    assert sfire_init_check(np.array([time_now + 1e-3]), DT)["passes"]
    report = sfire_init_check(np.array([time_now]), DT)
    assert not report["passes"]
    assert report["violations"] == 1
    assert "after the end of the first time step" in report["message"]


def test_init_check_counts_every_bad_node_and_says_how_far_to_shift():
    tign = np.array([[0.0, 1.0, 5.0], [10.0, np.nan, 3.0]])
    report = sfire_init_check(tign, DT)
    assert report["violations"] == 3          # 0, 1 and the NaN; 3, 5, 10 clear 2.0
    assert report["non_finite"] == 1
    assert report["min_tign"] == 0.0
    assert report["needs_shift_s"] == pytest.approx(DT)


def test_init_check_honours_a_nonzero_time_start():
    """time_now is time_start + dt, not just dt."""
    assert sfire_init_check(np.array([100.0]), DT, time_start=0.0)["passes"]
    assert not sfire_init_check(np.array([100.0]), DT, time_start=99.0)["passes"]


# ----------------------------------------------------------------------
# Step 6: a single pair
# ----------------------------------------------------------------------

def test_seconds_are_the_progress_field_scaled_and_offset(pair):
    """t_ign = t_i + s (t_f - t_i), cell for cell, with t_i counted from the run start."""
    initial, final, grid = pair
    ev = GaugeEvolution(initial, final, grid, metric=EuclideanMetric(), num_steps=4)
    start = initial.time - timedelta(hours=2)
    at = ArrivalTime.from_evolution(ev, start)

    expected = 7200.0 + ev.progress * ev.interval
    both = np.isfinite(expected)
    assert np.array_equal(np.isfinite(at.seconds), both)
    assert np.allclose(at.seconds[both], expected[both])
    assert at.t_first == pytest.approx(7200.0)
    assert at.t_last == pytest.approx(7200.0 + ev.interval)


def test_cells_burned_at_the_first_observation_carry_its_time(arrival):
    inside = arrival.first_mask
    assert np.all(arrival.seconds[inside] == arrival.t_first)


def test_cells_outside_the_last_perimeter_have_no_time(arrival):
    assert np.all(np.isnan(arrival.seconds[~arrival.last_mask]))
    assert np.all(np.isfinite(arrival.seconds[arrival.last_mask]))


def test_simulation_start_must_not_postdate_the_first_observation(pair):
    initial, final, grid = pair
    ev = GaugeEvolution(initial, final, grid, metric=EuclideanMetric())
    with pytest.raises(ValueError, match="before the simulation start"):
        ArrivalTime.from_evolution(ev, initial.time + timedelta(seconds=1))


def test_simulation_start_accepts_iso_strings_and_rejects_naive_datetimes(pair):
    initial, final, grid = pair
    ev = GaugeEvolution(initial, final, grid, metric=EuclideanMetric())
    at = ArrivalTime.from_evolution(ev, "2024-07-23T00:00:00Z")
    assert at.simulation_start == SIM_START
    with pytest.raises(ValueError, match="timezone-aware"):
        ArrivalTime.from_evolution(ev, datetime(2024, 7, 23))


# ----------------------------------------------------------------------
# Step 7: chaining
# ----------------------------------------------------------------------

def test_chain_is_continuous_and_monotone_across_the_join(arrival, elliptical_growth):
    """
    The first segment ends where the second begins, in time as well as space.

    Everything the first segment claims is no later than the middle
    observation; everything only the second claims is no earlier. The join
    itself is a tie in the minimum - both segments say the middle time there.
    """
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    m0, m1, m2 = (p.mask(grid) for p in series)
    t_mid = (series[1].time - arrival.simulation_start).total_seconds()

    first_band = m1 & ~m0
    second_band = m2 & ~m1
    assert np.nanmax(arrival.seconds[first_band]) <= t_mid + 1e-6
    assert np.nanmin(arrival.seconds[second_band]) >= t_mid - 1e-6
    # Just inside the middle perimeter the first segment is at s -> 1.
    rim = grid.boundary_cells(m1) & first_band
    assert np.nanmax(arrival.seconds[rim]) == pytest.approx(t_mid, abs=0.05 * GROWTH_SECONDS)


def test_chain_takes_the_earliest_claim(elliptical_growth):
    """
    A cell inside the middle perimeter is claimed by both segments; the first wins.

    The second segment sets s = 0 there, so it claims t_mid; the first claims
    something earlier. The chain must return the earlier one.
    """
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    evs = [GaugeEvolution(a, b, grid, num_steps=4) for a, b in series.pairs()]
    at = ArrivalTime.chain(evs, SIM_START)
    t_mid = (series[1].time - SIM_START).total_seconds()
    m0, m1 = series[0].mask(grid), series[1].mask(grid)
    band = m1 & ~m0
    second_alone = evs[1].arrival_seconds(t_mid)
    assert np.all(second_alone[band] == t_mid)
    assert np.all(at.seconds[band] <= t_mid)
    assert np.mean(at.seconds[band] < t_mid - 1.0) > 0.9


def test_chain_refuses_segments_on_different_grids(elliptical_growth):
    from grid import GridSpec

    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    other = GridSpec(grid.crs, grid.minx, grid.miny, grid.cell_size * 2,
                     grid.nx // 2, grid.ny // 2)
    evs = [GaugeEvolution(series[0], series[1], grid),
           GaugeEvolution(series[1], series[2], other)]
    with pytest.raises(ValueError, match="different grid"):
        ArrivalTime.chain(evs, SIM_START)


def test_chain_refuses_a_gap_in_time(elliptical_growth):
    """Skipping the middle observation is not a chain of consecutive pairs."""
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    evs = [GaugeEvolution(series[0], series[1], grid),
           GaugeEvolution(series[0], series[2], grid)]
    with pytest.raises(ValueError, match="ends at .* but segment 1 begins at"):
        ArrivalTime.chain(evs, SIM_START)


def test_chain_refuses_a_mask_mismatch_at_the_join(elliptical_growth):
    """
    Same time, different geometry: the observations disagree with each other.

    Built by giving the second segment a shrunken copy of the middle
    perimeter as its start, at the same timestamp.
    """
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    middle = series[1]
    shrunk = Perimeter(middle.geometry.buffer(-60.0), middle.crs, time=middle.time,
                       ros=middle.ros, lb=middle.lb)
    evs = [GaugeEvolution(series[0], middle, grid),
           GaugeEvolution(shrunk, series[2], grid)]
    with pytest.raises(ValueError, match="differ in .* cells"):
        ArrivalTime.chain(evs, SIM_START)


def test_from_series_refuses_growth_that_is_not_nested(elliptical_growth):
    """A later perimeter that drops part of an earlier one is a data problem."""
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    last = series[2]
    clipped = Perimeter(last.geometry.difference(series[0].geometry.centroid.buffer(150.0)),
                        last.crs, time=last.time, ros=last.ros, lb=last.lb)
    bad = PerimeterSeries([series[0], series[1], clipped])
    with pytest.raises(ValueError, match="not nested"):
        ArrivalTime.from_series(bad, grid, SIM_START)


# ----------------------------------------------------------------------
# The exported array
# ----------------------------------------------------------------------

def test_tign_g_is_finite_float32_everywhere(arrival):
    tign = arrival.tign_g()
    assert tign.dtype == np.float32
    assert tign.shape == arrival.grid.shape
    assert np.isfinite(tign).all()


def test_outward_extension_continues_the_field_as_scaled_distance(arrival):
    """
    Beyond the last perimeter, tign = t_last + distance / speed.

    Checked against the grid's own signed distance so the test is about the
    continuation, not about the SDF; and the extension rises monotonically
    with distance, which is what makes lfn a usable level-set seed.
    """
    speed = 0.30
    tign = arrival.tign_g(outside="extend", speed=speed)
    outside = ~arrival.last_mask
    phi = arrival.grid.sdf(arrival.last_mask)
    assert np.allclose(tign[outside], arrival.t_last + phi[outside] / speed, rtol=1e-6)
    assert tign[outside].min() > arrival.t_last
    # Inside is untouched by the outward extension.
    assert np.allclose(tign[arrival.last_mask], arrival.seconds[arrival.last_mask], rtol=1e-6)


def test_extension_speed_defaults_to_the_ellipse_head_rate(arrival):
    assert arrival.speed_hint == pytest.approx(0.30)
    assert np.array_equal(arrival.tign_g(), arrival.tign_g(speed=0.30))


def test_euclidean_segments_need_an_explicit_speed(pair):
    initial, final, grid = pair
    ev = GaugeEvolution(initial, final, grid, metric=EuclideanMetric())
    at = ArrivalTime.from_evolution(ev, initial.time - timedelta(hours=1))
    assert at.speed_hint is None
    with pytest.raises(ValueError, match="pass speed="):
        at.tign_g()
    assert np.isfinite(at.tign_g(speed=0.5)).all()


def test_sentinel_must_postdate_the_last_observation(arrival):
    with pytest.raises(ValueError, match="needs sentinel="):
        arrival.tign_g(outside="sentinel")
    with pytest.raises(ValueError, match="not later than the last observation"):
        arrival.tign_g(outside="sentinel", sentinel=arrival.t_last)
    tign = arrival.tign_g(outside="sentinel", sentinel=1.0e6)
    assert np.all(tign[~arrival.last_mask] == np.float32(1.0e6))


def test_inward_extension_ignites_the_centre_first(arrival):
    """A burn-in ramp: deepest inside the first perimeter is earliest."""
    flat = arrival.tign_g(inside="flat")
    ramp = arrival.tign_g(inside="extend")
    inside = arrival.first_mask
    assert np.all(flat[inside] == np.float32(arrival.t_first))
    assert ramp[inside].max() <= arrival.t_first + 1e-3
    assert ramp[inside].min() < arrival.t_first - 100.0
    # Nothing outside the first perimeter changes.
    assert np.array_equal(flat[~inside], ramp[~inside])


def test_unknown_modes_are_rejected(arrival):
    with pytest.raises(ValueError, match="unknown outside"):
        arrival.tign_g(outside="zero")
    with pytest.raises(ValueError, match="unknown inside"):
        arrival.tign_g(inside="ramp")


# ----------------------------------------------------------------------
# The exit criterion
# ----------------------------------------------------------------------

def test_the_array_passes_init_no_fire(arrival):
    """M5's exit criterion, against the reimplemented check."""
    for kwargs in ({}, {"outside": "sentinel", "sentinel": 1.0e6}, {"inside": "extend"}):
        report = arrival.check(DT, **kwargs)
        assert report["passes"]
        assert report["violations"] == 0
        assert report["min_tign"] > DT


def test_a_run_starting_at_the_first_observation_is_rejected_with_the_fix(elliptical_growth):
    """
    With t_i = 0 every cell inside the first perimeter reads 0 <= dt.

    This is the case the write-up warns about, and the message has to say
    how far to move the start rather than just that it failed.
    """
    series, grid = elliptical_growth["series"], elliptical_growth["grid"]
    at = ArrivalTime.from_series(series, grid, series[0].time, num_steps=4)
    with pytest.raises(ValueError, match="SFIRE would reject") as info:
        at.check(DT)
    assert f"at least {DT:.0f} s earlier" in str(info.value)
    report = sfire_init_check(at.tign_g(), DT)
    assert report["violations"] == int(at.first_mask.sum())


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

def test_netcdf_round_trips_the_array_with_coordinates(arrival, tmp_path):
    import netCDF4

    path = arrival.to_netcdf(str(tmp_path / "tign.nc"))
    with netCDF4.Dataset(path) as ds:
        tign = np.asarray(ds.variables["TIGN_G"][0])
        lon = np.asarray(ds.variables["XLONG"][0])
        lat = np.asarray(ds.variables["XLAT"][0])
        assert ds.simulation_start == SIM_START.isoformat()
        assert ds.variables["TIGN_G"].units == "s"
    assert np.array_equal(tign, arrival.tign_g())
    # Row 0 is the southern edge, and longitude increases eastward.
    assert lat[0, 0] < lat[-1, 0]
    assert lon[0, 0] < lon[0, -1]
    assert -123.0 < lon.mean() < -122.0 and 41.0 < lat.mean() < 42.0


def test_geotiff_is_north_up_and_georeferenced(arrival, tmp_path):
    """Sampling the GeoTIFF at a cell centre returns that cell's value."""
    import rasterio

    tign = arrival.tign_g()
    path = arrival.to_geotiff(str(tmp_path / "tign.tif"), tign=tign)
    grid = arrival.grid
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == grid.crs.to_epsg()
        for i, j in ((3, 5), (grid.ny // 2, grid.nx // 2), (grid.ny - 2, grid.nx - 4)):
            x, y = grid.x_coords[j], grid.y_coords[i]
            value = next(src.sample([(x, y)]))[0]
            assert value == pytest.approx(tign[i, j])


def test_wrfinput_gets_the_field_on_its_fire_mesh(arrival, fake_wrfinput):
    """
    Nodes inside the grid read the nearest cell; padding and beyond get the fill.

    And the array as written passes init_no_fire at the file's own DT, which
    is the actual deliverable.
    """
    import netCDF4

    report = arrival.write_wrfinput(fake_wrfinput["path"])
    assert report["passes"] and report["dt"] == DT
    assert report["nodes_inside_grid"] > 0 and report["nodes_filled"] > 0

    tign = arrival.tign_g()
    grid = arrival.grid
    FX, FY, sr = fake_wrfinput["FX"], fake_wrfinput["FY"], fake_wrfinput["sr"]
    with netCDF4.Dataset(fake_wrfinput["path"]) as ds:
        written = np.asarray(ds.variables["TIGN_G"][0])
        assert "perimeter evolution" in ds.TIGN_G_source
    assert written.shape == FX.shape
    assert sfire_init_check(written, DT)["passes"]

    # A node well inside the first perimeter reads t_first...
    i, j = np.unravel_index(np.argmin(np.hypot(FX - grid.x_coords[grid.nx // 3],
                                               FY - grid.y_coords[grid.ny // 3])), FX.shape)
    ci = int(round((FY[i, j] - grid.miny) / grid.cell_size))
    cj = int(round((FX[i, j] - grid.minx) / grid.cell_size))
    assert written[i, j] == pytest.approx(tign[ci, cj])
    # ...and the padding rows and columns hold the fill, not the blank zeros.
    assert np.all(written[-sr:, :] == report["fill"])
    assert np.all(written[:, -sr:] == report["fill"])
    assert report["fill"] == pytest.approx(float(tign.max()))


def test_wrfinput_is_left_untouched_when_the_check_fails(arrival, fake_wrfinput):
    """A DT longer than the lead time fails the check before anything is written."""
    import netCDF4

    with pytest.raises(ValueError, match="not written"):
        arrival.write_wrfinput(fake_wrfinput["path"], dt=arrival.t_first + 1.0)
    with netCDF4.Dataset(fake_wrfinput["path"]) as ds:
        assert np.all(np.asarray(ds.variables["TIGN_G"][0]) == 0.0)
        assert "TIGN_G_source" not in ds.ncattrs()


def test_wrfinput_without_a_fire_mesh_is_refused(arrival, tmp_path):
    import netCDF4

    path = str(tmp_path / "wrfinput_d01")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", 4)
        ds.createDimension("west_east", 4)
        ds.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
    with pytest.raises(KeyError, match="FXLONG"):
        arrival.write_wrfinput(path)


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def test_cli_writes_a_netcdf_from_shapefiles(perimeter_files, tmp_path, capsys):
    import netCDF4

    out = str(tmp_path / "tign.nc")
    code = main([perimeter_files["initial"], perimeter_files["final"],
                 "--simulation-start", "2024-07-22T20:00:00Z", "--dt", "2",
                 "--lb", "3", "--cell-size", "50", "--netcdf", out])
    assert code == 0
    text = capsys.readouterr().out
    assert "init_no_fire check     passes" in text
    with netCDF4.Dataset(out) as ds:
        tign = np.asarray(ds.variables["TIGN_G"][0])
    assert np.isfinite(tign).all()
    assert sfire_init_check(tign, 2.0)["passes"]
