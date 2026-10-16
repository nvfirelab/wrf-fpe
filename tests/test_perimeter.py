"""Tests for Perimeter and PerimeterSeries: loading, projection, rasterising."""

import numpy as np
import pytest
from datetime import datetime, timezone

from grid import GridSpec, meridian_convergence
from perimeter import (Perimeter, PerimeterSeries, bearing_of,
                       ros_from_speed_bearing)

UTM10N = "EPSG:32610"


# ----------------------------------------------------------------------
# ROS vector conventions
# ----------------------------------------------------------------------

def test_speed_and_bearing_round_trip():
    """dir_deg is degrees clockwise from north, so north is +v and east is +u."""
    assert ros_from_speed_bearing(1.0, 0.0) == pytest.approx([0.0, 1.0], abs=1e-12)
    assert ros_from_speed_bearing(1.0, 90.0) == pytest.approx([1.0, 0.0], abs=1e-12)
    assert ros_from_speed_bearing(2.0, 225.0) == pytest.approx(
        [-np.sqrt(2), -np.sqrt(2)], abs=1e-12)
    for bearing in (0.0, 37.0, 124.0, 271.0):
        assert bearing_of(ros_from_speed_bearing(3.0, bearing)) == pytest.approx(bearing)


# ----------------------------------------------------------------------
# Loading the kmz_to_shapefile schema
# ----------------------------------------------------------------------

def test_loads_time_and_vector_components(perimeter_files):
    p = Perimeter.from_shapefile(perimeter_files["initial"])
    assert p.time == datetime(2024, 7, 23, 1, 37, 31, tzinfo=timezone.utc)
    assert p.has_ros
    assert p.ros_speed == pytest.approx(0.25, abs=1e-3)
    assert p.ros_bearing == pytest.approx(54.462, abs=0.1)
    assert p.crs.to_epsg() == 4326


def test_falls_back_to_speed_and_bearing(perimeter_files):
    """The second file carries no vel_u/vel_v, only speed_mps and dir_deg."""
    p = Perimeter.from_shapefile(perimeter_files["final"])
    assert p.has_ros
    assert p.ros_speed == pytest.approx(0.31, abs=1e-9)
    assert p.ros_bearing == pytest.approx(60.0, abs=1e-6)


def test_explicit_arguments_override_the_file(perimeter_files):
    when = datetime(2024, 1, 1, tzinfo=timezone.utc)
    p = Perimeter.from_shapefile(perimeter_files["initial"], time=when,
                                 ros=(1.0, 0.0), lb=3.0)
    assert p.time == when
    assert p.ros_bearing == pytest.approx(90.0)
    assert p.lb == 3.0


def test_missing_vector_is_reported_not_faked(perimeter_files, initial_geometry):
    p = Perimeter(geometry=initial_geometry, crs=UTM10N)
    assert not p.has_ros
    assert p.ros_speed is None and p.ros_bearing is None


def test_interval_between_observations(perimeter_files):
    a = Perimeter.from_shapefile(perimeter_files["initial"])
    b = Perimeter.from_shapefile(perimeter_files["final"])
    assert a.interval_to(b) == pytest.approx(4390.0)
    with pytest.raises(ValueError, match="non-increasing"):
        b.interval_to(a)


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------

def test_projection_rotates_the_ros_vector_by_the_convergence(perimeter_files):
    """
    Components are given against true north; the grid's north is rotated by
    the meridian convergence, and the vector has to follow.
    """
    p = Perimeter.from_shapefile(perimeter_files["initial"])
    projected = p.to_crs(UTM10N)
    lon, lat = p.lonlat()
    gamma = meridian_convergence(UTM10N, lon, lat)

    assert abs(gamma) > 0.1                      # a real rotation, worth doing
    assert projected.ros_speed == pytest.approx(p.ros_speed, rel=1e-12)
    assert projected.ros_bearing == pytest.approx(p.ros_bearing - gamma, abs=1e-6)
    assert projected.crs.to_epsg() == 32610


def test_projection_is_idempotent_and_cheap(perimeter_files):
    p = Perimeter.from_shapefile(perimeter_files["initial"]).to_crs(UTM10N)
    assert p.to_crs(UTM10N) is p


def test_area_needs_a_projected_crs(perimeter_files):
    p = Perimeter.from_shapefile(perimeter_files["initial"])
    with pytest.raises(ValueError, match="project the perimeter"):
        p.area_m2
    assert p.to_crs(UTM10N).area_m2 > 500_000.0


# ----------------------------------------------------------------------
# Fields on the shared grid
# ----------------------------------------------------------------------

@pytest.fixture
def loaded(perimeter_files):
    series = PerimeterSeries.from_shapefiles(list(perimeter_files.values()))
    grid = series.common_grid(cell_size=25.0, padding=0.1)
    return series.on_grid(grid), grid


def test_both_perimeters_rasterise_onto_one_grid(loaded):
    """M1: one grid in metres, both observations on it, final containing initial."""
    series, grid = loaded
    initial, final = series[0], series[1]

    assert grid.crs.to_epsg() == 32610
    assert grid.cell_size == 25.0
    mi, mf = initial.mask(grid), final.mask(grid)
    assert mi.shape == mf.shape == grid.shape
    assert mi.any() and mf.any()
    assert (mi & ~mf).sum() == 0                 # the fire only grew
    assert mf.sum() > mi.sum()
    # Rasterised area agrees with the vector area to within the cell size.
    assert mi.sum() * grid.cell_area == pytest.approx(initial.area_m2, rel=0.02)
    assert mf.sum() * grid.cell_area == pytest.approx(final.area_m2, rel=0.02)


def test_grid_contains_both_perimeters_with_room(loaded):
    series, grid = loaded
    for perimeter in series:
        minx, miny, maxx, maxy = perimeter.geometry.bounds
        assert grid.minx < minx and grid.miny < miny
        assert grid.maxx > maxx and grid.maxy > maxy


def test_sdf_signs_agree_with_the_mask(loaded):
    series, grid = loaded
    for perimeter in series:
        mask = perimeter.mask(grid)
        phi = perimeter.sdf(grid)
        assert (phi[mask] < 0).all()
        assert (phi[~mask] > 0).all()


def test_detached_spot_survives_rasterisation(loaded):
    """C5: the final perimeter's detached component must still be there."""
    from scipy.ndimage import label
    series, grid = loaded
    _, n_components = label(series[1].mask(grid))
    assert n_components == 2


def test_fields_are_cached_per_grid(loaded):
    series, grid = loaded
    p = series[0]
    assert p.mask(grid) is p.mask(grid)
    assert p.sdf(grid) is p.sdf(grid)
    other = GridSpec.from_bounds(grid.bounds, cell_size=50.0, crs=grid.crs)
    assert p.mask(other).shape != p.mask(grid).shape


# ----------------------------------------------------------------------
# Gauge source vertices
# ----------------------------------------------------------------------

def test_geometry_vertices_respect_the_spacing(loaded):
    series, grid = loaded
    sx, sy = series[1].boundary_vertices(grid, source="geometry", spacing=25.0)
    assert sx.size == sy.size > 100
    boundary = series[1].geometry.boundary
    from shapely.geometry import Point
    assert max(boundary.distance(Point(x, y)) for x, y in zip(sx, sy)) < 1e-6


def test_geometry_vertex_count_is_independent_of_resolution(loaded):
    """
    Gauge cost is cells x vertices. Mask vertices grow with the grid; geometry
    vertices do not, which is what makes a fine grid affordable.
    """
    series, grid = loaded
    fine = GridSpec.from_bounds(grid.bounds, cell_size=grid.cell_size / 2.0,
                                crs=grid.crs)
    perimeter = series[1]
    coarse_geom = perimeter.boundary_vertices(grid, source="geometry", spacing=25.0)[0]
    fine_geom = perimeter.boundary_vertices(fine, source="geometry", spacing=25.0)[0]
    assert coarse_geom.size == fine_geom.size

    coarse_mask = perimeter.boundary_vertices(grid, source="mask")[0]
    fine_mask = perimeter.boundary_vertices(fine, source="mask")[0]
    assert fine_mask.size > 1.5 * coarse_mask.size


def test_unknown_vertex_source_is_rejected(loaded):
    series, grid = loaded
    with pytest.raises(ValueError, match="unknown vertex source"):
        series[0].boundary_vertices(grid, source="magic")


# ----------------------------------------------------------------------
# Series bookkeeping
# ----------------------------------------------------------------------

def test_series_sorts_by_time_and_pairs_consecutively(perimeter_files):
    series = PerimeterSeries.from_shapefiles(
        [perimeter_files["final"], perimeter_files["initial"]])
    assert [p.time for p in series] == sorted(p.time for p in series)
    pairs = series.pairs()
    assert len(pairs) == 1 and pairs[0][0].time < pairs[0][1].time


def test_nesting_check_passes_on_growth_and_flags_a_retreat(loaded, fire_origin):
    from tests.conftest import wobbly_circle
    series, grid = loaded
    report = series.check_nesting(grid=grid)
    assert len(report) == 1 and report[0]["nested"]
    assert report[0]["fraction_lost"] == 0.0

    shrunken = Perimeter(geometry=wobbly_circle(fire_origin, 300.0), crs=grid.crs,
                         time=series[1].time)
    bad = PerimeterSeries([series[0], shrunken]).check_nesting(grid=grid)
    assert not bad[0]["nested"]
    assert bad[0]["fraction_lost"] > 0.5


def test_vector_companion_files_are_not_loaded_as_perimeters(tmp_path, perimeter_files):
    import shutil
    import os
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        src = perimeter_files["initial"].replace(".shp", suffix)
        if os.path.exists(src):
            shutil.copy(src, tmp_path / f"perimeter_a{suffix}")
            shutil.copy(src, tmp_path / f"perimeter_a_vectors{suffix}")
    series = PerimeterSeries.from_shapefiles(str(tmp_path))
    assert len(series) == 1


# ----------------------------------------------------------------------
# The M1 deliverable: round-trip
# ----------------------------------------------------------------------

def test_rasterised_perimeter_round_trips_through_a_shapefile(loaded, tmp_path):
    """
    Burn each perimeter onto the grid, write the rasterisation back out as a
    shapefile, read it in again, and burn it once more: the two masks must be
    identical. This is what makes the grid inspectable in GIS.
    """
    series, grid = loaded
    for name, perimeter in zip(("initial", "final"), series):
        mask = perimeter.mask(grid)
        path = tmp_path / f"raster_{name}.shp"
        grid.to_geodataframe(mask, {"time_iso": perimeter.time.isoformat()}).to_file(path)

        reloaded = Perimeter.from_shapefile(str(path))
        assert reloaded.crs.equals(grid.crs)
        assert np.array_equal(reloaded.mask(grid), mask)
        assert reloaded.geometry.area == pytest.approx(mask.sum() * grid.cell_area,
                                                       rel=1e-6)


def test_perimeter_round_trips_as_vector(loaded, tmp_path):
    """A Perimeter written out and read back keeps its time and its vector."""
    series, _ = loaded
    path = tmp_path / "perimeter_out.shp"
    series[0].to_geodataframe().to_file(path)

    reloaded = Perimeter.from_shapefile(str(path))
    assert reloaded.time == series[0].time
    assert reloaded.ros_speed == pytest.approx(series[0].ros_speed, rel=1e-6)
    assert reloaded.ros_bearing == pytest.approx(series[0].ros_bearing, abs=1e-4)
    assert reloaded.geometry.area == pytest.approx(series[0].geometry.area, rel=1e-6)


def test_additive_file_reads_as_its_latest_perimeter(perimeter_files, tmp_path):
    """
    A cumulative file holds every earlier perimeter too; the newest one describes it.

    Built the way kmz_to_shapefile's additive mode builds it - the earlier
    record first - so taking the first row would return the earlier time
    and vector with the later footprint.
    """
    import geopandas as gpd
    import pandas as pd

    from perimeter import Perimeter

    early = gpd.read_file(perimeter_files["initial"])
    late = gpd.read_file(perimeter_files["final"])
    cumulative = gpd.GeoDataFrame(pd.concat([early, late], ignore_index=True), crs=early.crs)
    path = str(tmp_path / "perimeter_20240723T025041Z_cumulative.shp")
    cumulative.to_file(path)

    single = Perimeter.from_shapefile(perimeter_files["final"])
    merged = Perimeter.from_shapefile(path)
    assert merged.time == single.time
    assert np.allclose(merged.ros, single.ros)
    assert merged.geometry.area == pytest.approx(
        single.geometry.union(Perimeter.from_shapefile(perimeter_files["initial"]).geometry).area,
        rel=1e-9)
