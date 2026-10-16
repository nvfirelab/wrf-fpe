"""Tests for GridSpec: square cells, projection sanity, SDF, polygonisation."""

import numpy as np
import pytest
from shapely.geometry import Point

from grid import GridSpec, scale_distortion, utm_crs_for

UTM10N = "EPSG:32610"


def test_utm_zone_selection():
    assert utm_crs_for(-122.44, 41.50).to_epsg() == 32610
    assert utm_crs_for(-116.0, 39.5).to_epsg() == 32611
    assert utm_crs_for(-122.44, -41.50).to_epsg() == 32710


def test_cells_are_square_over_non_square_bounds():
    """A fixed cell *count* over non-square bounds is what corrupts distances."""
    grid = GridSpec.from_bounds((0.0, 0.0, 4000.0, 1000.0), cell_size=25.0, crs=UTM10N)
    x, y = grid.x_coords, grid.y_coords
    assert np.allclose(np.diff(x), 25.0)
    assert np.allclose(np.diff(y), 25.0)
    assert grid.nx > grid.ny
    assert grid.shape == (grid.ny, grid.nx)


def test_geographic_crs_is_rejected():
    with pytest.raises(ValueError, match="projected CRS"):
        GridSpec.from_bounds((-122.5, 41.4, -122.3, 41.6), cell_size=0.001,
                             crs="EPSG:4326")


def test_web_mercator_warns_about_scale():
    """The 1.33 factor the method write-up calls out, caught at construction."""
    assert scale_distortion("EPSG:3857", -122.44, 41.50) == pytest.approx(0.335, abs=0.01)
    assert scale_distortion(UTM10N, -122.44, 41.50) < 1e-3
    with pytest.warns(UserWarning, match="scale distortion"):
        GridSpec.from_bounds((-13630000.0, 5090000.0, -13620000.0, 5100000.0),
                             cell_size=25.0, crs="EPSG:3857")


def test_grid_covers_requested_bounds_with_padding():
    grid = GridSpec.from_bounds((1000.0, 2000.0, 3000.0, 4000.0), cell_size=20.0,
                                crs=UTM10N, padding=0.1)
    assert grid.minx <= 1000.0 - 200.0
    assert grid.miny <= 2000.0 - 200.0
    assert grid.maxx >= 3000.0 + 200.0
    assert grid.maxy >= 4000.0 + 200.0


def test_from_geometries_picks_utm_and_covers_everything():
    ring = Point(-122.44, 41.50).buffer(0.01)
    grid = GridSpec.from_geometries([ring], "EPSG:4326", cell_size=30.0)
    assert grid.crs.to_epsg() == 32610
    assert grid.cell_size == 30.0
    assert grid.nx > 10 and grid.ny > 10


def _circle_grid(radius=600.0, cell=10.0):
    grid = GridSpec.from_bounds((-1200.0, -1200.0, 1200.0, 1200.0), cell_size=cell,
                                crs=UTM10N)
    X, Y = grid.meshgrid()
    return grid, np.hypot(X, Y) < radius


def test_sdf_sign_convention_and_magnitude():
    """Negative inside, positive outside, and metres either way."""
    radius, cell = 600.0, 10.0
    grid, mask = _circle_grid(radius, cell)
    phi = grid.sdf(mask)
    X, Y = grid.meshgrid()
    r = np.hypot(X, Y)

    assert phi[mask].max() < 0.0
    assert phi[~mask].min() > 0.0
    # Away from the boundary the SDF is the analytic distance to the circle,
    # inflated by about half a cell because each transform measures to the
    # nearest cell of the opposite class.
    far = np.abs(r - radius) > 5 * cell
    assert np.abs(phi[far] - (r[far] - radius)).max() <= cell + 1e-9
    subcell = grid.sdf(mask, subcell=True)
    assert np.abs(subcell[far] - (r[far] - radius)).max() <= 0.5 * cell + 1e-9


def test_sdf_zero_contour_sits_on_the_boundary():
    """
    The field steps from -cell to +cell across the boundary, so the contour
    lives between cells; what has to be true is that the sign change happens
    where the circle is.
    """
    radius, cell = 600.0, 10.0
    grid, mask = _circle_grid(radius, cell)
    phi = grid.sdf(mask)
    X, Y = grid.meshgrid()
    r = np.hypot(X, Y)
    crossing = np.zeros(grid.shape, dtype=bool)
    crossing[:, :-1] |= np.sign(phi[:, :-1]) != np.sign(phi[:, 1:])
    crossing[:-1, :] |= np.sign(phi[:-1, :]) != np.sign(phi[1:, :])
    assert np.abs(r[crossing] - radius).max() < 1.5 * cell


def test_sdf_of_empty_and_full_masks():
    grid, _ = _circle_grid()
    empty = np.zeros(grid.shape, dtype=bool)
    assert np.isinf(grid.sdf(empty)).all() and (grid.sdf(empty) > 0).all()
    full = np.ones(grid.shape, dtype=bool)
    assert np.isinf(grid.sdf(full)).all() and (grid.sdf(full) < 0).all()


def test_boundary_cells_form_a_ring_inside_the_mask():
    radius, cell = 600.0, 10.0
    grid, mask = _circle_grid(radius, cell)
    boundary = grid.boundary_cells(mask)
    assert boundary.sum() > 0
    assert (boundary & ~mask).sum() == 0
    sx, sy = grid.boundary_vertices(mask)
    assert sx.size == boundary.sum()
    # Every source point lies within a cell of the true circle.
    assert np.abs(np.hypot(sx, sy) - radius).max() < 2 * cell


def test_rasterize_all_touched_dilates_the_mask():
    grid = GridSpec.from_bounds((-1200.0, -1200.0, 1200.0, 1200.0), cell_size=10.0,
                                crs=UTM10N)
    disc = Point(0.0, 0.0).buffer(600.0)
    centres = grid.rasterize(disc)
    touched = grid.rasterize(disc, all_touched=True)
    assert touched.sum() > centres.sum()
    assert (centres & ~touched).sum() == 0


def test_mask_to_polygon_area_matches_the_cell_count():
    radius, cell = 600.0, 10.0
    grid, mask = _circle_grid(radius, cell)
    polygon = grid.mask_to_polygon(mask)
    assert polygon.area == pytest.approx(mask.sum() * grid.cell_area, rel=1e-9)
    assert grid.rasterize(polygon).sum() == mask.sum()


def test_mask_to_polygon_keeps_detached_components_separate():
    grid = GridSpec.from_bounds((-1000.0, -1000.0, 3000.0, 3000.0), cell_size=20.0,
                                crs=UTM10N)
    mask = grid.rasterize(Point(0.0, 0.0).buffer(400.0)
                          .union(Point(2000.0, 2000.0).buffer(200.0)))
    polygon = grid.mask_to_polygon(mask)
    assert polygon.geom_type == "MultiPolygon"
    assert len(polygon.geoms) == 2


def test_box_union_fallback_agrees_with_rasterio():
    """The rasterio-free polygonisation path must give the same area."""
    grid = GridSpec.from_bounds((-300.0, -300.0, 300.0, 300.0), cell_size=20.0,
                                crs=UTM10N)
    mask = grid.rasterize(Point(0.0, 0.0).buffer(200.0))
    assert (grid._mask_to_polygon_boxes(mask).area
            == pytest.approx(grid.mask_to_polygon(mask).area, rel=1e-9))
