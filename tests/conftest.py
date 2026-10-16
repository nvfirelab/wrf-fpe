"""
Shared fixtures: synthetic perimeter shapefiles in the schema kmz_to_shapefile
writes, so the loading and rasterising path can be exercised without the real
Mill Fire data.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

# The linear prototype imports pyplot at module scope.
os.environ.setdefault("MPLBACKEND", "Agg")
from shapely.geometry import MultiPolygon, Point, Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Near the Mill Fire, Siskiyou County: UTM zone 10N, meridian convergence
# small but nonzero, Web Mercator distortion about 1.33.
FIRE_LON, FIRE_LAT = -122.44, 41.50
UTM10N = "EPSG:32610"


def wobbly_circle(centre, radius, amplitude=0.0, lobes=5, n=240, phase=0.0):
    """A closed polygon: a circle with an optional sinusoidal perturbation."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius * (1.0 + amplitude * np.sin(lobes * theta + phase))
    return Polygon(np.column_stack([centre[0] + r * np.cos(theta),
                                    centre[1] + r * np.sin(theta)]))


@pytest.fixture(scope="session")
def fire_origin():
    """Projected coordinates of the synthetic fire's ignition point, UTM 10N."""
    from pyproj import CRS, Transformer
    to_utm = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(UTM10N),
                                  always_xy=True)
    return to_utm.transform(FIRE_LON, FIRE_LAT)


@pytest.fixture(scope="session")
def initial_geometry(fire_origin):
    """The earlier observed perimeter: a 500 m irregular blob."""
    return wobbly_circle(fire_origin, 500.0, amplitude=0.12, lobes=5)


@pytest.fixture(scope="session")
def final_geometry(fire_origin, initial_geometry):
    """
    The later observed perimeter: the blob grown downwind, plus a detached spot.

    The spot is there because C5 - components of the final perimeter detached
    from the initial one - has to survive rasterisation and polygonisation
    without special handling.
    """
    x, y = fire_origin
    main = wobbly_circle((x + 350.0, y + 250.0), 1100.0, amplitude=0.10, lobes=7,
                         phase=0.7)
    main = main.union(initial_geometry)
    spot = Point(x + 2200.0, y + 1600.0).buffer(220.0)
    return MultiPolygon([main, spot])


def _write(path, geometry, attributes):
    import geopandas as gpd
    record = dict(attributes)
    record["geometry"] = geometry
    gpd.GeoDataFrame([record], crs="EPSG:4326").to_file(path)
    return path


@pytest.fixture(scope="session")
def perimeter_files(tmp_path_factory, initial_geometry, final_geometry):
    """
    Two perimeter shapefiles in WGS84, carrying the kmz_to_shapefile schema.

    The first supplies the ROS vector as components, the second as speed and
    bearing, so both extraction paths get exercised.

    Returns:
    --------
    paths : dict
        {"initial": path, "final": path}
    """
    from pyproj import CRS, Transformer
    from shapely.ops import transform

    to_wgs = Transformer.from_crs(CRS.from_user_input(UTM10N), CRS.from_epsg(4326),
                                  always_xy=True).transform
    out = tmp_path_factory.mktemp("perimeters")

    initial = _write(
        str(out / "perimeter_20240723T013731Z.shp"),
        transform(to_wgs, initial_geometry),
        {
            "step_idx": 1,
            "time_iso": "2024-07-23T01:37:31+00:00",
            "time_epoch": 1721698651,
            "speed_mps": 0.25,
            "dir_deg": 54.462,
            "vel_u_mps": 0.2034,      # 0.25 m/s toward 054 degrees
            "vel_v_mps": 0.1454,
            "n_vectors": 3,
            "vel_src": "vector_geometry",
        },
    )
    final = _write(
        str(out / "perimeter_20240723T025041Z.shp"),
        transform(to_wgs, final_geometry),
        {
            "step_idx": 2,
            "time_iso": "2024-07-23T02:50:41+00:00",
            "time_epoch": 1721703041,
            "speed_mps": 0.31,
            "dir_deg": 60.0,
            "n_vectors": 2,
            "vel_src": "extended_data",
        },
    )
    return {"initial": initial, "final": final}


@pytest.fixture
def pair(perimeter_files):
    """
    The two synthetic observations, on one 25 m grid, in the grid's CRS.

    Returns:
    --------
    initial, final, grid : Perimeter, Perimeter, GridSpec
    """
    from perimeter import PerimeterSeries

    series = PerimeterSeries.from_shapefiles([perimeter_files["initial"],
                                              perimeter_files["final"]])
    grid = series.common_grid(cell_size=25.0, padding=0.1)
    series = series.on_grid(grid)
    return series[0], series[1], grid


@pytest.fixture
def wavelet():
    """
    A fire ellipse for LB = 3 with a head rate of 0.25 m/s, heading 30 degrees.

    Returns:
    --------
    params : dict
        uhat, a, b, c, plus LB and HB
    """
    LB, head = 3.0, 0.25
    HB = (LB + np.sqrt(LB ** 2 - 1)) / (LB - np.sqrt(LB ** 2 - 1))
    back = head / HB
    a = 0.5 * (head + back)
    c = 0.5 * (head - back)
    theta = np.deg2rad(30.0)
    return {"uhat": np.array([np.cos(theta), np.sin(theta)]),
            "a": a, "b": a / LB, "c": c, "LB": LB, "HB": HB}


# ----------------------------------------------------------------------
# A series whose growth really is elliptical
# ----------------------------------------------------------------------

GROWTH_SECONDS = 3600.0


@pytest.fixture(scope="session")
def elliptical_growth(fire_origin):
    """
    Three perimeters generated by dilating the first one under a known ellipse.

    The intermediate perimeter is the exact level set of the gauge's arrival
    time, so it is the answer a perfect interpolation would return, and it was
    produced by the forward solve alone - the two-sided blend the method uses
    never saw it.

    Returns:
    --------
    fixture : dict
        series, grid, ellipse, ros and the total interval in seconds
    """
    from grid import GridSpec
    from metrics import EllipticalMetric, Source
    from perimeter import Perimeter, PerimeterSeries, ros_from_speed_bearing
    from wavelet import FireEllipse

    ox, oy = fire_origin
    grid = GridSpec.from_bounds((ox - 1200.0, oy - 1200.0, ox + 2200.0, oy + 2200.0),
                                cell_size=20.0, crs=UTM10N)
    ros = ros_from_speed_bearing(0.30, 60.0)
    ellipse = FireEllipse.from_ros(ros, 3.0)
    metric = EllipticalMetric(ellipse)

    start = grid.rasterize(wobbly_circle((ox, oy), 300.0, amplitude=0.10, lobes=5))
    sx, sy = grid.boundary_vertices(start)
    arrival = metric.arrival(Source(sx, sy, mask=start, direction="outward"), grid)

    t0 = datetime(2024, 7, 23, 1, 0, tzinfo=timezone.utc)
    perimeters = []
    for fraction in (0.0, 0.5, 1.0):
        # The gauge reports positive time inside the source too - distance
        # back to its boundary - so the level set alone would drop the
        # interior of the initial blob. Union it back in.
        mask = start | (arrival <= fraction * GROWTH_SECONDS)
        perimeters.append(Perimeter(
            grid.mask_to_polygon(mask), grid.crs,
            time=t0 + timedelta(seconds=fraction * GROWTH_SECONDS),
            ros=ros, lb=3.0))
    return {"series": PerimeterSeries(perimeters), "grid": grid,
            "ellipse": ellipse, "ros": ros, "interval": GROWTH_SECONDS}
