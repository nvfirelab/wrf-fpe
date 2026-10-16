"""
Module representing one observed fire perimeter and a time series of them.

A Perimeter bundles the three things the evolution needs about an observation:
the polygon, the time it was flown, and the bulk rate of spread vector that
accompanies it where one exists. Shapefiles written by kmz_to_shapefile.py
carry all three, so from_shapefile() reads that schema directly rather than
re-deriving anything.

The ROS vector is stored as eastward/northward components in metres per second
against *true* north. Projecting a perimeter rotates the vector by the
projection's meridian convergence, so that the bearing stays physically correct
on the grid the gauge solver works in.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import shapely
from pyproj import CRS, Transformer
from shapely.ops import transform, unary_union

from grid import GridSpec, meridian_convergence

# kmz_to_shapefile.py attribute names, in the order they are consulted.
TIME_FIELDS = ("time_iso", "time_begin", "time_end")
EPOCH_FIELDS = ("time_epoch",)
VECTOR_COMPONENT_FIELDS = ("vel_u_mps", "vel_v_mps")
SPEED_FIELDS = ("speed_mps",)
DIRECTION_FIELDS = ("dir_deg",)


def _parse_time(value):
    """Parse an ISO 8601 string, epoch seconds, or datetime into UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return None
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _finite(value):
    """Return a float if the value is a finite number, else None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def ros_from_speed_bearing(speed_mps, dir_deg):
    """
    Convert a speed and a direction of travel into eastward/northward components.

    Parameters:
    -----------
    speed_mps : float
        Spread speed in metres per second
    dir_deg : float
        Direction of travel in degrees clockwise from true north

    Returns:
    --------
    ros : numpy.ndarray
        (u, v) in metres per second, eastward and northward
    """
    theta = np.deg2rad(float(dir_deg))
    return np.array([speed_mps * np.sin(theta), speed_mps * np.cos(theta)], float)


def bearing_of(ros):
    """Direction of a ROS vector in degrees clockwise from north."""
    u, v = np.asarray(ros, float)
    return float(np.degrees(np.arctan2(u, v)) % 360.0)


@dataclass
class Perimeter:
    """
    One observed fire perimeter, with its time and bulk ROS vector.

    Attributes:
    -----------
    geometry : shapely geometry
        (Multi)Polygon of the burned area, in `crs`
    crs : pyproj.CRS
        CRS the geometry is expressed in
    time : datetime or None
        Observation time, timezone-aware UTC
    ros : numpy.ndarray or None
        Bulk rate of spread as (u, v) in metres per second against true north
    lb : float or None
        Length-to-breadth ratio of the fire ellipse, if known independently
    attrs : dict
        Remaining attributes from the source shapefile
    source : str or None
        Path the perimeter was read from
    """

    geometry: object
    crs: CRS
    time: datetime = None
    ros: np.ndarray = None
    lb: float = None
    attrs: dict = field(default_factory=dict, repr=False)
    source: str = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.crs = CRS.from_user_input(self.crs)
        if self.ros is not None:
            self.ros = np.asarray(self.ros, dtype=float).reshape(2)
        if not self.geometry.is_valid:
            self.geometry = self.geometry.buffer(0)
        if self.geometry.is_empty:
            raise ValueError("perimeter geometry is empty")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_shapefile(cls, path, time=None, ros=None, lb=None):
        """
        Read a perimeter from a shapefile written by kmz_to_shapefile.py.

        Every feature in the file is dissolved into one geometry, and the
        attributes are taken from the feature with the latest time - which
        makes additive (cumulative) files from kmz_to_shapefile read the same
        as individual ones. Time comes from time_iso, time_begin or
        time_epoch; the ROS vector from vel_u_mps/vel_v_mps, falling back to
        speed_mps with dir_deg. Explicit arguments override whatever the file
        says.

        Parameters:
        -----------
        path : str
            Path to the .shp file
        time : datetime, optional
            Observation time, overriding the file's
        ros : array-like, optional
            (u, v) in m/s against true north, overriding the file's
        lb : float, optional
            Length-to-breadth ratio

        Returns:
        --------
        perimeter : Perimeter
        """
        import geopandas as gpd

        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"{path} holds no features")
        if gdf.crs is None:
            raise ValueError(
                f"{path} has no CRS; set one before loading so the perimeter "
                "can be projected onto the grid."
            )

        geometry = gdf.geometry.union_all() if len(gdf) > 1 else gdf.geometry.iloc[0]
        record = cls._latest_record(gdf)

        if time is None:
            time = cls._time_from_record(record)
        if ros is None:
            ros = cls._ros_from_record(record)
        if lb is None:
            lb = _finite(record.get("lb"))

        return cls(geometry=geometry, crs=gdf.crs, time=time, ros=ros, lb=lb,
                   attrs=record, source=str(path))

    @classmethod
    def _latest_record(cls, gdf):
        """
        The attribute record that describes the file's *latest* feature.

        An individual-mode file has one feature, so this is trivial. An
        additive (cumulative) file from kmz_to_shapefile holds every earlier
        perimeter as well, and the dissolved geometry is the footprint at the
        time of the newest one - so its time and ROS vector are the ones that
        describe the file. Taking the first row instead would silently attach
        the earliest observation's attributes to the latest footprint.
        """
        records = [{k: v for k, v in row.items() if k != gdf.geometry.name}
                   for _, row in gdf.iterrows()]
        if len(records) == 1:
            return records[0]
        timed = [(cls._time_from_record(r), i) for i, r in enumerate(records)]
        timed = [(t, i) for t, i in timed if t is not None]
        if not timed:
            return records[-1]
        latest = max(timed, key=lambda item: (item[0], item[1]))[1]
        return records[latest]

    @staticmethod
    def _time_from_record(record):
        """Pull an observation time out of a shapefile attribute record."""
        for name in TIME_FIELDS:
            parsed = _parse_time(record.get(name))
            if parsed is not None:
                return parsed
        for name in EPOCH_FIELDS:
            parsed = _parse_time(record.get(name))
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _ros_from_record(record):
        """
        Pull a ROS vector out of a shapefile attribute record.

        Components win over speed-and-bearing: kmz_to_shapefile.py writes both
        when it has them, and the components are what the geodesic derivation
        produced. A zero vector is treated as no vector - it carries no
        heading, and the ellipse is undefined for it.
        """
        u = _finite(record.get(VECTOR_COMPONENT_FIELDS[0]))
        v = _finite(record.get(VECTOR_COMPONENT_FIELDS[1]))
        if u is not None and v is not None and (u or v):
            return np.array([u, v], float)

        speed = next((_finite(record.get(n)) for n in SPEED_FIELDS
                      if _finite(record.get(n))), None)
        direction = next((_finite(record.get(n)) for n in DIRECTION_FIELDS
                          if _finite(record.get(n)) is not None), None)
        if speed and direction is not None:
            return ros_from_speed_bearing(speed, direction)
        return None

    # ------------------------------------------------------------------
    # The observation itself
    # ------------------------------------------------------------------

    @property
    def has_ros(self):
        """Whether this perimeter carries a usable ROS vector."""
        return self.ros is not None and float(np.linalg.norm(self.ros)) > 0.0

    @property
    def ros_speed(self):
        """Magnitude of the ROS vector in m/s, or None."""
        return float(np.linalg.norm(self.ros)) if self.has_ros else None

    @property
    def ros_bearing(self):
        """Direction of the ROS vector in degrees clockwise from north, or None."""
        return bearing_of(self.ros) if self.has_ros else None

    @property
    def area_m2(self):
        """Area of the perimeter in square metres, measured in its own CRS."""
        if self.crs.is_geographic:
            raise ValueError("project the perimeter before measuring its area")
        return float(self.geometry.area)

    def interval_to(self, other):
        """
        Seconds from this perimeter's observation to another's.

        Parameters:
        -----------
        other : Perimeter
            The later observation

        Returns:
        --------
        seconds : float
            Positive elapsed time between the two observations
        """
        if self.time is None or other.time is None:
            raise ValueError("both perimeters need a time to measure an interval")
        seconds = (other.time - self.time).total_seconds()
        if seconds <= 0:
            raise ValueError(
                f"non-increasing interval of {seconds:g} s between "
                f"{self.time.isoformat()} and {other.time.isoformat()}"
            )
        return seconds

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def to_crs(self, crs):
        """
        Reproject the perimeter, rotating the ROS vector to match.

        The vector's components are given against true north; on a projected
        grid the axes are rotated by the meridian convergence at the fire, so
        the components are rotated by the same angle. Skipping this leaves a
        bearing error of up to about 3 degrees near a UTM zone edge.

        Parameters:
        -----------
        crs : pyproj.CRS
            Target CRS

        Returns:
        --------
        perimeter : Perimeter
            A new Perimeter in the target CRS
        """
        crs = CRS.from_user_input(crs)
        if self.crs.equals(crs):
            return self

        project = Transformer.from_crs(self.crs, crs, always_xy=True).transform
        geometry = transform(project, self.geometry)

        ros = self.ros
        if self.has_ros and not crs.is_geographic:
            lon, lat = self.lonlat()
            gamma = np.deg2rad(meridian_convergence(crs, lon, lat))
            # Grid north lies gamma east of true north, so components against
            # true north rotate by -gamma into grid axes.
            cos_g, sin_g = np.cos(gamma), np.sin(gamma)
            u, v = self.ros
            ros = np.array([u * cos_g - v * sin_g, u * sin_g + v * cos_g], float)

        return Perimeter(geometry=geometry, crs=crs, time=self.time, ros=ros,
                         lb=self.lb, attrs=dict(self.attrs), source=self.source)

    def lonlat(self):
        """Centroid of the perimeter as (longitude, latitude) in degrees."""
        centroid = self.geometry.centroid
        if self.crs.is_geographic:
            return float(centroid.x), float(centroid.y)
        to_wgs = Transformer.from_crs(self.crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = to_wgs.transform(centroid.x, centroid.y)
        return float(lon), float(lat)

    # ------------------------------------------------------------------
    # Fields on a grid
    # ------------------------------------------------------------------

    def _on_grid(self, grid):
        """This perimeter's geometry in the grid's CRS."""
        return self.geometry if self.crs.equals(grid.crs) else self.to_crs(grid.crs).geometry

    def mask(self, grid, all_touched=False):
        """
        Rasterise the perimeter onto a grid.

        Parameters:
        -----------
        grid : GridSpec
            The shared computational grid
        all_touched : bool
            Burn every cell the polygon touches rather than only those whose
            centre it contains

        Returns:
        --------
        mask : numpy.ndarray
            Boolean array of shape grid.shape, True inside the perimeter
        """
        cache_key = ("mask", grid.key(), all_touched)
        if cache_key not in self._cache:
            self._cache[cache_key] = grid.rasterize(self._on_grid(grid),
                                                    all_touched=all_touched)
        return self._cache[cache_key]

    def sdf(self, grid, all_touched=False, subcell=False):
        """
        Signed distance to the perimeter in metres: negative inside, positive outside.

        Parameters:
        -----------
        grid : GridSpec
            The shared computational grid
        all_touched : bool
            Passed through to the rasterisation
        subcell : bool
            Remove the half-cell inflation either side of the boundary

        Returns:
        --------
        phi : numpy.ndarray
            Signed distance field of shape grid.shape
        """
        cache_key = ("sdf", grid.key(), all_touched, subcell)
        if cache_key not in self._cache:
            self._cache[cache_key] = grid.sdf(self.mask(grid, all_touched=all_touched),
                                              subcell=subcell)
        return self._cache[cache_key]

    def boundary_vertices(self, grid, source="geometry", spacing=None):
        """
        Source points for a gauge solve.

        The gauge minimises over the source boundary, and its cost is
        cells x vertices, so where the vertices come from is both an accuracy
        and a runtime decision. "geometry" samples the polygon's own rings at
        a fixed spacing, which decouples the vertex count from the grid
        resolution; "mask" takes the rasterised boundary cells, which is what
        the method write-up specifies and grows with the grid.

        Parameters:
        -----------
        grid : GridSpec
            The shared computational grid, whose CRS the vertices come back in
        source : str
            "geometry" or "mask"
        spacing : float, optional
            Maximum distance between sampled vertices in metres, for
            source="geometry". Defaults to the grid's cell size.

        Returns:
        --------
        sx, sy : numpy.ndarray
            1D arrays of source coordinates in metres
        """
        if source == "mask":
            return grid.boundary_vertices(self.mask(grid))
        if source != "geometry":
            raise ValueError(f"unknown vertex source {source!r}")

        spacing = float(spacing if spacing is not None else grid.cell_size)
        cache_key = ("vertices", grid.key(), spacing)
        if cache_key not in self._cache:
            boundary = shapely.segmentize(self._on_grid(grid).boundary, spacing)
            coords = shapely.get_coordinates(boundary)
            coords = np.unique(coords, axis=0)
            self._cache[cache_key] = (coords[:, 0].copy(), coords[:, 1].copy())
        return self._cache[cache_key]

    def to_geodataframe(self, attributes=None):
        """
        Wrap the perimeter in a one-row GeoDataFrame ready to write out.

        Parameters:
        -----------
        attributes : dict, optional
            Extra attributes to write alongside the standard ones

        Returns:
        --------
        gdf : geopandas.GeoDataFrame
        """
        import geopandas as gpd

        record = {
            "time_iso": self.time.isoformat() if self.time else None,
            "speed_mps": self.ros_speed,
            "dir_deg": self.ros_bearing,
            "vel_u_mps": float(self.ros[0]) if self.has_ros else None,
            "vel_v_mps": float(self.ros[1]) if self.has_ros else None,
            "lb": self.lb,
        }
        record.update(attributes or {})
        record["geometry"] = self.geometry
        return gpd.GeoDataFrame([record], crs=self.crs)

    def __repr__(self):
        when = self.time.isoformat() if self.time else "no time"
        vector = (f"{self.ros_speed:.3f} m/s @ {self.ros_bearing:.0f} deg"
                  if self.has_ros else "no ROS")
        return f"Perimeter({when}, {vector}, {self.crs.name})"


class PerimeterSeries:
    """
    A time-ordered series of observed perimeters of one fire.

    Holds the shared grid decision - every pair in a series must be evolved on
    the same grid for the segments to chain - and the consistency checks that
    have to pass before chaining means anything.
    """

    def __init__(self, perimeters):
        """
        Parameters:
        -----------
        perimeters : iterable of Perimeter
            Observations of one fire. Sorted by time; perimeters without a
            time keep their input order after the timed ones.
        """
        perimeters = list(perimeters)
        if not perimeters:
            raise ValueError("a series needs at least one perimeter")
        untimed = [p for p in perimeters if p.time is None]
        timed = sorted((p for p in perimeters if p.time is not None),
                       key=lambda p: p.time)
        self.perimeters = timed + untimed

    @classmethod
    def from_shapefiles(cls, paths):
        """
        Read a series from a list of shapefile paths or a glob pattern.

        Parameters:
        -----------
        paths : str or iterable of str
            Glob pattern, directory, or explicit list of .shp paths

        Returns:
        --------
        series : PerimeterSeries
        """
        import glob
        import os

        if isinstance(paths, (str, os.PathLike)):
            text = str(paths)
            if os.path.isdir(text):
                found = sorted(glob.glob(os.path.join(text, "*.shp")))
            else:
                found = sorted(glob.glob(text))
        else:
            found = [str(p) for p in paths]
        if not found:
            raise ValueError(f"no shapefiles matched {paths!r}")
        # Companion vector files are line geometries, not perimeters.
        found = [p for p in found if not p.endswith("_vectors.shp")]
        return cls([Perimeter.from_shapefile(p) for p in found])

    def __len__(self):
        return len(self.perimeters)

    def __getitem__(self, index):
        return self.perimeters[index]

    def __iter__(self):
        return iter(self.perimeters)

    def pairs(self):
        """
        Consecutive (earlier, later) pairs, one per evolution segment.

        Returns:
        --------
        pairs : list of tuple
            Consecutive Perimeter pairs in time order
        """
        return list(zip(self.perimeters[:-1], self.perimeters[1:]))

    def common_grid(self, cell_size, padding=0.1, crs=None):
        """
        Build the one grid every segment of this series runs on.

        Parameters:
        -----------
        cell_size : float
            Square cell size in metres
        padding : float
            Fraction of the larger bound dimension added on every side
        crs : pyproj.CRS, optional
            Grid CRS. Defaults to the UTM zone containing the fire.

        Returns:
        --------
        grid : GridSpec
        """
        base_crs = self.perimeters[0].crs
        geometries = [p.geometry if p.crs.equals(base_crs)
                      else p.to_crs(base_crs).geometry for p in self.perimeters]
        return GridSpec.from_geometries(geometries, base_crs, cell_size,
                                        padding=padding, target_crs=crs)

    def on_grid(self, grid):
        """Return the series reprojected into a grid's CRS."""
        return PerimeterSeries([p.to_crs(grid.crs) for p in self.perimeters])

    def check_nesting(self, grid=None):
        """
        Report whether each observation contains the one before it.

        Growth that is not nested means the observations disagree with each
        other - a perimeter that receded, or a mapping inconsistency. The
        method treats both observations as hard constraints, so smoothing over
        a mismatch would hide a data problem rather than solve one.

        Parameters:
        -----------
        grid : GridSpec, optional
            When given, the comparison is made on rasterised masks, which is
            what the solver will actually see. Otherwise it is exact geometry.

        Returns:
        --------
        report : list of dict
            One entry per consecutive pair, with the fraction of the earlier
            perimeter's area lost in the later one and whether it is nested
        """
        report = []
        for earlier, later in self.pairs():
            if grid is None:
                a, b = earlier.geometry, later.geometry
                if not earlier.crs.equals(later.crs):
                    b = later.to_crs(earlier.crs).geometry
                lost = a.difference(b).area
                area = a.area
            else:
                ma, mb = earlier.mask(grid), later.mask(grid)
                lost = float((ma & ~mb).sum()) * grid.cell_area
                area = float(ma.sum()) * grid.cell_area
            fraction = lost / area if area else 0.0
            report.append({
                "earlier": earlier.time,
                "later": later.time,
                "area_lost_m2": float(lost),
                "fraction_lost": float(fraction),
                "nested": fraction <= 1e-9,
            })
        return report

    def __repr__(self):
        span = ""
        timed = [p.time for p in self.perimeters if p.time]
        if len(timed) > 1:
            span = f", {timed[0].isoformat()} to {timed[-1].isoformat()}"
        return f"PerimeterSeries({len(self.perimeters)} perimeters{span})"
