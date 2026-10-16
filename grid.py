"""
Module defining the computational grid shared by every stage of the evolution.

One GridSpec - CRS, origin, cell size, shape - is built once and passed to
everything downstream. The projection carries physical meaning here: the rate
of spread vector is in metres per second and it is the only physical input, so
the grid must be locally equidistant. Web Mercator is conformal but not
equidistant (scale factor ~1.33 at the latitude of northern California), which
would inflate every distance by a third and corrupt the one quantity the method
depends on. Use a local UTM zone or an equidistant azimuthal projection.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Proj
from scipy.ndimage import binary_erosion, distance_transform_edt
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union
import shapely

# Fractional scale distortion above which the projection is not equidistant
# enough to measure a rate of spread on. UTM sits at ~4e-4; Web Mercator at the
# Mill Fire's latitude sits at 0.335.
MAX_SCALE_DISTORTION = 0.01


def utm_crs_for(lon, lat):
    """
    Return the UTM CRS covering a longitude/latitude in degrees.

    Parameters:
    -----------
    lon : float
        Longitude in degrees east
    lat : float
        Latitude in degrees north

    Returns:
    --------
    crs : pyproj.CRS
        UTM zone CRS, northern or southern hemisphere as appropriate
    """
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def scale_distortion(crs, lon, lat):
    """
    Fractional departure from true scale of a projection at a point.

    Parameters:
    -----------
    crs : pyproj.CRS
        Projected CRS to test
    lon, lat : float
        Longitude and latitude in degrees where the projection is evaluated

    Returns:
    --------
    distortion : float
        max(|meridional scale - 1|, |parallel scale - 1|). Zero for a locally
        equidistant projection, 0.335 for Web Mercator at 41.5 N.
    """
    factors = Proj(crs).get_factors(lon, lat)
    return max(abs(factors.meridional_scale - 1.0),
               abs(factors.parallel_scale - 1.0))


def meridian_convergence(crs, lon, lat):
    """
    Angle in degrees between grid north and true north at a point.

    A ROS vector given as eastward/northward components is expressed against
    true north; on a UTM grid those axes are rotated by this angle. The
    rotation reaches about 3 degrees at a zone edge, which matters for a
    method whose whole physical input is a bearing.
    """
    return Proj(crs).get_factors(lon, lat).meridian_convergence


@dataclass(frozen=True)
class GridSpec:
    """
    A regular grid of square cells in a projected, metres-based CRS.

    Coordinates name cell centres: cell (i, j) sits at
    (minx + j*cell_size, miny + i*cell_size). Cells are square by
    construction, which the gauge solver relies on - it treats displacements
    as metres in both axes, so a grid built from a fixed cell *count* over
    non-square bounds would silently corrupt every travel time.
    """

    crs: CRS
    minx: float
    miny: float
    cell_size: float
    nx: int
    ny: int

    def __post_init__(self):
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if self.nx < 2 or self.ny < 2:
            raise ValueError("grid must be at least 2x2 cells")
        crs = CRS.from_user_input(self.crs)
        object.__setattr__(self, "crs", crs)
        if crs.is_geographic:
            raise ValueError(
                "GridSpec needs a projected CRS in metres; got a geographic "
                f"CRS ({crs.name}). Distances on it are degrees, not metres."
            )
        units = {axis.unit_name for axis in crs.axis_info}
        if not units <= {"metre", "meter", "m"}:
            raise ValueError(f"GridSpec needs a CRS in metres; {crs.name} uses {units}")
        self._warn_if_not_equidistant()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_bounds(cls, bounds, cell_size, crs, padding=0.0):
        """
        Build a grid covering projected bounds.

        Parameters:
        -----------
        bounds : tuple
            (minx, miny, maxx, maxy) in the CRS's units
        cell_size : float
            Square cell size in metres
        crs : pyproj.CRS or anything CRS.from_user_input accepts
            Projected, metres-based CRS
        padding : float
            Fraction of the larger bound dimension to add on every side

        Returns:
        --------
        grid : GridSpec
        """
        minx, miny, maxx, maxy = map(float, bounds)
        if maxx <= minx or maxy <= miny:
            raise ValueError(f"degenerate bounds {bounds}")
        if padding:
            pad = padding * max(maxx - minx, maxy - miny)
            minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
        # Snap the origin down to a multiple of the cell size so that grids
        # built from different perimeters of the same fire stay aligned.
        minx = np.floor(minx / cell_size) * cell_size
        miny = np.floor(miny / cell_size) * cell_size
        nx = int(np.ceil((maxx - minx) / cell_size)) + 1
        ny = int(np.ceil((maxy - miny) / cell_size)) + 1
        return cls(crs=crs, minx=minx, miny=miny, cell_size=cell_size, nx=nx, ny=ny)

    @classmethod
    def from_geometries(cls, geometries, crs, cell_size, padding=0.1, target_crs=None):
        """
        Build one grid covering every geometry, projecting them if needed.

        Parameters:
        -----------
        geometries : iterable of shapely geometries
            Geometries in `crs`
        crs : pyproj.CRS
            CRS the geometries are given in
        cell_size : float
            Square cell size in metres
        padding : float
            Fraction of the larger bound dimension added on every side
        target_crs : pyproj.CRS, optional
            Grid CRS. Defaults to the UTM zone of the geometries' centroid.

        Returns:
        --------
        grid : GridSpec
        """
        crs = CRS.from_user_input(crs)
        merged = unary_union(list(geometries))
        if merged.is_empty:
            raise ValueError("no geometry to build a grid from")

        if target_crs is None:
            centroid = merged.centroid
            if crs.is_geographic:
                lon, lat = centroid.x, centroid.y
            else:
                from pyproj import Transformer
                to_wgs = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
                lon, lat = to_wgs.transform(centroid.x, centroid.y)
            target_crs = utm_crs_for(lon, lat)
        target_crs = CRS.from_user_input(target_crs)

        if not crs.equals(target_crs):
            from pyproj import Transformer
            project = Transformer.from_crs(crs, target_crs, always_xy=True).transform
            merged = transform(project, merged)

        return cls.from_bounds(merged.bounds, cell_size, target_crs, padding=padding)

    # ------------------------------------------------------------------
    # Geometry of the grid itself
    # ------------------------------------------------------------------

    @property
    def maxx(self):
        """X coordinate of the last cell centre."""
        return self.minx + (self.nx - 1) * self.cell_size

    @property
    def maxy(self):
        """Y coordinate of the last cell centre."""
        return self.miny + (self.ny - 1) * self.cell_size

    @property
    def shape(self):
        """(ny, nx), the shape of every field defined on this grid."""
        return (self.ny, self.nx)

    @property
    def size(self):
        """Number of cells."""
        return self.nx * self.ny

    @property
    def bounds(self):
        """(minx, miny, maxx, maxy) of the cell centres."""
        return (self.minx, self.miny, self.maxx, self.maxy)

    @property
    def extent(self):
        """(minx, maxx, miny, maxy) of the outer cell edges, for imshow."""
        half = 0.5 * self.cell_size
        return (self.minx - half, self.maxx + half, self.miny - half, self.maxy + half)

    @property
    def cell_area(self):
        """Area of one cell in square metres."""
        return self.cell_size ** 2

    @property
    def x_coords(self):
        """1D array of cell-centre X coordinates."""
        return self.minx + np.arange(self.nx) * self.cell_size

    @property
    def y_coords(self):
        """1D array of cell-centre Y coordinates."""
        return self.miny + np.arange(self.ny) * self.cell_size

    def meshgrid(self):
        """
        Cell-centre coordinate arrays.

        Returns:
        --------
        X, Y : numpy.ndarray
            Arrays of shape (ny, nx) holding cell-centre coordinates in metres
        """
        return np.meshgrid(self.x_coords, self.y_coords)

    @property
    def transform(self):
        """
        Affine transform mapping (col, row) cell edges to CRS coordinates.

        North-up, as GDAL expects: row 0 of a raster written with this
        transform is the *top* row, so field arrays must be flipped vertically
        before writing (they are stored with row 0 at miny).
        """
        from rasterio.transform import from_origin
        half = 0.5 * self.cell_size
        return from_origin(self.minx - half, self.maxy + half,
                           self.cell_size, self.cell_size)

    def key(self):
        """Hashable identity of this grid, for caching fields against it."""
        return (self.crs.to_string(), self.minx, self.miny,
                self.cell_size, self.nx, self.ny)

    def _warn_if_not_equidistant(self):
        """Warn if the projection's scale factor would corrupt distances."""
        try:
            from pyproj import Transformer
            to_wgs = Transformer.from_crs(self.crs, CRS.from_epsg(4326), always_xy=True)
            lon, lat = to_wgs.transform(
                self.minx + 0.5 * (self.nx - 1) * self.cell_size,
                self.miny + 0.5 * (self.ny - 1) * self.cell_size,
            )
            distortion = scale_distortion(self.crs, lon, lat)
        except Exception:  # projection factors are unavailable for some CRSs
            return
        if distortion > MAX_SCALE_DISTORTION:
            warnings.warn(
                f"{self.crs.name} has a scale distortion of {distortion:.3f} at the "
                f"grid centre: distances on this grid are wrong by "
                f"{distortion * 100:.1f}%, which corrupts the ROS vector the method "
                f"depends on. Use a local UTM zone or an equidistant azimuthal "
                f"projection.",
                stacklevel=3,
            )

    def __repr__(self):
        return (f"GridSpec({self.nx}x{self.ny} cells of {self.cell_size:g} m, "
                f"{(self.maxx - self.minx) / 1000:.1f}x{(self.maxy - self.miny) / 1000:.1f} km, "
                f"{self.crs.name})")

    # ------------------------------------------------------------------
    # Fields on the grid
    # ------------------------------------------------------------------

    def rasterize(self, geometry, all_touched=False):
        """
        Burn a polygon onto the grid.

        A cell is burned when its centre lies inside the polygon, which keeps
        the mask consistent with the cell-centre coordinates the gauge solver
        works in. all_touched=True instead burns every cell the polygon
        touches, which dilates the perimeter by up to one cell.

        Parameters:
        -----------
        geometry : shapely geometry
            (Multi)Polygon in this grid's CRS
        all_touched : bool
            Burn cells the polygon merely intersects as well

        Returns:
        --------
        mask : numpy.ndarray
            Boolean array of shape (ny, nx), True inside the polygon
        """
        if geometry.is_empty:
            return np.zeros(self.shape, dtype=bool)

        X, Y = self.meshgrid()
        shapely.prepare(geometry)
        if all_touched:
            half = 0.5 * self.cell_size
            cells = shapely.box(X - half, Y - half, X + half, Y + half)
            mask = shapely.intersects(geometry, cells)
        else:
            mask = shapely.contains_xy(geometry, X, Y)
        return np.asarray(mask, dtype=bool)

    def sdf(self, mask, subcell=False):
        """
        Signed distance to the mask's boundary, in metres.

        Negative inside, positive outside, following the convention the
        prototypes use.

        Each transform measures distance to the nearest cell of the *opposite*
        class, so the field steps from -cell_size to +cell_size across the
        boundary: the zero crossing lands on the cell edge, which is where the
        rasterised perimeter is, but every magnitude is inflated by about half
        a cell. On a 600 m circle at 10 m cells that is a mean bias of +2.2 m
        and a worst case of one cell. subcell=True removes the inflation by
        pulling each side back by half a cell, which leaves the zero contour
        where it is and halves the worst case. The default is the plain form,
        which is what the method write-up specifies.

        Parameters:
        -----------
        mask : numpy.ndarray
            Boolean array of shape (ny, nx)
        subcell : bool
            Remove the half-cell inflation either side of the boundary

        Returns:
        --------
        phi : numpy.ndarray
            Signed distance in metres, same shape as mask
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} does not match grid {self.shape}")
        if not mask.any():
            return np.full(self.shape, np.inf)
        if mask.all():
            return np.full(self.shape, -np.inf)
        phi = (distance_transform_edt(~mask, sampling=self.cell_size)
               - distance_transform_edt(mask, sampling=self.cell_size))
        if subcell:
            phi -= 0.5 * self.cell_size * np.sign(phi)
        return phi

    def boundary_cells(self, mask):
        """
        The inner boundary layer of a mask: cells inside with a neighbour outside.

        Parameters:
        -----------
        mask : numpy.ndarray
            Boolean array of shape (ny, nx)

        Returns:
        --------
        boundary : numpy.ndarray
            Boolean array, True on the mask's inner edge
        """
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return np.zeros(self.shape, dtype=bool)
        # border_value=1 keeps a mask running off the grid edge from being
        # treated as a boundary there.
        return mask & ~binary_erosion(mask, border_value=1)

    def boundary_vertices(self, mask):
        """
        Coordinates of a mask's boundary cells, as the gauge solver's sources.

        Parameters:
        -----------
        mask : numpy.ndarray
            Boolean array of shape (ny, nx)

        Returns:
        --------
        sx, sy : numpy.ndarray
            1D arrays of boundary cell-centre coordinates in metres
        """
        X, Y = self.meshgrid()
        iy, ix = np.nonzero(self.boundary_cells(mask))
        return X[iy, ix], Y[iy, ix]

    # ------------------------------------------------------------------
    # Back to vector
    # ------------------------------------------------------------------

    def mask_to_polygon(self, mask):
        """
        Polygonise a mask, for writing a rasterised perimeter back to a shapefile.

        Parameters:
        -----------
        mask : numpy.ndarray
            Boolean array of shape (ny, nx)

        Returns:
        --------
        geometry : shapely geometry
            Union of the burned cells, in this grid's CRS. Empty if nothing
            is burned. Cell-square edges, so it is the rasterisation itself
            rather than a smoothed reconstruction of the original polygon.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} does not match grid {self.shape}")
        if not mask.any():
            from shapely.geometry import Polygon
            return Polygon()

        try:
            from rasterio.features import shapes as raster_shapes
        except ImportError:
            return self._mask_to_polygon_boxes(mask)

        # rasterio walks rows top-down; the grid stores row 0 at miny.
        flipped = np.flipud(mask).astype(np.uint8)
        parts = [shape(geom) for geom, value
                 in raster_shapes(flipped, mask=flipped.astype(bool),
                                  transform=self.transform)
                 if value == 1]
        return unary_union(parts)

    def _mask_to_polygon_boxes(self, mask):
        """Polygonise by unioning cell squares, when rasterio is unavailable."""
        half = 0.5 * self.cell_size
        X, Y = self.meshgrid()
        iy, ix = np.nonzero(mask)
        cells = [box(X[i, j] - half, Y[i, j] - half, X[i, j] + half, Y[i, j] + half)
                 for i, j in zip(iy, ix)]
        return unary_union(cells)

    def to_geodataframe(self, mask, attributes=None):
        """
        Wrap a polygonised mask in a GeoDataFrame ready to write to a shapefile.

        Parameters:
        -----------
        mask : numpy.ndarray
            Boolean array of shape (ny, nx)
        attributes : dict, optional
            Attribute values for the single output row

        Returns:
        --------
        gdf : geopandas.GeoDataFrame
            One row holding the polygonised mask, in this grid's CRS
        """
        import geopandas as gpd
        record = dict(attributes or {})
        record["geometry"] = self.mask_to_polygon(mask)
        return gpd.GeoDataFrame([record], crs=self.crs)
