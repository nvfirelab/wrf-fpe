"""
Module for loading signed distance functions from shapefiles.

This module provides functionality to read shapefiles and convert them
into signed distance function (SDF) arrays for use in contour evolution.
Supports automatic conversion to cartesian coordinates (UTM) for georeferenced shapefiles.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from pyproj import CRS


def _get_cartesian_crs(shapefile_path):
    """
    Determine the appropriate cartesian CRS (UTM) for a shapefile based on its location.
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile
    
    Returns:
    --------
    crs : pyproj.CRS
        The cartesian CRS (UTM zone) appropriate for the shapefile's location
    """
    gdf = gpd.read_file(shapefile_path)
    
    # Get the CRS of the shapefile
    source_crs = gdf.crs
    
    # If already in a projected CRS, check if it's cartesian-like
    if source_crs is not None:
        if source_crs.is_projected:
            # Check if it's already a cartesian CRS (UTM, State Plane, etc.)
            crs_name = str(source_crs).lower()
            if 'utm' in crs_name or 'state plane' in crs_name or 'epsg:326' in crs_name or 'epsg:327' in crs_name:
                return source_crs
            # If it's another projected CRS, we'll transform to UTM
        # Get centroid for UTM zone determination
        if source_crs.is_geographic:
            gdf_wgs84 = gdf
        else:
            gdf_wgs84 = gdf.to_crs('EPSG:4326')
    else:
        # Assume WGS84 if no CRS - check if coordinates look like lat/lon
        bounds = gdf.total_bounds
        if bounds[0] > -180 and bounds[0] < 180 and bounds[1] > -90 and bounds[1] < 90:
            # Looks like lat/lon, treat as WGS84
            gdf_wgs84 = gdf
        else:
            # Assume it's already in a projected CRS, try to determine UTM from bounds
            # For now, assume it needs conversion - will need manual CRS specification
            gdf_wgs84 = gdf
    
    # Get centroid
    centroid = gdf_wgs84.geometry.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    
    # Determine UTM zone (1-60 for northern hemisphere, add 32700 for southern)
    utm_zone = int((lon + 180) / 6) + 1
    # Determine hemisphere
    if lat >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
    
    return CRS.from_epsg(epsg_code)


def _load_phi_from_shapefile(shapefile_path, X, Y, target_crs=None):
    """
    Internal helper function to load a signed distance function from a shapefile.
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile (.shp file)
    X : numpy.ndarray
        Meshgrid of X coordinates (in cartesian CRS)
    Y : numpy.ndarray
        Meshgrid of Y coordinates (in cartesian CRS)
    target_crs : pyproj.CRS, optional
        Target cartesian CRS. If None, will be determined from shapefile location.
    
    Returns:
    --------
    phi : numpy.ndarray
        Signed distance function array (negative inside, positive outside)
        Same shape as X and Y
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Determine target CRS if not provided
    if target_crs is None:
        target_crs = _get_cartesian_crs(shapefile_path)
    
    # Transform to cartesian CRS if needed
    if gdf.crs is None:
        # If no CRS, assume it's already in the target CRS or we need to set it
        # Try to infer from bounds
        pass
    elif str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    
    # Union all geometries in case there are multiple features
    if len(gdf) > 1:
        geometry = unary_union(gdf.geometry.values)
    else:
        geometry = gdf.geometry.iloc[0]
    
    # Initialize phi array
    phi = np.zeros_like(X)
    
    # Compute SDF for each point in the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = Point(X[i, j], Y[i, j])
            
            # Compute distance to the geometry boundary
            distance_to_boundary = geometry.boundary.distance(point)
            
            # Check if point is inside or outside
            # touches() checks if point is on boundary
            if geometry.touches(point):
                # On boundary: SDF = 0
                phi[i, j] = 0.0
            elif geometry.contains(point):
                # Inside: negative distance (SDF convention)
                phi[i, j] = -distance_to_boundary
            else:
                # Outside: positive distance
                phi[i, j] = distance_to_boundary
    
    return phi


def get_shapefile_bounds(shapefile_path, padding=0.1):
    """
    Get the bounds of a shapefile in cartesian coordinates.
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile
    padding : float
        Padding factor (0.1 = 10% padding) to add around the bounds
    
    Returns:
    --------
    bounds : dict
        Dictionary with keys: 'minx', 'miny', 'maxx', 'maxy', 'crs'
    """
    gdf = gpd.read_file(shapefile_path)
    
    # Determine target CRS
    target_crs = _get_cartesian_crs(shapefile_path)
    
    # Transform to cartesian CRS if needed
    if gdf.crs is None:
        # If no CRS, try to set it based on typical geodetic coordinates
        # Check if coordinates look like lat/lon
        bounds = gdf.total_bounds
        if bounds[0] > -180 and bounds[0] < 180 and bounds[1] > -90 and bounds[1] < 90:
            gdf.set_crs('EPSG:4326', inplace=True)
    
    if gdf.crs is not None:
        # Compare CRS objects properly
        if str(gdf.crs) != str(target_crs):
            gdf = gdf.to_crs(target_crs)
    
    # Get bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Add padding
    width = maxx - minx
    height = maxy - miny
    padding_x = width * padding
    padding_y = height * padding
    
    return {
        'minx': minx - padding_x,
        'miny': miny - padding_y,
        'maxx': maxx + padding_x,
        'maxy': maxy + padding_y,
        'crs': target_crs
    }


def load_phi_outer_from_shapefile(shapefile_path_outer=None, shapefile_path_inner=None, X=None, Y=None, target_crs=None):
    """
    Load phi_outer and/or phi_inner from shapefiles and compute signed distance functions.
    Automatically converts georeferenced shapefiles to cartesian coordinates (UTM).
    
    Parameters:
    -----------
    shapefile_path_outer : str, optional
        Path to the shapefile for outer contour (.shp file)
        If None, returns None for phi_outer
    shapefile_path_inner : str, optional
        Path to the shapefile for inner contour (.shp file)
        If None, returns None for phi_inner
    X : numpy.ndarray, optional
        Meshgrid of X coordinates in cartesian CRS (required if shapefile_path_outer or shapefile_path_inner is provided)
    Y : numpy.ndarray, optional
        Meshgrid of Y coordinates in cartesian CRS (required if shapefile_path_outer or shapefile_path_inner is provided)
    target_crs : pyproj.CRS, optional
        Target cartesian CRS. If None, will be determined from shapefile location.
    
    Returns:
    --------
    phi_outer : numpy.ndarray or None
        Signed distance function array for outer contour (negative inside, positive outside)
        Same shape as X and Y, or None if shapefile_path_outer is None
    phi_inner : numpy.ndarray or None
        Signed distance function array for inner contour (negative inside, positive outside)
        Same shape as X and Y, or None if shapefile_path_inner is None
    """
    phi_outer = None
    phi_inner = None
    
    # Determine target CRS from shapefiles if not provided
    if target_crs is None:
        if shapefile_path_outer is not None:
            target_crs = _get_cartesian_crs(shapefile_path_outer)
        elif shapefile_path_inner is not None:
            target_crs = _get_cartesian_crs(shapefile_path_inner)
    
    if shapefile_path_outer is not None:
        if X is None or Y is None:
            raise ValueError("X and Y must be provided when loading shapefiles")
        phi_outer = _load_phi_from_shapefile(shapefile_path_outer, X, Y, target_crs=target_crs)
    
    if shapefile_path_inner is not None:
        if X is None or Y is None:
            raise ValueError("X and Y must be provided when loading shapefiles")
        phi_inner = _load_phi_from_shapefile(shapefile_path_inner, X, Y, target_crs=target_crs)
    
    return phi_outer, phi_inner

