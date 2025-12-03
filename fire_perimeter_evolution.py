"""
Main script for fire perimeter evolution simulation.

This script evolves one contour (inner) to match another contour (outer) over time 
using linear interpolation of their signed distance functions (SDFs).
The inner contour starts inside the outer and grows to match it without shrinking or overshooting.
The evolution ensures the entire contour transforms synchronously, reaching the outer boundary at the same time T.
The interior is filled with colors corresponding to the speed at the time points are incorporated, 
leaving a trace of speeds experienced.
The speed map's values and color mapping are kept constant across frames by fixing the colorbar range.
"""

import numpy as np
from shapefile_loader import load_phi_outer_from_shapefile, get_shapefile_bounds
from contour_generator import generate_contour
from perimeter_evolution import PerimeterEvolution
from perimter_evolution_v2 import PerimeterEvolutionV2

# Configuration parameters
contour_type = 1  # 1 for random Fourier, 2 for tongues with spots
# Optional: path to shapefile for outer contour, or None to use generate_contour
shapefile_path_outer = '/Users/ryanpurciel/Documents/UNR/LahainaWRFSFire/SHAPE_FILES_LAHAINA/530.shp'  
# Optional: path to shapefile for inner contour, or None to use default circle
shapefile_path_inner = '/Users/ryanpurciel/Documents/UNR/LahainaWRFSFire/SHAPE_FILES_LAHAINA/430.shp'

# Define the grid for computation
# If shapefiles are provided, use their bounds in cartesian coordinates
# Otherwise, use default bounds
if shapefile_path_outer is not None:
    bounds = get_shapefile_bounds(shapefile_path_outer, padding=0.1)
    grid_size = 200
    x = np.linspace(bounds['minx'], bounds['maxx'], grid_size)
    y = np.linspace(bounds['miny'], bounds['maxy'], grid_size)
    X, Y = np.meshgrid(x, y)
    xlim = (bounds['minx'], bounds['maxx'])
    ylim = (bounds['miny'], bounds['maxy'])
elif shapefile_path_inner is not None:
    bounds = get_shapefile_bounds(shapefile_path_inner, padding=0.1)
    grid_size = 200
    x = np.linspace(bounds['minx'], bounds['maxx'], grid_size)
    y = np.linspace(bounds['miny'], bounds['maxy'], grid_size)
    X, Y = np.meshgrid(x, y)
    xlim = (bounds['minx'], bounds['maxx'])
    ylim = (bounds['miny'], bounds['maxy'])
else:
    # Default bounds for generated contours
    grid_size = 200
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x, y)
    xlim = (-2, 2)
    ylim = (-2, 2)

# Load phi_inner and phi_outer from shapefiles if provided
# Each shapefile is loaded independently - you can specify one, both, or neither
phi_outer, phi_inner = load_phi_outer_from_shapefile(
    shapefile_path_outer=shapefile_path_outer,
    shapefile_path_inner=shapefile_path_inner,
    X=X,
    Y=Y
)

# Define the inner contour SDF: default to smaller circle if not loaded from shapefile
# SDF convention: negative inside the contour, positive outside
if phi_inner is None:
    # Create a circle centered at the middle of the grid
    center_x = (xlim[0] + xlim[1]) / 2
    center_y = (ylim[0] + ylim[1]) / 2
    # Use a radius that's 10% of the smaller dimension
    radius = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.15
    phi_inner = np.sqrt((X - center_x)**2 + (Y - center_y)**2) - radius

# Define the outer contour SDF: default to generate_contour if not loaded from shapefile
if phi_outer is None:
    if contour_type in [1, 2]:
        phi_outer = generate_contour(contour_type, X, Y)
    else:
        raise ValueError(f"Invalid configuration: either shapefile_path_outer must be set or contour_type must be 1 or 2")

# Set evolution parameters
T = 1.0  # Total time
num_steps = 20  # Number of animation frames

# Choose evolution behavior: 'v1' (varying front speeds) or 'v2' (uniform front speed)
evolution_mode = 'v2'

# Create perimeter evolution instance
if evolution_mode == 'v2':
    evolution = PerimeterEvolutionV2(phi_inner, phi_outer, X, Y, T=T, num_steps=num_steps)
else:
    evolution = PerimeterEvolution(phi_inner, phi_outer, X, Y, T=T, num_steps=num_steps)

# Create and display the animation
evolution.create_animation(interval=200, xlim=xlim, ylim=ylim)

# Save the animation as a GIF
outfile = 'contour_evolution_v2_330_to_430.gif' if evolution_mode == 'v2' else 'contour_evolution_v1_330_to_430.gif'
evolution.save_animation(outfile, fps=10)
