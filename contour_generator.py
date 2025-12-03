"""
Module for generating random contour shapes.

This module provides functions to generate various types of random contours
using signed distance functions (SDFs) for use in contour evolution simulations.
"""

import numpy as np


def generate_fourier_perturbed_contour(X, Y, num_modes=15, seed=42, center_x=0.2, center_y=0.0, r_base=1.0):
    """
    Generate a randomly perturbed circle using Fourier series.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Meshgrid of X coordinates
    Y : numpy.ndarray
        Meshgrid of Y coordinates
    num_modes : int
        Number of Fourier modes for perturbation
    seed : int
        Random seed for reproducibility
    center_x : float
        X coordinate of contour center
    center_y : float
        Y coordinate of contour center
    r_base : float
        Base radius of the circle
    
    Returns:
    --------
    phi_outer : numpy.ndarray
        Signed distance function array (negative inside, positive outside)
    """
    np.random.seed(seed)
    a = np.random.uniform(-0.3, 0.3, num_modes) / np.arange(1, num_modes + 1)
    b = np.random.uniform(-0.3, 0.3, num_modes) / np.arange(1, num_modes + 1)
    
    dx = X - center_x
    dy = Y - center_y
    theta = np.arctan2(dy, dx)
    perturbation = np.zeros_like(theta)
    
    for k in range(1, num_modes + 1):
        perturbation += a[k - 1] * np.cos(k * theta) + b[k - 1] * np.sin(k * theta)
    
    r = r_base + perturbation
    dist = np.sqrt(dx**2 + dy**2) - r
    
    return dist


def generate_tongues_with_spots_contour(X, Y, num_tongues=3, seed=43, center_x=0.2, center_y=0.0, 
                                        r_base=1.0, gap=0.2, r_spot_base=0.1, spot_modes=5):
    """
    Generate an asymmetrical contour with long tongues and disconnected spot contours at the ends.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Meshgrid of X coordinates
    Y : numpy.ndarray
        Meshgrid of Y coordinates
    num_tongues : int
        Number of tongues extending from the main contour
    seed : int
        Random seed for reproducibility
    center_x : float
        X coordinate of contour center
    center_y : float
        Y coordinate of contour center
    r_base : float
        Base radius of the main contour
    gap : float
        Gap between tongue tip and spot contour
    r_spot_base : float
        Base radius of spot contours
    spot_modes : int
        Number of Fourier modes for spot perturbations
    
    Returns:
    --------
    phi_outer : numpy.ndarray
        Signed distance function array (negative inside, positive outside)
    """
    np.random.seed(seed)
    
    dx = X - center_x
    dy = Y - center_y
    theta = np.arctan2(dy, dx)
    
    # Generate tongue parameters
    theta_is = np.random.uniform(0, 2 * np.pi, num_tongues)
    A_is = np.random.uniform(0.5, 2.0, num_tongues)  # Varying lengths for asymmetry
    sigma = 0.1  # Width of tongues
    
    # Create main contour with tongues
    perturbation = np.zeros_like(theta)
    for i in range(num_tongues):
        dtheta = (theta - theta_is[i] + np.pi) % (2 * np.pi) - np.pi  # Periodic difference
        perturbation += A_is[i] * np.exp(-dtheta**2 / (2 * sigma**2))
    
    r = r_base + perturbation
    phi_main = np.sqrt(dx**2 + dy**2) - r
    
    # Add spot contours at tongue tips
    phi_outer = phi_main.copy()
    for i in range(num_tongues):
        dir_x = np.cos(theta_is[i])
        dir_y = np.sin(theta_is[i])
        tip_r = r_base + A_is[i]
        tip_x = center_x + tip_r * dir_x
        tip_y = center_y + tip_r * dir_y
        
        # Position spot contour
        spot_cx = tip_x + gap * dir_x
        spot_cy = tip_y + gap * dir_y
        sdx = X - spot_cx
        sdy = Y - spot_cy
        stheta = np.arctan2(sdy, sdx)
        
        # Perturb spot for non-exact circle
        sa = np.random.uniform(-0.03, 0.03, spot_modes) / np.arange(1, spot_modes + 1)
        sb = np.random.uniform(-0.03, 0.03, spot_modes) / np.arange(1, spot_modes + 1)
        spert = np.zeros_like(stheta)
        for k in range(1, spot_modes + 1):
            spert += sa[k - 1] * np.cos(k * stheta) + sb[k - 1] * np.sin(k * stheta)
        
        r_spot = r_spot_base + spert
        phi_spot = np.sqrt(sdx**2 + sdy**2) - r_spot
        phi_outer = np.minimum(phi_outer, phi_spot)
    
    return phi_outer


def generate_contour(contour_type, X, Y):
    """
    Generate a contour based on the specified type.
    
    Parameters:
    -----------
    contour_type : int
        Type of contour to generate:
        1 = Fourier perturbed circle
        2 = Tongues with spots
    X : numpy.ndarray
        Meshgrid of X coordinates
    Y : numpy.ndarray
        Meshgrid of Y coordinates
    
    Returns:
    --------
    phi_outer : numpy.ndarray
        Signed distance function array (negative inside, positive outside)
    
    Raises:
    -------
    ValueError
        If contour_type is not 1 or 2
    """
    if contour_type == 1:
        return generate_fourier_perturbed_contour(X, Y)
    elif contour_type == 2:
        return generate_tongues_with_spots_contour(X, Y)
    else:
        raise ValueError(f"Invalid contour_type: {contour_type}. Must be 1 or 2.")

