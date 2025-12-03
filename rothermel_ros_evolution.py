"""
Module for fire perimeter evolution using the Rothermel rate-of-spread (ROS) formula.

This module implements a fire spread model based on Rothermel's 1972 rate-of-spread
formula, which computes fire spread rate based on fuel properties, wind, and slope.
The model evolves fire perimeters using level set methods with ROS-computed speeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. YAML config file loading will be disabled.")


class RothermelROSEvolution:
    """
    Class to handle perimeter evolution using Rothermel rate-of-spread formula.
    
    This class evolves an inner contour to match an outer contour over time,
    using fire spread speeds computed from the Rothermel ROS model. The model
    accounts for fuel properties, wind speed, and slope effects.
    """
    
    def __init__(self, phi_inner, phi_outer, X, Y, T=100.0, num_steps=100,
                 fuel_load=None, fuel_depth=None, fuel_moisture=None,
                 wind_speed=None, wind_direction=None, slope=None, slope_aspect=None,
                 config_file=None):
        """
        Initialize the Rothermel ROS perimeter evolution.
        
        Parameters:
        -----------
        phi_inner : numpy.ndarray
            Signed distance function for inner contour (negative inside, positive outside)
        phi_outer : numpy.ndarray
            Signed distance function for outer contour (negative inside, positive outside)
        X : numpy.ndarray
            Meshgrid of X coordinates
        Y : numpy.ndarray
            Meshgrid of Y coordinates
        T : float
            Maximum evolution time (arbitrary units). Default 100.0.
            The fire will evolve naturally and may not reach phi_outer at this time.
        num_steps : int
            Number of animation frames. Default 100.
        fuel_load : numpy.ndarray, optional
            Fuel load (kg/m²) at each grid point. If None, uses uniform default.
        fuel_depth : numpy.ndarray, optional
            Fuel depth (m) at each grid point. If None, uses uniform default.
        fuel_moisture : numpy.ndarray, optional
            Fuel moisture content (fraction) at each grid point. If None, uses uniform default.
        wind_speed : numpy.ndarray, optional
            Wind speed (m/s) at each grid point. If None, uses uniform default.
        wind_direction : numpy.ndarray, optional
            Wind direction (radians, 0 = east, increases counterclockwise) at each grid point.
            If None, uses value from config file (which is in degrees, 0 = north, increases clockwise).
            Note: When passed directly, use radians in east-based convention for backward compatibility.
        slope : numpy.ndarray, optional
            Slope (radians) at each grid point. If None, uses uniform default.
        slope_aspect : numpy.ndarray, optional
            Slope aspect (radians, 0 = east) at each grid point. If None, uses uniform default.
        config_file : str, optional
            Path to YAML configuration file. If None, uses default config file in same directory,
            or falls back to hardcoded defaults if file doesn't exist.
        """
        # Ensure phi_inner is a proper signed distance function
        # The zero level set (phi = 0) represents the fire perimeter
        # Negative values = inside fire, positive values = outside fire
        self.phi_inner = phi_inner.copy()
        self.X = X
        self.Y = Y
        self.T = T
        self.num_steps = num_steps
        
        # Compute grid spacing first (needed for gradient calculations)
        x = X[0, :]
        y = Y[:, 0]
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        
        # Verify and optionally renormalize phi_inner to ensure it's a proper SDF
        # Check if |∇phi| ≈ 1 (property of a signed distance function)
        # Note: We assume phi_inner is already a proper SDF, but we'll renormalize during evolution
        
        # Set phi_outer to be at the edge of the domain (free evolution)
        # Create a signed distance function where the boundary is at the domain edges
        x_min, x_max = X[0, 0], X[0, -1]
        y_min, y_max = Y[0, 0], Y[-1, 0]
        
        # Compute distance to nearest domain edge
        # phi_outer should be negative inside domain, zero at edges, positive outside
        dist_to_left = X - x_min
        dist_to_right = x_max - X
        dist_to_bottom = Y - y_min
        dist_to_top = y_max - Y
        
        # Distance to nearest edge (negative means inside)
        phi_outer = -np.minimum(np.minimum(dist_to_left, dist_to_right),
                               np.minimum(dist_to_bottom, dist_to_top))
        
        self.phi_outer = phi_outer
        
        # Find center of initial contour (origin)
        zero_mask = np.abs(phi_inner) < 0.05 * np.max(np.abs(phi_inner))
        if np.any(zero_mask):
            self.origin_x = np.mean(X[zero_mask])
            self.origin_y = np.mean(Y[zero_mask])
        else:
            # Fallback: use center of grid
            self.origin_x = (X[0, 0] + X[-1, -1]) / 2.0
            self.origin_y = (Y[0, 0] + Y[-1, -1]) / 2.0
        
        # Convert coordinates to feet from origin
        # Assuming input coordinates are in meters, convert to feet
        # 1 meter = 3.28084 feet
        METERS_TO_FEET = 3.28084
        self.X_ft = (X - self.origin_x) * METERS_TO_FEET
        self.Y_ft = (Y - self.origin_y) * METERS_TO_FEET
        
        # Compute grid spacing in feet
        x_ft = self.X_ft[0, :]
        y_ft = self.Y_ft[:, 0]
        self.dx_ft = x_ft[1] - x_ft[0]
        self.dy_ft = y_ft[1] - y_ft[0]
        
        # Load configuration from YAML file
        config = self._load_config(config_file)
        
        # Set default fuel and environmental parameters if not provided
        # Use values from config file, or fall back to hardcoded defaults
        self.fuel_load = fuel_load if fuel_load is not None else np.ones_like(X) * config['fuel_load']
        self.fuel_depth = fuel_depth if fuel_depth is not None else np.ones_like(X) * config['fuel_depth']
        self.fuel_moisture = fuel_moisture if fuel_moisture is not None else np.ones_like(X) * config['fuel_moisture']
        self.wind_speed = wind_speed if wind_speed is not None else np.ones_like(X) * config['wind_speed']
        
        # Convert wind_direction from degrees (north-based) to radians (east-based)
        # Config file uses: 0° = north (wind from north, blowing south), increases clockwise
        # Code expects: 0 radians = east (wind blowing east), increases counterclockwise
        # Conversion mapping:
        #   Config 0° (north, blowing south) -> Math 270° (3π/2 rad, pointing south)
        #   Config 90° (east, blowing west) -> Math 180° (π rad, pointing west)
        #   Config 180° (south, blowing north) -> Math 90° (π/2 rad, pointing north)
        #   Config 270° (west, blowing east) -> Math 0° (0 rad, pointing east)
        # Formula: math_angle = (270 - config_angle) % 360, then convert to radians
        if wind_direction is not None:
            # If passed directly, assume it's already in radians (east-based) for backward compatibility
            self.wind_direction = wind_direction
        else:
            # Convert from config: degrees (north-based) to radians (east-based)
            wind_dir_deg = config['wind_direction']
            # Convert north-based (0° = north, clockwise) to east-based (0° = east, counterclockwise)
            # Formula: (270 - config_angle) % 360 converts correctly
            wind_dir_math_deg = (270.0 - wind_dir_deg) % 360.0
            wind_dir_rad = np.deg2rad(wind_dir_math_deg)
            self.wind_direction = np.ones_like(X) * wind_dir_rad
        
        self.slope = slope if slope is not None else np.ones_like(X) * config['slope']
        self.slope_aspect = slope_aspect if slope_aspect is not None else np.ones_like(X) * config['slope_aspect']
        
        # Rothermel model constants from config
        self.heat_content = config['heat_content']
        self.mineral_damping = config['mineral_damping']
        self.max_reaction_velocity = 0.0  # Will be computed from fuel properties
        
        # Initialize tracking variables
        # phi_prev starts as phi_inner - the zero level set is the initial fire perimeter
        self.phi_prev = self.phi_inner.copy()
        
        # Track which points have reached phi_outer (stop evolving them)
        self.reached_outer = np.zeros_like(phi_inner, dtype=bool)
        
        # Animation variables
        self.fig = None
        self.ax = None
        self.ani = None
    
    def _load_config(self, config_file=None):
        """
        Load configuration from YAML file.
        
        Parameters:
        -----------
        config_file : str, optional
            Path to YAML configuration file. If None, uses default config file.
        
        Returns:
        --------
        config : dict
            Configuration dictionary with parameter values
        """
        # Default configuration values (fallback)
        # Note: wind_direction is in degrees (0 = north, increases clockwise)
        default_config = {
            'fuel_load': 0.5,
            'fuel_depth': 0.3,
            'fuel_moisture': 0.15,
            'wind_speed': 5.0,
            'wind_direction': 0.0,  # degrees: 0 = north, increases clockwise
            'slope': 0.0,
            'slope_aspect': 0.0,
            'heat_content': 18600.0,
            'mineral_damping': 0.174
        }
        
        # Determine config file path
        if config_file is None:
            # Use default config file in same directory as this module
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(module_dir, 'rothermel_config.yaml')
        
        # Try to load from file
        if not HAS_YAML:
            print("Warning: PyYAML not available. Using default configuration values.")
            return default_config
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config is not None:
                        # Merge file config with defaults (file config takes precedence)
                        default_config.update(file_config)
            except (yaml.YAMLError, IOError) as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
                print("Using default configuration values.")
        else:
            print(f"Warning: Config file {config_file} not found. Using default configuration values.")
        
        return default_config
    
    def _compute_rothermel_ros(self, phi, t):
        """
        Compute rate of spread using Rothermel's formula at each grid point.
        
        This is a simplified implementation of the Rothermel model that computes
        ROS based on fuel properties, wind, and slope.
        
        Parameters:
        -----------
        phi : numpy.ndarray
            Current signed distance function
        t : float
            Current time
        
        Returns:
        --------
        ros : numpy.ndarray
            Rate of spread (ft/min) at each grid point
        """
        # Compute gradient to determine fire front direction
        grad_y, grad_x = np.gradient(phi, self.dy, self.dx)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
        
        # Normalized gradient (points outward from fire)
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag
        
        # Compute bulk density
        bulk_density = self.fuel_load / (self.fuel_depth + 1e-10)  # kg/m³
        
        # Simplified reaction intensity (based on fuel load and moisture)
        # Higher fuel load and lower moisture = higher reaction intensity
        moisture_factor = np.exp(-3.59 * self.fuel_moisture)  # Moisture damping
        reaction_intensity = self.heat_content * self.fuel_load * moisture_factor * self.mineral_damping
        
        # Wind factor (simplified)
        # Wind speed in m/s, convert to m/min for Rothermel units
        wind_speed_m_per_min = self.wind_speed * 60.0
        
        # Compute wind direction relative to fire front normal
        # Wind direction: 0 = east (positive x), increases counterclockwise
        # Fire front normal: (nx, ny) points outward
        wind_nx = np.cos(self.wind_direction)
        wind_ny = np.sin(self.wind_direction)
        
        # Project wind onto fire front normal (positive = wind pushing fire outward)
        wind_component = wind_nx * nx + wind_ny * ny
        
        # Wind factor (simplified Rothermel wind function)
        # C = 7.47 * exp(-0.133 * sigma^0.55) where sigma is SAVR
        # For simplicity, use a constant C
        C = 7.47 * np.exp(-0.133 * 88.0**0.55)  # Using typical SAVR = 88 m⁻¹
        B = 1.0  # Wind exponent (typically 1.0-2.0 in Rothermel model)
        wind_factor = C * (wind_speed_m_per_min / 88.0)**B * wind_component
        # Limit wind factor to reasonable range
        wind_factor = np.clip(wind_factor, -0.5, 3.0)
        
        # Slope factor
        # Compute slope direction relative to fire front normal
        slope_nx = np.cos(self.slope_aspect)
        slope_ny = np.sin(self.slope_aspect)
        slope_component = slope_nx * nx + slope_ny * ny
        
        # Slope factor (simplified)
        slope_factor = 5.275 * (self.slope / np.tan(np.radians(33))) * slope_component
        slope_factor = np.clip(slope_factor, -0.5, 2.0)
        
        # Effective heating number (simplified)
        # Typically depends on fuel particle size, using constant for simplicity
        effective_heating_number = 0.5  # dimensionless
        
        # Heat of preignition (simplified)
        heat_of_preignition = 250.0  # kJ/kg
        
        # Rate of spread (simplified Rothermel formula)
        # ROS = (IR * ξ * (1 + φW + φS)) / (ρb * ε * Qig)
        # where IR = reaction intensity, ξ = propagating flux ratio, 
        # φW = wind factor, φS = slope factor, ρb = bulk density,
        # ε = effective heating number, Qig = heat of preignition
        
        # Propagating flux ratio (simplified, typically 0.1-0.3)
        propagating_flux_ratio = 0.2
        
        # Compute ROS
        numerator = reaction_intensity * propagating_flux_ratio * (1.0 + wind_factor + slope_factor)
        denominator = bulk_density * effective_heating_number * heat_of_preignition + 1e-10
        
        ros = numerator / denominator  # m/s
        
        # Ensure ROS is non-negative (fire can't spread backward)
        ros = np.maximum(ros, 0.0)
        
        # Convert ROS from m/s to ft/min
        # 1 m/s = 196.8504 ft/min
        METERS_PER_SEC_TO_FEET_PER_MIN = 196.8504
        ros_ft_per_min = ros * METERS_PER_SEC_TO_FEET_PER_MIN
        
        return ros_ft_per_min
    
    def _update_frame(self, frame):
        """
        Update function for animation frames.
        
        Evolves the fire perimeter using Rothermel ROS speeds at each point.
        Different points evolve at different rates based on local ROS values.
        
        Parameters:
        -----------
        frame : int
            Current frame number
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The updated axes object
        """
        t = frame * (self.T / self.num_steps)  # Current time in minutes
        
        # Compute ROS at current fire front (in ft/min)
        ros_ft_per_min = self._compute_rothermel_ros(self.phi_prev, t)
        
        # Compute gradient for level set evolution
        grad_y, grad_x = np.gradient(self.phi_prev, self.dy, self.dx)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
        
        # Time step in minutes
        dt_min = self.T / self.num_steps
        
        # Convert ROS from ft/min to meters per minute, then to displacement per timestep
        # 1 foot = 0.3048 meters
        METERS_TO_FEET = 3.28084
        ros_m_per_min = ros_ft_per_min / METERS_TO_FEET
        ros_displacement_m = ros_m_per_min * dt_min  # displacement in meters per timestep
        
        # Level set evolution: dphi/dt = -V * |∇phi|
        # For expansion outward, we subtract: phi_new = phi_old - V * |∇phi| * dt
        # This makes phi more negative (expands the interior, since negative = inside)
        # phi is in meters, so ros_displacement_m is in the correct units
        phi = self.phi_prev.copy()
        
        # Apply level set evolution
        # The fire front is at phi = 0, and we want it to expand outward
        phi = phi - ros_displacement_m * grad_mag
        
        # Renormalize to maintain signed distance function property (|∇phi| ≈ 1)
        # This ensures the zero level set (phi = 0) accurately represents the perimeter
        # Simple renormalization: phi_renorm = phi * sign(phi) / (|∇phi| + epsilon)
        # But a better approach is to use the fast marching method or simply ensure
        # the zero level set is preserved while maintaining SDF property near it
        # For now, we'll do a simple renormalization near the zero level set
        zero_band = np.abs(phi) < 2.0 * max(abs(self.dx), abs(self.dy))  # Within 2 grid cells of zero
        if np.any(zero_band):
            # Renormalize: phi_new = phi_old * sign(phi_old) / max(|∇phi|, 1)
            # This maintains the sign and ensures |∇phi| doesn't become too large
            grad_y_phi, grad_x_phi = np.gradient(phi, self.dy, self.dx)
            grad_mag_phi = np.sqrt(grad_x_phi**2 + grad_y_phi**2) + 1e-10
            # Only renormalize where gradient magnitude is significantly different from 1
            renormalize_mask = zero_band & (grad_mag_phi > 1.5)
            if np.any(renormalize_mask):
                phi[renormalize_mask] = phi[renormalize_mask] / grad_mag_phi[renormalize_mask]
        
        # Ensure no contraction for points that started inside the fire
        # For points inside (phi_inner < 0), expansion means phi becomes MORE NEGATIVE
        inside_initial = self.phi_inner < 0
        contraction_mask = inside_initial & (phi > self.phi_prev)
        if np.any(contraction_mask):
            phi[contraction_mask] = self.phi_prev[contraction_mask]
        
        # Update phi_prev
        self.phi_prev = phi.copy()
        
        # Clear and replot
        self.ax.clear()
        # Use persisted limits from create_animation (cartesian grid)
        if hasattr(self, '_xlim') and hasattr(self, '_ylim'):
            self.ax.set_xlim(self._xlim[0], self._xlim[1])
            self.ax.set_ylim(self._ylim[0], self._ylim[1])
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Rothermel ROS Evolution at t = {t:.2f} minutes')
        
        # Plot the evolving contour and label contour values
        cs = self.ax.contour(self.X, self.Y, phi, colors='red', linewidths=2)
        self.ax.clabel(cs, inline=True, fontsize=8)
        
        return self.ax,
    
    def create_animation(self, interval=200, xlim=(-2, 2), ylim=(-2, 2)):
        """
        Create and display the animation.
        
        Parameters:
        -----------
        interval : int
            Time between frames in milliseconds
        xlim : tuple
            X-axis limits (min, max)
        ylim : tuple
            Y-axis limits (min, max)
        """
        # Persist axis limits for use during frame updates
        self._xlim = xlim
        self._ylim = ylim

        # Reset phi_prev for animation
        self.phi_prev = self.phi_inner.copy()
        
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])
        self.ax.set_aspect('equal')
        self.ax.set_title('Fire Perimeter Evolution via Rothermel Rate-of-Spread Model')
        
        
        # Create the animation
        self.ani = FuncAnimation(self.fig, self._update_frame, 
                                 frames=range(self.num_steps + 1), interval=interval)
        
        # Display the animation
        plt.show()
    
    def save_animation(self, filename='contour_evolution_rothermel.gif', fps=10):
        """
        Save the animation to a file.
        
        Parameters:
        -----------
        filename : str
            Output filename for the animation
        fps : int
            Frames per second for the saved animation
        
        Raises:
        -------
        RuntimeError
            If animation has not been created yet
        """
        if self.ani is None:
            raise RuntimeError("Animation must be created before saving. Call create_animation() first.")
        
        self.ani.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved to {filename}")

