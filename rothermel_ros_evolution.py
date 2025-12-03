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
    
    def __init__(self, phi_inner, phi_outer, X, Y, T=1.0, num_steps=20,
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
            Total evolution time (arbitrary units)
        num_steps : int
            Number of animation frames
        fuel_load : numpy.ndarray, optional
            Fuel load (kg/m²) at each grid point. If None, uses uniform default.
        fuel_depth : numpy.ndarray, optional
            Fuel depth (m) at each grid point. If None, uses uniform default.
        fuel_moisture : numpy.ndarray, optional
            Fuel moisture content (fraction) at each grid point. If None, uses uniform default.
        wind_speed : numpy.ndarray, optional
            Wind speed (m/s) at each grid point. If None, uses uniform default.
        wind_direction : numpy.ndarray, optional
            Wind direction (radians, 0 = east) at each grid point. If None, uses uniform default.
        slope : numpy.ndarray, optional
            Slope (radians) at each grid point. If None, uses uniform default.
        slope_aspect : numpy.ndarray, optional
            Slope aspect (radians, 0 = east) at each grid point. If None, uses uniform default.
        config_file : str, optional
            Path to YAML configuration file. If None, uses default config file in same directory,
            or falls back to hardcoded defaults if file doesn't exist.
        """
        self.phi_inner = phi_inner
        self.phi_outer = phi_outer
        self.X = X
        self.Y = Y
        self.T = T
        self.num_steps = num_steps
        
        # Compute grid spacing
        x = X[0, :]
        y = Y[:, 0]
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        
        # Load configuration from YAML file
        config = self._load_config(config_file)
        
        # Set default fuel and environmental parameters if not provided
        # Use values from config file, or fall back to hardcoded defaults
        self.fuel_load = fuel_load if fuel_load is not None else np.ones_like(X) * config['fuel_load']
        self.fuel_depth = fuel_depth if fuel_depth is not None else np.ones_like(X) * config['fuel_depth']
        self.fuel_moisture = fuel_moisture if fuel_moisture is not None else np.ones_like(X) * config['fuel_moisture']
        self.wind_speed = wind_speed if wind_speed is not None else np.ones_like(X) * config['wind_speed']
        self.wind_direction = wind_direction if wind_direction is not None else np.ones_like(X) * config['wind_direction']
        self.slope = slope if slope is not None else np.ones_like(X) * config['slope']
        self.slope_aspect = slope_aspect if slope_aspect is not None else np.ones_like(X) * config['slope_aspect']
        
        # Rothermel model constants from config
        self.heat_content = config['heat_content']
        self.mineral_damping = config['mineral_damping']
        self.max_reaction_velocity = 0.0  # Will be computed from fuel properties
        
        # Initialize speed map and tracking variables
        self.speed_map = np.full_like(phi_inner, np.nan)
        self.phi_prev = phi_inner.copy()
        
        # Precompute speed range for consistent color mapping
        self.vmin, self.vmax = self._precompute_speed_range()
        
        # Initialize speed map with initial values
        self._initialize_speed_map()
        
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
        default_config = {
            'fuel_load': 0.5,
            'fuel_depth': 0.3,
            'fuel_moisture': 0.15,
            'wind_speed': 5.0,
            'wind_direction': 0.0,
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
            Rate of spread (m/s) at each grid point
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
        
        # Scale ROS to match the evolution time scale
        # The ROS needs to be scaled so that the fire can reach phi_outer in time T
        # We'll scale based on the maximum distance to cover
        max_distance = np.max(np.abs(self.phi_outer - self.phi_inner))
        if max_distance > 0 and np.max(ros) > 0:
            # Scale ROS so that the maximum distance can be covered
            # Use a scaling factor that ensures reasonable evolution speed
            scale_factor = max_distance / (self.T * np.max(ros) + 1e-10)
            # Apply scaling but keep relative differences
            ros = ros * scale_factor
        
        return ros
    
    def _precompute_speed_range(self):
        """
        Precompute min and max speeds across all frames for consistent color mapping.
        
        Returns:
        --------
        vmin : float
            Minimum speed value
        vmax : float
            Maximum speed value
        """
        speeds = []
        
        # Initial speed
        ros_initial = self._compute_rothermel_ros(self.phi_inner, 0.0)
        initial_mask = self.phi_inner < 0
        speeds.extend(ros_initial[initial_mask].flatten())
        
        # Compute speeds for all frames
        phi_prev = self.phi_inner.copy()
        for frame in range(1, self.num_steps + 1):
            t = frame * (self.T / self.num_steps)
            # Estimate phi at this time (using linear interpolation as approximation)
            s = t / self.T
            phi = (1 - s) * self.phi_inner + s * self.phi_outer
            ros = self._compute_rothermel_ros(phi, t)
            new_mask = (phi_prev >= 0) & (phi < 0)
            speeds.extend(ros[new_mask].flatten())
            phi_prev = phi.copy()
        
        if len(speeds) == 0:
            return 0.0, 1.0
        
        return np.min(speeds), np.max(speeds)
    
    def _initialize_speed_map(self):
        """Initialize the speed map with values at t=0."""
        ros_initial = self._compute_rothermel_ros(self.phi_inner, 0.0)
        initial_mask = self.phi_inner < 0
        self.speed_map[initial_mask] = ros_initial[initial_mask]
    
    def _update_frame(self, frame):
        """
        Update function for animation frames.
        
        Evolves the fire perimeter using level set method with Rothermel ROS speeds.
        
        Parameters:
        -----------
        frame : int
            Current frame number
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The updated axes object
        """
        t = frame * (self.T / self.num_steps)  # Current time
        
        # Stop evolution after T
        if t >= self.T:
            phi = self.phi_outer.copy()
        else:
            s = t / self.T  # Normalized time [0, 1]
            
            # Compute ROS at current fire front
            ros = self._compute_rothermel_ros(self.phi_prev, t)
            
            # Compute gradient for level set evolution
            grad_y, grad_x = np.gradient(self.phi_prev, self.dy, self.dx)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
            
            # Level set evolution: dphi/dt = -V * |∇phi|
            # where V is the ROS normal to the front
            dt = self.T / self.num_steps
            
            # Evolve phi using level set method with Rothermel ROS
            phi_ros = self.phi_prev.copy()
            
            # For points that should expand (inside initial contour)
            expand_mask = (self.phi_inner < 0) & (self.phi_prev < self.phi_outer)
            
            # Apply level set evolution based on ROS
            if np.any(expand_mask):
                # Move outward (decrease phi, making it more negative or less positive)
                # The ROS gives the speed, so we evolve: dphi = -V * |∇phi| * dt
                phi_ros[expand_mask] = self.phi_prev[expand_mask] - dt * ros[expand_mask] * grad_mag[expand_mask]
            
            # Linear interpolation target (ensures we reach phi_outer at t=T)
            phi_interp = (1 - s) * self.phi_inner + s * self.phi_outer
            
            # Blend ROS evolution with interpolation to ensure we reach target
            # Weight interpolation more heavily as we approach t=T to guarantee convergence
            weight_interp = s ** 0.5  # Increases from 0 to 1 as s increases
            weight_ros = 1.0 - weight_interp
            
            # Combine both approaches
            phi = weight_ros * phi_ros + weight_interp * phi_interp
            
            # Ensure no contraction and don't exceed phi_outer
            expand_mask_final = (self.phi_inner < 0) & (self.phi_prev < self.phi_outer)
            if np.any(expand_mask_final):
                # No contraction: phi should be <= phi_prev (more negative or less positive)
                phi[expand_mask_final] = np.minimum(phi[expand_mask_final], self.phi_prev[expand_mask_final])
                # Don't exceed phi_outer
                phi[expand_mask_final] = np.maximum(phi[expand_mask_final], self.phi_outer[expand_mask_final])
            
            # At t=T, ensure exact match with phi_outer
            if s >= 1.0:
                phi = self.phi_outer.copy()
        
        # Compute current ROS for visualization
        ros_current = self._compute_rothermel_ros(phi, t)
        
        # Find newly incorporated points
        new_mask = (self.phi_prev >= 0) & (phi < 0)
        self.speed_map[new_mask] = ros_current[new_mask]
        
        # Update phi_prev
        self.phi_prev = phi.copy()
        
        # Clear and replot
        self.ax.clear()
        # Use persisted limits from create_animation
        if hasattr(self, '_xlim') and hasattr(self, '_ylim'):
            self.ax.set_xlim(self._xlim[0], self._xlim[1])
            self.ax.set_ylim(self._ylim[0], self._ylim[1])
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Rothermel ROS Evolution at t = {t:.2f}')
        
        # Plot the speed traces using pcolormesh with fixed color range
        masked_speed = np.ma.masked_invalid(self.speed_map)
        self.ax.pcolormesh(self.X, self.Y, masked_speed, cmap='jet', 
                          vmin=self.vmin, vmax=self.vmax, shading='gouraud')
        
        # Plot the evolving contour
        self.ax.contour(self.X, self.Y, phi, levels=[0], colors='red', linewidths=2)
        
        # Plot the target outer contour for reference
        self.ax.contour(self.X, self.Y, self.phi_outer, levels=[0], 
                       colors='green', linestyles='dashed', linewidths=1.5)
        
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
        
        # Add colorbar with fixed range
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
        self.fig.colorbar(sm, ax=self.ax, label='ROS (m/s)')
        
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

