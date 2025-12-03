"""
Module for evolving perimeters and creating animations.

This module contains the main logic for evolving contours over time using
signed distance function interpolation, computing speed maps, and creating
animations of the evolution process.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PerimeterEvolutionV2:
    """
    Class to handle perimeter evolution with speed tracing.
    
    This class evolves an inner contour to match an outer contour over time,
    tracking the speed at which points are incorporated and creating visualizations.
    """
    
    def __init__(self, phi_inner, phi_outer, X, Y, T=1.0, num_steps=20):
        """
        Initialize the perimeter evolution.
        
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
        
        # Compute constant dphi/dt for each point
        self.delta = (phi_outer - phi_inner) / T
        
        # Initialize speed map and tracking variables
        self.speed_map = np.full_like(phi_inner, np.nan)
        self.phi_prev = phi_inner.copy()
        
        # Precompute speed range for consistent color mapping
        self.vmin, self.vmax = self._precompute_speed_range()
        
        # Initialize speed map with initial values
        self._initialize_speed_map()
        
        # Compute single bias vector from zero level of phi_inner
        # Find point where expansion is largest (phi_outer is greatest distance from phi_inner)
        # The expansion magnitude is |phi_outer - phi_inner|, largest where this is maximum
        delta_phi = phi_outer - phi_inner
        
        # Find points near zero level of phi_inner (within a small threshold)
        zero_threshold = 0.05 * np.max(np.abs(phi_inner))  # 5% of max absolute value
        zero_mask = np.abs(phi_inner) < zero_threshold
        
        if np.any(zero_mask):
            # Get expansion values at zero level
            # Expansion is largest where delta_phi is most negative (phi_outer << phi_inner)
            # or where |delta_phi| is maximum
            expansion_at_zero = delta_phi[zero_mask]
            
            # Find point with largest expansion (most negative delta_phi means largest distance)
            # Create a masked array to find index in original grid
            expansion_masked = np.full_like(delta_phi, np.nan)
            expansion_masked[zero_mask] = expansion_at_zero
            
            # Find point with maximum expansion magnitude (most negative delta_phi)
            max_expansion_idx = np.unravel_index(np.nanargmin(expansion_masked), expansion_masked.shape)
            
            # At the point of largest expansion, compute direction toward phi_outer
            # The direction is given by the gradient of phi_outer (points toward outer contour)
            grad_y_outer, grad_x_outer = np.gradient(phi_outer, self.dy, self.dx)
            grad_mag_outer = np.sqrt(grad_x_outer**2 + grad_y_outer**2) + 1e-10
            
            # Normalize gradient to get direction vector (points toward phi_outer = 0)
            self.bias_nx = grad_x_outer[max_expansion_idx] / grad_mag_outer[max_expansion_idx]
            self.bias_ny = grad_y_outer[max_expansion_idx] / grad_mag_outer[max_expansion_idx]
            self.bias_magnitude = np.abs(delta_phi[max_expansion_idx])
        else:
            # Fallback: find point of maximum expansion in entire field
            max_expansion_idx = np.unravel_index(np.argmin(delta_phi), delta_phi.shape)
            grad_y_outer, grad_x_outer = np.gradient(phi_outer, self.dy, self.dx)
            grad_mag_outer = np.sqrt(grad_x_outer**2 + grad_y_outer**2) + 1e-10
            self.bias_nx = grad_x_outer[max_expansion_idx] / grad_mag_outer[max_expansion_idx]
            self.bias_ny = grad_y_outer[max_expansion_idx] / grad_mag_outer[max_expansion_idx]
            self.bias_magnitude = np.abs(delta_phi[max_expansion_idx])
        
        # Animation variables
        self.fig = None
        self.ax = None
        self.ani = None
    
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
        grad_y, grad_x = np.gradient(self.phi_inner, self.dy, self.dx)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10  # Avoid division by zero
        V_field = -self.delta / grad_mag
        initial_mask = self.phi_inner < 0
        speeds.extend(V_field[initial_mask].flatten())
        
        # Compute speeds for all frames
        phi_prev = self.phi_inner.copy()
        for frame in range(1, self.num_steps + 1):
            t = frame * (self.T / self.num_steps)
            s = t / self.T
            phi = (1 - s) * self.phi_inner + s * self.phi_outer
            grad_y, grad_x = np.gradient(phi, self.dy, self.dx)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
            V_field = -self.delta / grad_mag
            new_mask = (phi_prev >= 0) & (phi < 0)
            speeds.extend(V_field[new_mask].flatten())
            phi_prev = phi.copy()
        
        return np.min(speeds), np.max(speeds)
    
    def _initialize_speed_map(self):
        """Initialize the speed map with values at t=0."""
        grad_y, grad_x = np.gradient(self.phi_inner, self.dy, self.dx)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
        V_field = -self.delta / grad_mag
        initial_mask = self.phi_inner < 0
        self.speed_map[initial_mask] = V_field[initial_mask]
    
    def _update_frame(self, frame):
        """
        Update function for animation frames.
        
        Uses a two-step evolution process:
        1. Biasing step: Push contour in direction of largest expansion
        2. Interpolation step: Linear interpolation of SDFs
        Combined with stronger weighting toward interpolation.
        
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
        s = t / self.T  # Normalized time [0, 1]
        
        # Stop evolution after T=1
        if t >= self.T:
            phi = self.phi_outer.copy()
        else:
            # Step 1: Biasing factor - use single bias vector computed from zero level
            # The bias vector is already computed in __init__ and stored as self.bias_nx, self.bias_ny
            bias_nx = self.bias_nx
            bias_ny = self.bias_ny
            
            # Compute expansion magnitude for speed calculation
            delta_phi = self.phi_outer - self.phi_inner
            
            # Compute gradient of current phi for level set evolution
            grad_y_prev, grad_x_prev = np.gradient(self.phi_prev, self.dy, self.dx)
            grad_mag_prev = np.sqrt(grad_x_prev**2 + grad_y_prev**2) + 1e-10
            
            # Biasing step: Push contour in direction of largest expansion
            # Use level set evolution: dphi/dt = -V * |∇phi| where V is speed in bias direction
            # Speed is proportional to the magnitude of expansion and time step
            dt = self.T / self.num_steps
            bias_speed = np.abs(delta_phi) / self.T  # Speed based on expansion magnitude
            # Project bias direction onto gradient direction
            bias_component = bias_nx * (grad_x_prev / grad_mag_prev) + bias_ny * (grad_y_prev / grad_mag_prev)
            # Only expand (bias_component should be positive for outward expansion)
            bias_component = np.maximum(bias_component, 0)
            
            # Apply biasing step: move in direction of largest expansion
            phi_bias = self.phi_prev.copy()
            # Only apply to points that need expansion (inside initial contour)
            expand_mask = (self.phi_inner < 0) & (self.phi_prev < self.phi_outer)
            if np.any(expand_mask):
                # Move outward (make phi less negative or more positive)
                phi_bias[expand_mask] = self.phi_prev[expand_mask] + dt * bias_speed[expand_mask] * bias_component[expand_mask] * grad_mag_prev[expand_mask]
                # Ensure no contraction: phi_bias should be >= phi_prev for expanding regions
                phi_bias[expand_mask] = np.maximum(phi_bias[expand_mask], self.phi_prev[expand_mask])
                # Clamp to not exceed phi_outer
                phi_bias[expand_mask] = np.minimum(phi_bias[expand_mask], self.phi_outer[expand_mask])
            
            # Step 2: Linear interpolation of SDFs (standard method)
            phi_interp = (1 - s) * self.phi_inner + s * self.phi_outer
            
            # Time-dependent weights to smooth expansion:
            # Early timesteps (s small): favor interpolation for faster expansion
            # Late timesteps (s large): favor bias for slower expansion
            # This creates more uniform expansion speed throughout
            # At s=0: weight_interp = 0.8, weight_bias = 0.2 (faster early expansion)
            # At s=1: weight_interp = 0.4, weight_bias = 0.6 (slower late expansion)
            weight_interp = 0.8 - 0.4 * s  # Decreases from 0.8 to 0.4 as s increases
            weight_bias = 0.2 + 5 * s    # Increases from 0.2 to 0.6 as s increases
            
            # Additional time-dependent factor to further reduce late acceleration
            # Stronger factor to compensate for interpolation acceleration
            k = 0.1  # Increased from 0.3 to 0.5 for stronger speed normalization
            time_factor = 1 #1.0 - k * s
            
            # Apply time factor to interpolation step to prevent acceleration
            # At s=0: time_factor = 1.0 (full interpolation)
            # At s=1: time_factor = 0.5 (reduced interpolation, more bias)
            phi = weight_bias * phi_bias + weight_interp * phi_interp * time_factor
            
            # Normalize to ensure we still reach phi_outer at t=T
            # Scale the result to maintain the target at s=1
            if s < 1.0:
                # Compute what phi_interp would be without time factor
                phi_interp_full = (1 - s) * self.phi_inner + s * self.phi_outer
                # Compute weighted average without time factor using time-dependent weights
                phi_full = weight_bias * phi_bias + weight_interp * phi_interp_full
                # Blend between time-adjusted and full to ensure we reach target
                # Use a blend factor that increases as s approaches 1
                blend_factor = s ** 2  # Quadratic to ensure smooth transition
                phi = (1 - blend_factor) * phi + blend_factor * phi_full
            
            # Ensure constraints:
            # 1. No contraction: phi should be >= phi_prev for expanding regions
            expand_mask_final = (self.phi_inner < 0) & (self.phi_prev < self.phi_outer)
            if np.any(expand_mask_final):
                phi[expand_mask_final] = np.maximum(phi[expand_mask_final], self.phi_prev[expand_mask_final])
                # 2. Don't exceed phi_outer
                phi[expand_mask_final] = np.minimum(phi[expand_mask_final], self.phi_outer[expand_mask_final])
            
            # At t=1, ensure exact match with phi_outer
            if s >= 1.0:
                phi = self.phi_outer.copy()
        
        # Compute current gradient and speed field for visualization
        grad_y, grad_x = np.gradient(phi, self.dy, self.dx)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
        V_field = -self.delta / grad_mag
        
        # Find newly incorporated points
        new_mask = (self.phi_prev >= 0) & (phi < 0)
        self.speed_map[new_mask] = V_field[new_mask]
        
        # Update phi_prev
        self.phi_prev = phi.copy()
        
        # Clear and replot
        self.ax.clear()
        # Use persisted limits from create_animation (cartesian grid)
        if hasattr(self, '_xlim') and hasattr(self, '_ylim'):
            self.ax.set_xlim(self._xlim[0], self._xlim[1])
            self.ax.set_ylim(self._ylim[0], self._ylim[1])
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Biased Contour Evolution at t = {t:.2f}')
        
        # Plot the speed traces using pcolormesh with fixed color range
        masked_speed = np.ma.masked_invalid(self.speed_map)
        self.ax.pcolormesh(self.X, self.Y, masked_speed, cmap='jet', 
                          vmin=self.vmin, vmax=self.vmax, shading='gouraud')
        
        # Plot single bias vector
        # Find representative point on zero contour for vector placement (centroid)
        zero_mask = np.abs(self.phi_inner) < 0.05 * np.max(np.abs(self.phi_inner))
        if np.any(zero_mask):
            # Use centroid of zero contour
            vec_x = np.mean(self.X[zero_mask])
            vec_y = np.mean(self.Y[zero_mask])
        else:
            # Use center of grid as fallback
            vec_x = (self.X[0, 0] + self.X[-1, -1]) / 2
            vec_y = (self.Y[0, 0] + self.Y[-1, -1]) / 2
        
        # Scale vector for visualization
        if hasattr(self, '_xlim') and hasattr(self, '_ylim'):
            xlim = self._xlim
            ylim = self._ylim
        else:
            xlim = (-2, 2)
            ylim = (-2, 2)
        scale = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 10  # Scale to ~1/10 of plot size
        
        # Plot single bias vector
        self.ax.quiver(vec_x, vec_y, self.bias_nx * scale, self.bias_ny * scale,
                      color='cyan', alpha=0.8, scale=1.0, width=0.005, 
                      headwidth=5, headlength=6, label='Bias Direction')
        
        # Plot the evolving contour
        self.ax.contour(self.X, self.Y, phi, levels=[0], colors='red')
        
        # Plot the target outer contour for reference
        self.ax.contour(self.X, self.Y, self.phi_outer, levels=[0], 
                       colors='green', linestyles='dashed')
        
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
        self.ax.set_title('Biased Contour Evolution via Two-Step SDF Evolution')
        
        # Plot single bias vector
        # Find representative point on zero contour for vector placement (centroid)
        zero_mask = np.abs(self.phi_inner) < 0.05 * np.max(np.abs(self.phi_inner))
        if np.any(zero_mask):
            # Use centroid of zero contour
            vec_x = np.mean(self.X[zero_mask])
            vec_y = np.mean(self.Y[zero_mask])
        else:
            # Use center of grid as fallback
            vec_x = (self.X[0, 0] + self.X[-1, -1]) / 2
            vec_y = (self.Y[0, 0] + self.Y[-1, -1]) / 2
        
        # Scale vector for visualization
        scale = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 10  # Scale to ~1/10 of plot size
        
        # Plot single bias vector
        self.ax.quiver(vec_x, vec_y, self.bias_nx * scale, self.bias_ny * scale,
                      color='cyan', alpha=0.8, scale=1.0, width=0.005, 
                      headwidth=5, headlength=6, label='Bias Direction')
        
        # Add colorbar with fixed range
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
        self.fig.colorbar(sm, ax=self.ax, label='Speed')
        
        # Create the animation
        self.ani = FuncAnimation(self.fig, self._update_frame, 
                                 frames=range(self.num_steps + 1), interval=interval)
        
        # Display the animation
        plt.show()
    
    def save_animation(self, filename='contour_evolution_with_speed_traces.gif', fps=10):
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
