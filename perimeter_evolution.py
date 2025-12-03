"""
Module for evolving perimeters and creating animations.

This module contains the main logic for evolving contours over time using
signed distance function interpolation, computing speed maps, and creating
animations of the evolution process.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PerimeterEvolution:
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
        
        # Linear interpolation of SDFs
        phi = (1 - s) * self.phi_inner + s * self.phi_outer
        
        # Compute current gradient and speed field
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
        self.ax.set_title(f'Contour Evolution at t = {t:.2f}')
        
        # Plot the speed traces using pcolormesh with fixed color range
        masked_speed = np.ma.masked_invalid(self.speed_map)
        self.ax.pcolormesh(self.X, self.Y, masked_speed, cmap='jet', 
                          vmin=self.vmin, vmax=self.vmax, shading='gouraud')
        
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
        self.ax.set_title('Contour Evolution via SDF Interpolation with Speed Traces')
        
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

