import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
import sys

# --- Helper: Load Config ---
def load_config(filename="config.yaml"):
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        sys.exit(1)

# --- Physics Engine ---
class RothermelCalculator:
    """
    Calculates ROS based on fuel properties using simplified
    relationships from Rothermel (1972) & Albini (1976).
    """
    def __init__(self, fuel_cfg, env_cfg):
        # 1. Extract Inputs
        self.w0 = fuel_cfg['load']         # kg/m^2
        self.delta = fuel_cfg['depth']     # m
        self.Mf = fuel_cfg['moisture']     # fraction
        self.Mx = fuel_cfg['extinction_moisture']
        self.U = env_cfg['wind_speed']     # m/s
        self.wind_dir = np.array(env_cfg['wind_direction'])
        
        # Constants for "Grass-like" fuel
        self.rhop = 512.0  # Particle density (kg/m^3) - approx for wood/vegetation
        self.sigma = 4000.0 # Surface-area-to-volume ratio (1/ft converted -> ~4000 1/m for grass)
        
        # 2. Calculate Intermediate Fuel Parameters
        # Packing Ratio (beta): Fraction of bed volume occupied by fuel
        # beta = Load / (Depth * Particle Density)
        self.beta = self.w0 / (self.delta * self.rhop)
        
        # Optimum Packing Ratio (beta_op)
        # Simplified approximation for typical fuels
        self.beta_op = 0.2 
        
        # 3. Calculate Base Rate of Spread (R0) - No Wind
        # Moisture Damping coeff (eta_M): 1 if dry, 0 if at extinction
        self.eta_M = 1.0 - (self.Mf / self.Mx)
        if self.eta_M < 0: self.eta_M = 0
        
        # Propagating Flux (xi):
        # A complex function of packing ratio. We use a Gaussian approximation
        # where spread is best near optimum packing and drops off if too loose or too tight.
        xi = np.exp(-138.0 * (self.beta - self.beta_op)**2)
        
        # Theoretical max spread for this fuel type (arbitrary scaling factor for demo)
        # In full model, this depends on heat content, mineral damping, etc.
        max_spread_potential = 2.0 # m/s for dry, optimal grass
        
        self.R0 = max_spread_potential * self.w0 * xi * self.eta_M * 10.0
        
        # 4. Calculate Wind Coefficient (phi_w)
        # Wind effect depends on packing ratio.
        # Denser fuel (high beta) is less affected by wind.
        # C, B, E are standard Rothermel coefficients
        C = 7.47 * np.exp(-0.133 * self.sigma / 3.28) # sigma converted back for coeff
        B = 0.02526 * (self.sigma / 3.28)**0.54
        E = 0.715 * np.exp(-3.59e-4 * self.sigma / 3.28)
        
        # phi_w = C * U^B * (beta / beta_op)^-E
        # Simplified for robustness in this demo:
        self.wind_factor = (self.U ** 1.4) * (1 + 0.5*np.exp(-self.beta * 100)) * 0.1

    def get_local_ros(self, nx, ny):
        """
        Returns ROS at a point given the normal vector (nx, ny).
        """
        # Base Rate
        ros = self.R0
        
        # Alignment with wind vector
        alignment = nx * self.wind_dir[0] + ny * self.wind_dir[1]
        
        # Calculate Phi_w (Wind multiplier)
        # Only effective if wind is pushing the fire (alignment > 0)
        phi_w = 0.0
        if alignment > 0:
            phi_w = self.wind_factor * (alignment ** 1.2)
            
        return ros * (1 + phi_w)

class LevelSetFire:
    def __init__(self, config):
        domain_cfg = config['domain']
        self.domain_size = domain_cfg['size']
        self.grid_res = domain_cfg['resolution']
        
        # Setup Grid
        self.dx = self.domain_size / self.grid_res
        self.dy = self.domain_size / self.grid_res
        x = np.linspace(0, self.domain_size, self.grid_res)
        y = np.linspace(0, self.domain_size, self.grid_res)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initial SDF
        center = domain_cfg['center']
        radius = domain_cfg['initial_radius']
        self.phi = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2) - radius
        
        # Initialize Physics
        self.physics = RothermelCalculator(config['fuel'], config['environment'])
        self.cfl_limit = config['simulation']['cfl_limit']

    def get_upwind_grad_mag(self):
        phi = self.phi
        dx, dy = self.dx, self.dy
        
        phi_pad_x = np.pad(phi, ((0,0), (1,1)), mode='edge')
        D_minus_x = (phi - phi_pad_x[:, :-2]) / dx
        D_plus_x  = (phi_pad_x[:, 2:] - phi) / dx
        
        phi_pad_y = np.pad(phi, ((1,1), (0,0)), mode='edge')
        D_minus_y = (phi - phi_pad_y[:-2, :]) / dy
        D_plus_y  = (phi_pad_y[2:, :] - phi) / dy
        
        term_x = np.maximum(np.maximum(D_minus_x, 0)**2, np.minimum(D_plus_x, 0)**2)
        term_y = np.maximum(np.maximum(D_minus_y, 0)**2, np.minimum(D_plus_y, 0)**2)
        return np.sqrt(term_x + term_y)

    def get_central_normals(self):
        dy, dx = np.gradient(self.phi, self.dy, self.dx)
        mag = np.sqrt(dx**2 + dy**2)
        mag[mag < 1e-6] = 1e-6 
        return dx/mag, dy/mag

    def update_physics_step(self, max_dt_visual):
        t_remaining = max_dt_visual
        
        # Pre-calculate speed map (Optimization: In a uniform bed, R0 and Wind Factor are constant)
        # But directional component changes with normal.
        
        while t_remaining > 1e-5:
            nx, ny = self.get_central_normals()
            
            # Vectorized speed calculation
            # We map the scalar function get_local_ros over the grid
            # Since inputs are arrays (nx, ny), we need to do this efficiently
            
            # Extract constants from physics engine
            R0 = self.physics.R0
            wind_fac = self.physics.wind_factor
            wdir = self.physics.wind_dir
            
            # Inline vectorized calculation for speed
            alignment = nx * wdir[0] + ny * wdir[1]
            phi_w = np.zeros_like(alignment)
            mask = alignment > 0
            phi_w[mask] = wind_fac * (alignment[mask] ** 1.2)
            F = R0 * (1 + phi_w)
            
            # CFL Check
            max_v = np.max(F)
            if max_v > 1e-6:
                dt_cfl = self.cfl_limit * self.dx / max_v
            else:
                dt_cfl = t_remaining
                
            dt = min(dt_cfl, t_remaining)
            
            grad_mag = self.get_upwind_grad_mag()
            delta_phi = -dt * F * grad_mag
            
            self.phi += np.minimum(delta_phi, 0)
            t_remaining -= dt

# --- Main Execution ---
if __name__ == "__main__":
    cfg = load_config('wrf-fpe/rothermel-test/config.yaml')
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    d_size = cfg['domain']['size']
    ax.set_aspect('equal')
    ax.set_xlim(0, d_size)
    ax.set_ylim(0, d_size)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Visual Context
    w_speed = cfg['environment']['wind_speed']
    ax.arrow(d_size*0.05, d_size*0.05, d_size*0.08, 0, 
             head_width=d_size*0.02, color='blue', alpha=0.5)
    ax.text(d_size*0.05, d_size*0.08, f"Wind: {w_speed} m/s", color='blue')

    fire_model = LevelSetFire(cfg)
    
    contour_collection = []
    current_sim_time = 0.0
    visual_dt = cfg['simulation']['visual_dt']
    total_frames = cfg['simulation']['timesteps']

    def animate(i):
        global current_sim_time
        fire_model.update_physics_step(visual_dt)
        current_sim_time += visual_dt
        
        # Info String
        f = cfg['fuel']
        info = f"Time: {current_sim_time:.0f}s | Moist: {f['moisture']} | Load: {f['load']}"
        ax.set_title(f"Rothermel Fire Spread\n{info}")
        
        for c in contour_collection:
            c.remove()
        contour_collection.clear()
        
        c = ax.contour(fire_model.X, fire_model.Y, fire_model.phi, 
                       levels=[0], colors='red', linewidths=2)
        contour_collection.append(c)
        return contour_collection

    print("Simulating...")
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=50, blit=False)
    
    # --- SAVE LOGIC ---
    plt.show()
    if cfg['simulation'].get('save_gif', False):
        output_file = cfg['simulation'].get('output_filename', 'fire.gif')
        print(f"Saving animation to {output_file} (this may take a moment)...")
        # Requires 'pillow' installed
        ani.save(output_file, writer='pillow', fps=15)
        print("Save complete.")