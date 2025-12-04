import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
import sys
from scipy.ndimage import gaussian_filter

# --- Helper: Math & Config ---
def load_config(filename="config.yaml"):
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        sys.exit(1)

def degrees_to_vec(deg):
    """Converts meteorological degrees (0=N, 90=E) to unit vector."""
    rad = np.radians(deg)
    return np.array([np.sin(rad), np.cos(rad)])

# --- Physics Engine ---
class RothermelCalculator:
    def __init__(self, fuel_cfg, env_cfg):
        self.w0 = fuel_cfg['load']         
        self.delta = fuel_cfg['depth']     
        self.Mf = fuel_cfg['moisture']     
        self.Mx = fuel_cfg['extinction_moisture']
        self.U = env_cfg['wind_speed']     
        self.wind_dir = degrees_to_vec(env_cfg['wind_direction'])
        
        # Standard Constants
        self.rhop = 512.0  
        self.sigma = 4000.0 
        
        # Fuel Calcs
        self.beta = self.w0 / (self.delta * self.rhop)
        self.beta_op = 0.2 
        
        self.eta_M = 1.0 - (self.Mf / self.Mx)
        if self.eta_M < 0: self.eta_M = 0
        
        xi = np.exp(-138.0 * (self.beta - self.beta_op)**2)
        max_spread_potential = 2.0 
        self.R0 = max_spread_potential * self.w0 * xi * self.eta_M * 10.0
        
        self.wind_factor_coef = (1 + 0.5*np.exp(-self.beta * 100)) * 0.1

    def calculate_ros_map(self, nx, ny, terrain_grad_x, terrain_grad_y):
        ros = self.R0
        
        # Wind Alignment
        wind_align = nx * self.wind_dir[0] + ny * self.wind_dir[1]
        phi_w = np.zeros_like(wind_align)
        mask_w = wind_align > 0
        phi_w[mask_w] = self.wind_factor_coef * (self.U ** 1.4) * (wind_align[mask_w] ** 1.2)
        
        # Slope Alignment
        slope_align = nx * terrain_grad_x + ny * terrain_grad_y
        phi_s = np.zeros_like(slope_align)
        mask_s = slope_align > 0
        
        if np.any(mask_s):
            tan_phi = slope_align[mask_s]
            slope_coef = 5.275 * (self.beta ** -0.3)
            phi_s[mask_s] = slope_coef * (tan_phi ** 2)

        total_ros = ros * (1 + phi_w + phi_s)
        return total_ros

class LevelSetFire:
    def __init__(self, config):
        domain_cfg = config['domain']
        self.domain_size = domain_cfg['size']
        self.grid_res = domain_cfg['resolution']
        
        self.dx = self.domain_size / self.grid_res
        self.dy = self.domain_size / self.grid_res
        x = np.linspace(0, self.domain_size, self.grid_res)
        y = np.linspace(0, self.domain_size, self.grid_res)
        self.X, self.Y = np.meshgrid(x, y)
        
        # 1. Generate Terrain Elevation (Z)
        terrain_cfg = config.get('terrain', {})
        self.Z = self._generate_elevation_grid(terrain_cfg)
        self.dz_dy, self.dz_dx = np.gradient(self.Z, self.dy, self.dx)

        # 2. Generate Fuel Mask (Lakes)
        self.burn_mask = self._generate_lakes(terrain_cfg.get('lakes', {}))

        # 3. Initial SDF
        center = domain_cfg['center']
        radius = domain_cfg['initial_radius']
        self.phi = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2) - radius
        
        self.physics = RothermelCalculator(config['fuel'], config['environment'])
        self.cfl_limit = config['simulation']['cfl_limit']

    def _generate_elevation_grid(self, t_cfg):
        """
        Generates Z based on selected model and parameters.
        """
        model = t_cfg.get('model', 'flat').lower()
        max_z = t_cfg.get('max_elevation', 0.0)
        Z = np.zeros_like(self.X)
        
        if model == 'flat':
            pass
            
        elif model == 'gradient':
            # --- Gradient Logic ---
            grad_deg = t_cfg.get('gradient_direction', 90.0)
            ramp_dir = degrees_to_vec(grad_deg)
            slope_mag = max_z / self.domain_size
            Z = slope_mag * (self.X * ramp_dir[0] + self.Y * ramp_dir[1])
            
        elif model == 'periodic':
            # --- Periodic Logic ---
            freq = t_cfg.get('frequency', 1.0)
            kx = (self.X / self.domain_size) * 2 * np.pi * freq
            ky = (self.Y / self.domain_size) * 2 * np.pi * freq
            wave = (np.sin(kx) * np.sin(ky)) 
            Z = ((wave + 1) / 2) * max_z
            
        elif model == 'random':
            # --- Random Logic ---
            seed = t_cfg.get('seed', 42)
            smoothness = t_cfg.get('smoothness', 10.0)
            
            if seed is not None:
                np.random.seed(seed)
                
            noise = np.random.rand(*self.X.shape)
            Z = gaussian_filter(noise, sigma=smoothness)
            # Normalize to 0-1 then scale to max_z
            Z = (Z - Z.min()) / (Z.max() - Z.min()) * max_z
            
        # Clean up
        Z -= np.min(Z)
        return Z

    def _generate_lakes(self, lake_cfg):
        """Creates binary mask: 1.0=Burnable, 0.0=Lake."""
        mask = np.ones_like(self.X)
        
        if not lake_cfg.get('enabled', False):
            return mask

        count = lake_cfg.get('count', 5)
        r_min = lake_cfg.get('min_radius', 20)
        r_max = lake_cfg.get('max_radius', 50)
        seed = lake_cfg.get('seed', None)
        
        # Use a local RandomState so we don't mess up global seed if used elsewhere
        rng = np.random.RandomState(seed) if seed is not None else np.random
            
        for _ in range(count):
            cx = rng.uniform(0, self.domain_size)
            cy = rng.uniform(0, self.domain_size)
            r = rng.uniform(r_min, r_max)
            dist = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
            mask[dist < r] = 0.0
            
        return mask

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
        while t_remaining > 1e-5:
            nx, ny = self.get_central_normals()
            F = self.physics.calculate_ros_map(nx, ny, self.dz_dx, self.dz_dy)
            
            # Apply Lakes
            F = F * self.burn_mask
            
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
    cfg = load_config("wrf-fpe/rothermel-test/config.yaml")
    
    fig, ax = plt.subplots(figsize=(8, 9))
    plt.subplots_adjust(bottom=0.22)
    
    d_size = cfg['domain']['size']
    ax.set_aspect('equal')
    ax.set_xlim(0, d_size)
    ax.set_ylim(0, d_size)
    
    fire_model = LevelSetFire(cfg)
    
    # 1. Plot Terrain Background
    elev_contours = ax.contour(fire_model.X, fire_model.Y, fire_model.Z, 
                               levels=15, cmap='Greys', alpha=0.4, linewidths=1)
    
    # 2. Plot Lakes
    if cfg['terrain'].get('lakes', {}).get('enabled', False):
        lake_mask = 1.0 - fire_model.burn_mask
        ax.contourf(fire_model.X, fire_model.Y, lake_mask, levels=[0.5, 1.5], 
                    colors='lightblue', hatches=['XX'])
        ax.contour(fire_model.X, fire_model.Y, lake_mask, levels=[0.5], 
                   colors='blue', linewidths=0.5)

    # 3. Wind Arrow
    w_speed = cfg['environment']['wind_speed']
    w_deg = cfg['environment']['wind_direction']
    if w_speed > 0:
        w_vec = degrees_to_vec(w_deg)
        ax.arrow(d_size*0.05, d_size*0.95, w_vec[0]*d_size*0.08, w_vec[1]*d_size*0.08, 
                 head_width=d_size*0.02, color='blue', zorder=10)
        ax.text(d_size*0.02, d_size*0.9, f"Wind: {w_speed}m/s\n@{w_deg}°", color='blue', fontsize=9)
    
    # 4. Info Caption
    t_cfg = cfg['terrain']
    f_cfg = cfg['fuel']
    
    # Dynamic terrain detail string based on model
    t_model = t_cfg.get('model')
    t_details = ""
    if t_model == 'gradient':
        t_details = f"Dir: {t_cfg.get('gradient_direction')}°"
    elif t_model == 'periodic':
        t_details = f"Freq: {t_cfg.get('frequency')}"
    elif t_model == 'random':
        t_details = f"Smooth: {t_cfg.get('smoothness')}"
        
    lakes_on = t_cfg.get('lakes', {}).get('enabled')
    lake_txt = f"{t_cfg.get('lakes',{}).get('count')} Lakes" if lakes_on else "No Lakes"
    
    caption_text = (
        f"CONFIG: {t_model.upper()} Terrain ({t_details})\n"
        f"Max Elev: {t_cfg.get('max_elevation')}m | {lake_txt}\n"
        f"Fuel: Load {f_cfg['load']} | Depth {f_cfg['depth']} | Moist {f_cfg['moisture']}"
    )
    
    fig.text(0.12, 0.04, caption_text, fontsize=9, fontfamily='monospace',
             bbox=dict(facecolor='#f0f0f0', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    contour_collection = []
    current_sim_time = 0.0
    visual_dt = cfg['simulation']['visual_dt']
    total_frames = cfg['simulation']['timesteps']

    def animate(i):
        global current_sim_time
        fire_model.update_physics_step(visual_dt)
        current_sim_time += visual_dt
        
        ax.set_title(f"Fire Spread Simulation\nTime: {current_sim_time:.0f}s")
        
        for c in contour_collection:
            c.remove()
        contour_collection.clear()
        
        c = ax.contour(fire_model.X, fire_model.Y, fire_model.phi, 
                       levels=[0], colors='red', linewidths=2)
        contour_collection.append(c)
        return contour_collection

    print(f"Simulating {t_model} terrain...")
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=30, blit=False)

    plt.show()
    
    if cfg['simulation'].get('save_gif', False):
        out = cfg['simulation']['output_filename']
        print(f"Saving to {out}...")
        ani.save(out, writer='pillow', fps=15)
        print("Done.")