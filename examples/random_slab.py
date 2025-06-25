import sys
import os
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
import pickle
import meep as mp
import numpy as np
import matplotlib
import h5py
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def choi_2011_geometry_slab(width_k0: float = None, sizez_k0:float = None, seed: int = 42):
    
    wavelength = 0.6  # microns
    k0 = 2 * np.pi / wavelength  # wavevector magnitude
    slab_width = width_k0 / k0
    particle_size = 0.2
    
    size_z = sizez_k0 / k0  # slab thickness in microns
    np.random.seed(seed)
    
    # Poisson disk sampling parameters - for squares
    min_distance = particle_size * 1.1  # Still used for initial spacing in annulus
    k = 1500  # More attempts before rejection to get better packing
    
    # Domain bounds - particles must fit entirely within bounds
    x_min = -slab_width/2 + particle_size/2
    x_max = slab_width/2 - particle_size/2
    z_min = -size_z/2 + 1.5 * particle_size/2
    z_max = size_z/2 - 1.5 * particle_size/2
    
    # Grid for fast neighbor lookup
    cell_size = min_distance / np.sqrt(2)
    grid_width = int(np.ceil((x_max - x_min) / cell_size)) + 1
    grid_height = int(np.ceil((z_max - z_min) / cell_size)) + 1
    grid = np.full((grid_width, grid_height), -1, dtype=int)
    
    points = []
    active_list = []
    
    # Start with random initial point
    initial_x = np.random.uniform(x_min, x_max)
    initial_z = np.random.uniform(z_min, z_max)
    points.append([initial_x, initial_z])
    active_list.append(0)
    
    # Add to grid
    grid_x = int((initial_x - x_min) / cell_size)
    grid_z = int((initial_z - z_min) / cell_size)
    if 0 <= grid_x < grid_width and 0 <= grid_z < grid_height:
        grid[grid_x, grid_z] = 0
    
    while active_list:
        # Pick random point from active list
        active_idx = np.random.randint(len(active_list))
        current_point_idx = active_list[active_idx]
        current_x, current_z = points[current_point_idx]
        
        found_valid = False
        
        for _ in range(k):
            # Generate random point in annulus around current point
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(min_distance, 2 * min_distance)
            
            new_x = current_x + radius * np.cos(angle)
            new_z = current_z + radius * np.sin(angle)
            
            # Check bounds
            if not (x_min <= new_x <= x_max and z_min <= new_z <= z_max):
                continue
            
            # Check grid position
            grid_x = int((new_x - x_min) / cell_size)
            grid_z = int((new_z - z_min) / cell_size)
            
            if not (0 <= grid_x < grid_width and 0 <= grid_z < grid_height):
                continue
            
            # Check neighboring cells for conflicts - strict no-overlap check for SQUARES
            valid = True
            half_size = particle_size / 2
            gap = particle_size * 0.05  # Small gap to ensure no touching
            
            for dx in range(max(0, grid_x-2), min(grid_width, grid_x+3)):
                for dz in range(max(0, grid_z-2), min(grid_height, grid_z+3)):
                    if grid[dx, dz] != -1:
                        neighbor_idx = grid[dx, dz]
                        neighbor_x, neighbor_z = points[neighbor_idx]
                        
                        # Check if squares overlap or touch
                        # Square 1: [new_x - half_size, new_x + half_size] x [new_z - half_size, new_z + half_size]
                        # Square 2: [neighbor_x - half_size, neighbor_x + half_size] x [neighbor_z - half_size, neighbor_z + half_size]
                        
                        # Check X overlap
                        x_overlap = not ((new_x + half_size + gap) <= (neighbor_x - half_size) or 
                                       (new_x - half_size) >= (neighbor_x + half_size + gap))
                        
                        # Check Z overlap  
                        z_overlap = not ((new_z + half_size + gap) <= (neighbor_z - half_size) or 
                                       (new_z - half_size) >= (neighbor_z + half_size + gap))
                        
                        # If both X and Z overlap, squares overlap
                        if x_overlap and z_overlap:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Add new point
                points.append([new_x, new_z])
                grid[grid_x, grid_z] = len(points) - 1
                active_list.append(len(points) - 1)
                found_valid = True
                break
        
        if not found_valid:
            # Remove from active list
            active_list.pop(active_idx)
    
    particle_centers = np.array(points)
    
    particle_centers = np.array(points)


    # Define the Meep material function
    def material_func(p):
        for cx, cz in particle_centers:
            if (abs(p.x - cx) <= half_size) and (abs(p.y - cz) <= half_size):
                return mp.Medium(index=2.0)  # eps=4 => n=2
        return mp.air

    material_func.do_averaging = False

    geometry = [mp.Block(center=mp.Vector3(),
                     size=mp.Vector3(slab_width,size_z),
                     material=material_func)]
    return geometry



def export_geometry(rot_angle=0):
    box_size = 2  # size of the box in μm
    box_eps = 4

    resolution = 60/0.6  # pixels/μm
    k0 = 2 * np.pi / 0.6  # wavevector magnitude for wavelength = 0.6 μm
    cell_y = np.ceil(100 / k0)
    cell_x = np.ceil(150 / k0 + 4)
    cell_size = mp.Vector3(cell_x, cell_y, 0)
    pml_layers = [mp.PML(thickness=3, direction=mp.X)]
    fsrc = 1.0 / 0.6  # frequency of planewave (wavelength = 1/fsrc)
    n = 1  # refractive index of homogeneous material
    default_material = mp.Medium(index=n)
    k_point = mp.Vector3(fsrc * n).rotate(mp.Vector3(z=1), rot_angle)

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fsrc, fwidth=fsrc/7, is_integrated=True),
            # src=mp.ContinuousSource(fsrc),
            amplitude=1.0,
            center=mp.Vector3(-(50 / k0), 0, 0),
            size=mp.Vector3(y=cell_y),
            direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
            eig_kpoint=k_point,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        # boundary_layers=pml_layers,
        geometry = choi_2011_geometry_slab(width_k0 = 50, sizez_k0 = 100, seed  = 42),
        # eps_averaging = False
        # force_complex_fields=True,
        # sources=sources,
        k_point=k_point,
        # default_material=default_material
    )

    sim.init_sim()
    eps_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Dielectric)
    
    with h5py.File("exported_epsilon_random_90.h5", "w") as f:
        f.create_dataset("epsilon", data=eps_data)
        f.attrs["resolution"] = resolution
        f.attrs["cell_x"] = cell_x
        f.attrs["cell_y"] = cell_y


def run_sim(rot_angle=0):
    
    # with h5py.File("exported_epsilon_random_90.h5", "r") as f:
    #     resolution = f.attrs["resolution"]
    #     cell_x = f.attrs["cell_x"]
    #     cell_y = f.attrs["cell_y"]
    #     eps_shape = f["epsilon"].shape
        # print(f"Loaded epsilon shape: {eps_shape}")


    resolution = 40/0.6  # pixels/μm
    k0 = 2 * np.pi / 0.6  # wavevector magnitude for wavelength = 0.6 μm
    cell_y = np.ceil(100 / k0)
    cell_x = np.ceil(150 / k0 + 4)
    cell_size = mp.Vector3(cell_x, cell_y, 0)
    pml_layers = [mp.PML(thickness=3, direction=mp.X)]
    fsrc = 1.0 / 0.6  # frequency of planewave (wavelength = 1/fsrc)
    n = 1  # refractive index of homogeneous material
    default_material = mp.Medium(index=n)
    k_point = mp.Vector3(fsrc * n).rotate(mp.Vector3(z=1), rot_angle)

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fsrc, fwidth=fsrc/7, is_integrated=True),
            # src=mp.ContinuousSource(fsrc),
            amplitude=1.0,
            center=mp.Vector3(-(50 / k0), 0, 0),
            size=mp.Vector3(y=cell_y),
            direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
            eig_kpoint=k_point,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=pml_layers,
        epsilon_input_file='exported_epsilon_random_90.h5',
        eps_averaging = True,
        # geometry = choi_2011_geometry_slab(width_k0 = 50, sizez_k0 = 100, seed  = 42),
        force_complex_fields=True,
        sources=sources,
        k_point=k_point,
        # default_material=default_material
    )

    flux_region = mp.FluxRegion(center=mp.Vector3(5, 0, 0), size=mp.Vector3(0, cell_y, 0))
    flux = sim.add_flux(fsrc, 0, 1, flux_region)

    ez_p_f = sim.add_dft_fields([mp.Ez], fsrc, 0, 1, where=mp.Volume(center=mp.Vector3(5, 0, 0), size=mp.Vector3(0,0,0)))
    hy_p_f = sim.add_dft_fields([mp.Hy], fsrc, 0, 1, where=mp.Volume(center=mp.Vector3(5, 0, 0), size=mp.Vector3(0,0,0)))
    ez_freq = sim.add_dft_fields([mp.Ez], fsrc, 0, 1, where=mp.Volume(center=mp.Vector3(0,0,0), size=cell_size))

    sim.run(until=int(2000/fsrc))
    # sim.solve_cw(
    #     tol=1e-6,                # Tolerance for convergence
    #     maxiters=10000          # Maximum number of iterations
    # )
    

    ez_val = sim.get_dft_array(ez_freq, mp.Ez, 0)
    hy_val_p = sim.get_dft_array(hy_p_f, mp.Hy, 0)
    ez_val_p = sim.get_dft_array(ez_p_f, mp.Ez, 0)
    flux_value = mp.get_fluxes(flux)[0]
    flux_freqs = mp.get_flux_freqs(flux)
    (x, y, z, w) = sim.get_array_metadata(vol=mp.Volume(center=mp.Vector3(0,0,0), size=cell_size))

    return {
        # 'sim': sim,
        'flux': flux,
        'hy_val_p': hy_val_p,
        'ez_val_p': ez_val_p,
        'ez_val': ez_val,
        'flux_value': flux_value,
        'flux_freqs': flux_freqs,
        'x': x,
        'y': y,
        'cell_x': cell_x,
        'cell_y': cell_y,
        'cell_size': cell_size
    }

if __name__ == "__main__":


    # export_geometry(0)  # Export the geometry to a file

    results = run_sim(0)  # Example rotation angle of 45 degrees
    # # plot_sim_results(results)    

    if rank == 0:
        # Strip Meep objects that aren't pickle-safe
        results_to_save = {
            k: v for k, v in results.items() if k not in ['sim', 'flux']
        }

        # Save to a pickle file
        pickle_file = "results_random_slab_0.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results_to_save, f)