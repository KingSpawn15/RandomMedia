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

    resolution = 40/0.6  # pixels/μm
    k0 = 2 * np.pi / 0.6  # wavevector magnitude for wavelength = 0.6 μm
    
    cell_y = 50 / k0

    # cell_y = 500 / k0
    
    cell_x = 150 / k0 + 4
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
        # geometry = choi_2011_geometry_slab(width_k0 = 50, sizez_k0 = 500, seed  = 42),
        force_complex_fields=True,
        sources=sources,
        k_point=k_point,
        default_material=default_material
    )

    sim.init_sim()
    # eps_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Dielectric)
    
    # with h5py.File("exported_epsilon_random_90.h5", "w") as f:
    #     f.create_dataset("epsilon", data=eps_data)
    #     f.attrs["resolution"] = resolution
    #     f.attrs["cell_x"] = cell_x
    #     f.attrs["cell_y"] = cell_y
    # sim.dump("examples/random_media_500_res40/",single_parallel_file=False)
    sim.dump("examples/vacuum_test/",single_parallel_file=False)

def run_sim(rot_angle=0):
    
    resolution = 40/0.6  # pixels/μm
    k0 = 2 * np.pi / 0.6  # wavevector magnitude for wavelength = 0.6 μm
    

    ###delte
    cell_y = 50 / k0
    
    
    # cell_y = 500 / k0


    cell_x = 150 / k0 + 4
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
        # geometry = choi_2011_geometry_slab(width_k0 = 50, sizez_k0 = 500, seed  = 42),
        force_complex_fields=True,
        sources=sources,
        k_point=k_point,
        default_material=default_material
    )
    sim.init_sim()

    sim.load("examples/vacuum_test/", single_parallel_file=False)


    # sim.load("examples/random_media_500_res40/", single_parallel_file=False)

    kp = mp.Vector3(fsrc * n).rotate(mp.Vector3(z=1), rot_angle)

    sources_c = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fsrc, fwidth=fsrc/7, is_integrated=True),
            # src=mp.ContinuousSource(fsrc),
            amplitude=1.0,
            center=mp.Vector3(-(50 / k0), 0, 0),
            size=mp.Vector3(y=cell_y),
            direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
            eig_kpoint=kp,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    sim.change_sources(sources_c)
    sim.change_k_point(kp)

    # flux_region = mp.FluxRegion(center=mp.Vector3(5, 0, 0), size=mp.Vector3(0, cell_y, 0))
    # flux = sim.add_flux(fsrc, 0, 1, flux_region)


    ez_freq = sim.add_dft_fields([mp.Ez], fsrc, 0, 1, where=mp.Volume(center=mp.Vector3(50 / k0, 0, 0), size=mp.Vector3(0,cell_y,0)))

    sim.run(until=int(2500/fsrc))

    ez_val = sim.get_dft_array(ez_freq, mp.Ez, 0)
    (x, y, z, w) = sim.get_array_metadata(vol=mp.Volume(center=mp.Vector3(50 / k0, 0, 0), size=mp.Vector3(0,cell_y,0)))

    return ez_val, y


def mode_to_angle(mode, k0, LM):
    """
    Given a waveguide mode number, wavenumber k0, and waveguide width LM,
    compute the propagation angle (in radians) of the mode.
    ky = 2 * mode * pi / LM
    kx = sqrt(k0**2 - ky**2)
    angle = arctan(ky / kx)
    Warn or raise error if mode is evanescent (ky > k0).
    """
    ky = 2 * mode * np.pi / LM
    if abs(ky) > abs(k0):
        raise ValueError("ky is greater than k0; mode is not guided (evanescent).")
    elif abs(ky) == abs(k0):
        raise ValueError("ky equals k0; mode is at cutoff (evanescent).")
    kx_sq = k0**2 - ky**2
    if kx_sq < 0:
        raise ValueError("kx is imaginary; mode is evanescent.")
    kx = np.sqrt(kx_sq)
    angle = np.arctan2(ky, kx)
    return angle

def max_propagating_mode(k0, LM, additional=0):
    """
    Returns the largest integer mode number (>=0) for which the mode is still propagating,
    i.e., ky = 2 * mode * pi / LM <= k0.
    """
    max_mode = int(np.floor(abs(k0 * LM) / (2 * np.pi))) + additional
    return max_mode

def ey_to_kz(Ez: np.ndarray, y: np.ndarray, LM: float) -> tuple[np.ndarray, np.ndarray]:
    Ny = len(Ez)

    # FFT
    dy = y[1] - y[0]
    E_ky = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Ez))) / Ny
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy)) * LM

    return E_ky, ky

if __name__ == "__main__":

    
    args = sys.argv[1:]
    
    do_export = "--export" in args or "-e" in args
    do_run = "--run" in args or "-r" in args
    
    if not do_export and not do_run:
        do_export = do_run = True
    
    
    if do_export:
        export_geometry(0)
        
    if do_run:
        k0 = 2 * np.pi / 0.6
        LM = 500 / k0
        
        # Get permitted angles
        max_mode = max_propagating_mode(k0, LM)
        modes = np.arange(-max_mode, max_mode + 1)
        # angles = [mode_to_angle(mode, k0, LM) for mode in modes]
        
        # Collect field FFTs for each angle
        transmission_matrix = []
        
        limit_to_cutoff = True
        additional = 0  

        for mode in modes:
            print(f"Running simulation for mode: {mode}")
            ez_freq, y = run_sim(rot_angle=mode_to_angle(mode, k0, LM))
            
            # ez_freq already contains the DFT, compute FFT
            E_ky, ky = ey_to_kz(ez_freq, y, LM)
            
            if limit_to_cutoff:
                n, c = max_propagating_mode(k0, LM, additional), len(ky) // 2  # <-- Pass `additional` here
                ky, E_ky = ky[c - n : c + n + 1], E_ky[c - n : c + n + 1]

            # Create the directory if it doesn't exist
            os.makedirs("e_field_save", exist_ok=True)

            # Save the data
            with open(f"e_field_save/free_space_{mode}.pkl", "wb") as fe:
                pickle.dump(E_ky, fe)

            transmission_matrix.append(E_ky)
        
        # Convert to numpy array
        T_matrix = np.array(transmission_matrix).T  
        
        # Save the transmission matrix
        # with open("transmission_matrix_random_500_res_40.pkl", "wb") as f:
        #     pickle.dump(T_matrix, f)
        

        # Save the transmission matrix
        with open("vacuum_test.pkl", "wb") as f:
            pickle.dump(T_matrix, f)
