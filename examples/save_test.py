import meep as mp
import numpy as np
import h5py
import os
# Simulation parameters

def save_file():
resolution = 40
    wavelength = 0.6
    fsrc = 1/wavelength  # f = c/λ (assuming c = 1)
    cell_x = 10
    cell_y = 5
    cell_size = mp.Vector3(cell_x, cell_y, 0)  # 2D simulation (z = 0)

    # PML layers - only on x boundaries
    pml_layers = [mp.PML(thickness=1, direction=mp.X)]

    # Parameters for eigenmode source
    rot_angle = 0  # Set to 0 for standard configuration
    k_point = mp.Vector3(0, 0, 0)  # Bloch wavevector
    default_material = mp.Medium(epsilon=1.0)  # Default material (air/vacuum)

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fsrc, fwidth=0.2, is_integrated=True),
            amplitude=1.0,
            center=mp.Vector3(-2, 0, 0),
            size=mp.Vector3(y=cell_y),
            direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
            eig_kpoint=k_point,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    # Create simulation object
    simx = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        dimensions=2,  # 2D simulation
        boundary_layers=pml_layers,
        eps_averaging=False,
        geometry=[
            mp.Block(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(2, 0.5, mp.inf),
                material=mp.Medium(epsilon=4))],
        sources=sources,
        k_point=k_point,
        default_material=default_material
    )

    # simx.run(until=0)
    simx.init_sim()
    simx.dump("examples/",single_parallel_file=False)

if __name__ == "__main__":
    save_file()