import sys
import os
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
import pickle
import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def create_oblique_plane_wave_2d(mode, k0 = 2 * np.pi / 0.6, cell_y = None):
    """
    Create oblique unidirectional plane wave in 2D using both E and H components
    
    Parameters:
    - theta_deg: Oblique angle in degrees from normal
    - frequency: Wave frequency
    """


    def mode_to_angle(n, wavelength, L):
        """
        Given mode index n, wavelength, and slab width L, return the propagation angle in radians.
        Only works for propagating modes (|ky| <= k0).
        """
        k0 = 2 * np.pi / wavelength
        ky = 2 * np.pi * n / L
        if np.abs(ky) >= k0:
            raise ValueError("Mode n={} is evanescent for wavelength={} and L={}".format(n, wavelength, L))
        theta = np.arcsin(ky / k0)
        return theta

    theta = mode_to_angle(mode, (2 * np.pi)/k0, cell_y)

    fsrc = k0 / (2 * np.pi)
    kx = k0 * np.cos(theta)
    ky = k0 * np.sin(theta)
    # Amplitude functions for proper E-H relationships
    def amp_func_ez(p):
        """E field (Ez component) with spatial phase"""
        x, y = p.x, p.y
        phase = kx * x + ky * y
        return np.exp(1j * phase)
    
    def amp_func_hx(p):
        """H field (Hx component) for unidirectional propagation"""
        x, y = p.x, p.y
        phase = kx * x + ky * y
        # For TM mode: Hx = (ky/k) * Ez / Z0
        amplitude = (ky / k0)
        return amplitude * np.exp(1j * phase)
    
    def amp_func_hy(p):
        """H field (Hy component) for unidirectional propagation"""
        x, y = p.x, p.y
        phase = kx * x + ky * y
        # For TM mode: Hy = -(kx/k) * Ez / Z0
        amplitude = -(kx / k0)
        return amplitude * np.exp(1j * phase)
    
    # Create sources with proper E-H relationships
    sources = [
        # E field component
        mp.Source(mp.GaussianSource(fsrc, fwidth=fsrc/7, is_integrated=True),
                  component=mp.Ez,
                  center=mp.Vector3(-(5), 0, 0),
                  size=mp.Vector3(y=cell_y),
                  amp_func=amp_func_ez),
        
        # # # H field components for directionality
        # mp.Source(mp.GaussianSource(fsrc, fwidth=fsrc/7, is_integrated=True),
        #           component=mp.Hx,
        #           center=mp.Vector3(-(5), 0, 0),
        #           size=mp.Vector3(y=cell_y),
        #           amp_func=amp_func_hx),
        
        mp.Source(mp.GaussianSource(fsrc, fwidth=fsrc/7, is_integrated=True),
                  component=mp.Hy,
                  center=mp.Vector3(-(5), 0, 0),
                  size=mp.Vector3(y=cell_y),
                  amp_func=amp_func_hy)
    ]
    
    # Create simulation

    
    return sources

def run_sim(wavelength = 0.6, mesh_resolution = 40, source_amplitude = [1.0]):

    def mode_to_angle(n, wavelength, L):
        """
        Given mode index n, wavelength, and slab width L, return the propagation angle in radians.
        Only works for propagating modes (|ky| <= k0).
        """
        k0 = 2 * np.pi / wavelength
        ky = 2 * np.pi * n / L
        if np.abs(ky) >= k0:
            raise ValueError("Mode n={} is evanescent for wavelength={} and L={}".format(n, wavelength, L))
        theta = np.arcsin(ky / k0)
        return theta

    resolution = mesh_resolution/wavelength  # pixels/μm
    k0 = 2 * np.pi / wavelength  # wavevector magnitude for wavelength = 0.6 μm
    cell_y = 5
    cell_x = 15 + 6
    cell_size = mp.Vector3(cell_x, cell_y, 0)
    pml_layers = [mp.PML(thickness=3, direction=mp.X)]
    fsrc = 1.0 / wavelength  # frequency of planewave (wavelength = 1/fsrc)
    n = 1  # refractive index of homogeneous material
    default_material = mp.Medium(index=n)
    # k_point = mp.Vector3(fsrc * n).rotate(mp.Vector3(z=1), rot_angle)

    # eig_src = [mp.EigenModeSource(
    #     src=mp.GaussianSource(fsrc, fwidth=fsrc/7, is_integrated=True),
    #     # src=mp.ContinuousSource(fsrc),
    #     amplitude=amp[0],
    #     center=mp.Vector3(-(5), 0, 0),
    #     size=mp.Vector3(y=cell_y),
    #     # direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
    #     direction=mp.AUTOMATIC if amp[1] == 0 else mp.NO_DIRECTION,
    #     eig_kpoint= mp.Vector3(fsrc * n).rotate(mp.Vector3(z=1), mode_to_angle(amp[1], wavelength = wavelength, L = cell_y)),
    #     eig_band=1,
    #     eig_parity=mp.EVEN_Y + mp.ODD_Z if amp[1] == 0 else mp.ODD_Z,
    #     eig_match_freq=True,
    # )  for amp in source_amplitude]
    sources = create_oblique_plane_wave_2d(4, k0 = k0, cell_y = cell_y)
    # sources.append(create_oblique_plane_wave_2d(-2, k0 = k0, cell_y = cell_y)[0])


    # pow_frc = eig_src.eig_power(fsrc)

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=pml_layers,
        # force_complex_fields=True,
        sources=sources,
        k_point=mp.Vector3(fsrc * n).rotate(mp.Vector3(z=1), mode_to_angle(4, wavelength, cell_y)),
        # k_point=mp.Vector3(0,fsrc * n,0),
        default_material=default_material
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
        # 'eig_power': pow_frc
    }

def call_test():
    print("done_1")

def plot_sim_results(results):
    # sim = results['sim']
    ez_val = -results['ez_val']
    x = results['x']
    y = results['y']
    cell_x = results['cell_x']
    cell_y = results['cell_y']
    cell_size = results['cell_size']

    print("Flux after the incident plane wave:", results['flux_value'])
    print("Flux after the incident plane wave:", results['flux_freqs'])

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), dpi=100)
    field_types = [np.real, np.imag, np.abs]
    titles = ['Real(Ez)', 'Imag(Ez)', '|Ez|']

    # for ax, func, title in zip(axs, field_types, titles):
    #     sim.plot2D(
    #         fields=mp.Ez,
    #         ax=ax,
    #         field_parameters={'alpha': 0.8, 'cmap': 'RdBu', 'interpolation': 'none', 'post_process': func, 'colorbar': True},
    #         boundary_parameters={'hatch': 'o', 'linewidth': 1.5, 'facecolor': 'y', 'edgecolor': 'b', 'alpha': 0.3}
    #     )
    #     ax.set_title(title)

    

    # Plot subplots for real, imaginary, and absolute values of Ez using x and y coordinates
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))
    ez_components = [np.real(ez_val), np.imag(ez_val), np.abs(ez_val)]
    # (x, y, z, w) = sim.get_array_metadata(vol=mp.Volume(center=mp.Vector3(0,0,0), size=cell_size))

    titles = ['Real(Ez)', 'Imag(Ez)', '|Ez|']
    cmaps = ['RdBu', 'RdBu', 'viridis']

    X, Y = np.meshgrid(x, y, indexing='ij')

    for ax2, comp, title, cmap in zip(axs2, ez_components, titles, cmaps):
        im = ax2.pcolormesh(X, Y, comp, cmap=cmap, shading='auto')
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title(title)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_xlim(-5, 5)
        ax2.set_aspect(1)

    plt.tight_layout()
    plt.savefig("fields_0.png", dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    
    # New plot: Ez components at specified x0 and all y, in a 2x2 grid
    x0 = 5  # x0 value to label
    x0_idx = np.abs(X[:,0] - x0).argmin()  # Find closest index to x0=5
    ez_real_line = np.real(ez_val[x0_idx, :])
    ez_imag_line = np.imag(ez_val[x0_idx, :])
    ez_abs_line = np.abs(ez_val[x0_idx, :])
    ez_phase_line = np.angle(ez_val[x0_idx, :])

    fig3, axs3 = plt.subplots(2, 2, figsize=(8, 6))
    fig3.suptitle(f'Ez Components at x₀ = {x0}', fontsize=16, fontweight='bold')

    # Real(Ez)
    axs3[0, 0].plot(Y[0, :], ez_real_line, color='royalblue')
    axs3[0, 0].set_title('Real(Ez)')
    axs3[0, 0].set_xlabel('y')
    axs3[0, 0].set_ylabel('Ez')
    axs3[0, 0].grid(True, alpha=0.3)
    axs3[0, 0].axvline(0, color='gray', linestyle='--', linewidth=1)
    axs3[0, 0].tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)

    # Imag(Ez)
    axs3[0, 1].plot(Y[0, :], ez_imag_line, color='darkorange')
    axs3[0, 1].set_title('Imag(Ez)')
    axs3[0, 1].set_xlabel('y')
    axs3[0, 1].set_ylabel('Ez')
    axs3[0, 1].grid(True, alpha=0.3)
    axs3[0, 1].axvline(0, color='gray', linestyle='--', linewidth=1)
    axs3[0, 1].tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)

    # |Ez|
    axs3[1, 0].plot(Y[0, :], ez_abs_line, color='seagreen')
    axs3[1, 0].set_title('|Ez|')
    axs3[1, 0].set_xlabel('y')
    axs3[1, 0].set_ylabel('Ez')
    axs3[1, 0].grid(True, alpha=0.3)
    axs3[1, 0].axvline(0, color='gray', linestyle='--', linewidth=1)
    axs3[1, 0].tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)

    # Phase(Ez)
    axs3[1, 1].plot(Y[0, :], ez_phase_line, color='purple')
    axs3[1, 1].set_title('Phase(Ez)')
    axs3[1, 1].set_xlabel('y')
    axs3[1, 1].set_ylabel('Ez')
    axs3[1, 1].grid(True, alpha=0.3)
    axs3[1, 1].axvline(0, color='gray', linestyle='--', linewidth=1)
    axs3[1, 1].tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)

    for ax in axs3.flat:
        ax.set_xlim(Y[0, 0], Y[0, -1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("fields.png", dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    print("done_0")
    

if __name__ == "__main__":
    for (ind, amp) in enumerate( [[(1, -2)]]):  # Example source amplitudes
        for mesh_resolution in [60]:
            for wavelength in [0.6]:
                for angle in [0]:
                    results = run_sim(wavelength = wavelength , mesh_resolution=mesh_resolution, source_amplitude = amp )  # Example rotation angle of 45 degrees
                    # plot_sim_results(results)    

                    if rank == 0:
                        # Strip Meep objects that aren't pickle-safe
                        results_to_save = {
                            k: v for k, v in results.items() if k not in ['sim', 'flux']
                        }

                        # Save to a pickle file
                        pickle_file = f"free_space_source_test.pkl"
                        with open(pickle_file, 'wb') as f:
                            pickle.dump(results_to_save, f)

        # print(f"Pickled results to: {os.path.abspath(pickle_file)}")

   
