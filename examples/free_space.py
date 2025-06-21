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

def run_sim(rot_angle=0):
    box_size = 2  # size of the box in μm
    box_eps = 4

    resolution = 60/0.6  # pixels/μm
    k0 = 2 * np.pi / 0.6  # wavevector magnitude for wavelength = 0.6 μm
    cell_y = 5
    cell_x = 15 + 6
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
            center=mp.Vector3(-(5), 0, 0),
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
        force_complex_fields=True,
        sources=sources,
        k_point=k_point,
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
    results = run_sim()  # Example rotation angle of 45 degrees
    # plot_sim_results(results)    

    if rank == 0:
        # Strip Meep objects that aren't pickle-safe
        results_to_save = {
            k: v for k, v in results.items() if k not in ['sim', 'flux']
        }

        # Save to a pickle file
        pickle_file = "results_free_space_-30.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results_to_save, f)

        # print(f"Pickled results to: {os.path.abspath(pickle_file)}")

   
