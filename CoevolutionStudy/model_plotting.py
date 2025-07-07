
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_all_species_trajectories(npz_file: str = 'results_all.npz', output_dir: str = 'plots'):
    """
    Load simulation results from a .npz file and plot trajectories for each species.
    Each species is plotted with individual trajectories in gray and the mean in blue.
    Saves one PDF per species into the specified output directory.

    Parameters:
    - npz_file: Path to the .npz file containing:
        * 'times'    : 1D array of time points (shape: T,)
        * 'species'  : 1D array of species names (shape: S,)
        * 'all_data' : 3D array of trajectories (shape: n_reps x T x S)
    - output_dir: Directory where PDF plots will be saved.
    """
    # Verify file existence
    if not os.path.isfile(npz_file):
        raise FileNotFoundError(f"File '{npz_file}' not found. Please check the path.")

    # Load data
    data = np.load(npz_file, allow_pickle=True)
    times = data['times']            # shape (T,)
    species = data['species']        # shape (S,)
    all_data = data['all_data']      # shape (n_reps, T, S)

    n_reps, n_times, n_species = all_data.shape

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot trajectories for each species
    for idx, sp in enumerate(species):
        plt.figure(figsize=(8, 5))
        # Plot all replicates
        for rep in range(n_reps):
            plt.plot(times, all_data[rep, :, idx], color='gray', alpha=0.1, linewidth=0.5)
        # Plot mean trajectory
        mean_traj = np.mean(all_data[:, :, idx], axis=0)
        plt.plot(times, mean_traj, color='blue', linewidth=2, label='Mean')

        # Annotate
        plt.xlabel('Time')
        plt.ylabel(f'{sp} count')
        plt.title(f'Stochastic Trajectories of {sp}')
        plt.legend()
        plt.tight_layout()

        # Save PDF
        outfile = os.path.join(output_dir, f'{sp}_trajectories.pdf')
        plt.savefig(outfile)
        plt.close()

    print(f"Plots saved for species: {', '.join(species)} in '{output_dir}'")