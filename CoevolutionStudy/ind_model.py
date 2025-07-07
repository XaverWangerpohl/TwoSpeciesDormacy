#!/usr/bin/env python3
import tellurium as te
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import os
import sys

# Determine the directory one level up
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Insert it at the front of sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from invasion import compute_nontrivial_slice


W_birth = 0.4
W_death = 0.1
Y_birth = 0.9
Y_death = 0.15

X_in = 0.1
X_out = 0.1
Z_in = 0.5
Z_out = 0.25

U_out = 0.9
U_in = 0.9
X_out = 0.06
X_in = X_out

W,Y = compute_nontrivial_slice(W_birth, W_death, Y_birth, Y_death)
W0 = W/2
X0 = W0 / (X_out / X_in)
V0 = W - W0
U0 = V0 / (U_out / U_in)
Y0 = Y
Z0 = Y0/ (Z_out /Z_in)

size = 100  #  max number of individuals per species

# ── Antimony model ──
ANT_MODEL = f"""
model birth_death()
  U = {int(U0*size)}
  V = {int(V0*size)}
  W = {int(W0*size)}
  X = {int(X0*size)}
  Y = {int(Y0*size)}

  size     = {size}
  W_birth  = {W_birth};   W_death  = {W_death}
  Y_birth  = {Y_birth};   Y_death  = {Y_death}
  U_in     = {U_in};   U_out    = {U_out}
  X_in     = {X_in};   X_out    = {X_out}

  W_symbiosis:    => W; W_birth/size * W * Y
  V_symbiosis:    => V; W_birth/size * V * Y
  Y_symbiosis:    => Y; Y_birth/size * (W + V) * Y

  V_death_reaction: V => ; W_death * V
  W_death_reaction: W => ; W_death * W
  Y_death_reaction: Y => ; Y_death * Y

  U_quiescence_in:   V => U; U_in  * V
  X_quiescence_in:   W => X; X_in  * W 
  U_quiescence_out:  U => V; U_out * U
  X_quiescence_out:  X => W; X_out * X

  V_compete: V => ; W_birth/(size^2) * V * (W + V) * Y
  W_compete: W => ; W_birth/(size^2) * W * (V + W) * Y
  Y_compete: Y => ; Y_birth/(size^2) * Y * (V + W) * Y
end
"""

def simulate_once(seed: int, t_end: float, n_points: int) -> np.ndarray:
    """
    Simulate one Gillespie trajectory of all species.
    Returns an array shape (n_points+1, n_species).
    """
    rr = te.loadAntimonyModel(ANT_MODEL)
    rr.setIntegrator('gillespie')
    rr.resetAll()
    rr.setSeed(seed)
    traj = rr.gillespie(0, t_end, n_points)
    # Drop the time column; return only species columns [U,V,W,X,Y]
    return traj[:, 1:]

def plot_all_species_trajectories(npz_file: str = 'results_all.npz',
                                  output_dir: str = 'plots'):
    """
    Load simulation results from a .npz file and plot trajectories for each species.
    Each plot shows:
      – all individual replicate trajectories in light gray
      – the mean trajectory in blue
      – the percentage of replicates that reached zero (extinction) in the title.

    Parameters:
    - npz_file:   Path to the .npz file containing 'times', 'species', and 'all_data'.
    - output_dir: Directory where PDF plots will be saved.
    """
    # 1) Verify file
    if not os.path.isfile(npz_file):
        raise FileNotFoundError(f"File '{npz_file}' not found.")

    # 2) Load data arrays
    data      = np.load(npz_file, allow_pickle=True)
    times     = data['times']      # shape: (T,)
    species   = data['species']    # shape: (S,)
    all_data  = data['all_data']   # shape: (n_reps, T, S)

    n_reps, _, n_species = all_data.shape

    # 3) Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # 4) Loop over species and plot
    for idx, sp in enumerate(species):
        plt.figure(figsize=(8, 5))

        print(all_data[:, :, idx].min())
        
        # 4a) individual replicates
        for rep in range(n_reps):
            plt.plot(times, all_data[rep, :, idx],
                     color='gray', alpha=0.1, linewidth=0.5)
            

        # 4b) mean trajectory
        mean_traj = all_data[:, :, idx].mean(axis=0)
        plt.plot(times, mean_traj,
                 color='blue', linewidth=2, label='Mean')

        # 4c) extinction percentage
        extinct_count = np.sum((all_data[:, :, idx] == 0).any(axis=1))
        pct_extinct   = extinct_count / n_reps * 100

        # 5) Annotate plot
        plt.title(f"Stochastic Trajectories of {sp} — {pct_extinct:.1f}% extinct")
        plt.xlabel('Time')
        plt.ylabel(f'{sp} count')
        plt.legend()
        plt.tight_layout()

        # 6) Save as PDF
        out_path = os.path.join(output_dir, f'{sp}_trajectories.pdf')
        plt.savefig(out_path)
        plt.close()

    print(f"Saved trajectory plots (with extinction %) for species: {', '.join(species)}\n"
          f"Into directory: '{output_dir}/'")

def plot_VWY_trajectories(npz_file: str = 'results_all.npz',
                          output_dir: str = 'plots'):
    """
    Load results_all.npz, extract V/W/Y trajectories, compute extinction %,
    and save a combined PDF 'VWY_trajectories.pdf'.
    """
    if not os.path.isfile(npz_file):
        raise FileNotFoundError(f"Missing {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    times   = data['times']       # (T,)
    species = list(data['species'])  # ['U','V','W','X','Y']
    all_data= data['all_data']    # (n_reps, T, 5)
    
    # indices for V, W, Y
    idxs = [species.index(s) for s in ('V','W','Y')]
    n_reps = all_data.shape[0]

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10,6))
    colors = {'Y':'darkblue','W':'darkgreen','V':'orange'}
    pct = {}
    # plot individual
    for sp,i in zip(('V','W','Y'), idxs):
        for rep in range(n_reps):
            plt.plot(times, all_data[rep,:,i], color=colors[sp], alpha=0.005, lw=0.1)
    # plot mean & compute extinction
    for sp,i in zip(('V','W','Y'), idxs):
        mean_traj = all_data[:,:,i].mean(axis=0)
        plt.plot(times, mean_traj, color=colors[sp], lw=2, label=f'{sp} mean')
        extinct = np.sum((all_data[:,:,i]==0).any(axis=1))
        pct[sp] = extinct/n_reps*100

    title = ', '.join(f"{sp}: {pct[sp]:.1f}%" for sp in pct)
    plt.title(f"V, W, Y trajectories ({title} extinct)")
    plt.xlabel('Time'); plt.ylabel('Count')
    plt.legend(); plt.tight_layout()
    out = os.path.join(output_dir,'VWY_trajectories.pdf')
    plt.savefig(out); plt.close()
    print(f"Saved VWY plot to {out}")


def main():
    parser = argparse.ArgumentParser(description="Parallel Gillespie sims for all species")
    parser.add_argument('--n-reps',   type=int,   default=1000, help="Number of trajectories")
    parser.add_argument('--t-end',    type=float, default=10000.0, help="End time")
    parser.add_argument('--n-points', type=int,   default=500,   help="Output intervals")
    parser.add_argument('--cpus',     type=int,   default=os.cpu_count(), help="Worker count")
    parser.add_argument('--out',      type=str,   default='results_all.npz', help="Output file")
    parser.add_argument('--filter-y-extinct', action='store_true',
                        help='Remove trajectories where Y goes extinct', default=True)
    args = parser.parse_args()

    # Precompute times and species list
    rr0 = te.loadAntimonyModel(ANT_MODEL)
    rr0.setIntegrator('gillespie')
    full = rr0.gillespie(0, args.t_end, args.n_points)
    times   = full[:, 0]
    species = rr0.getFloatingSpeciesIds()  # ['U','V','W','X','Y']

    # Parallel execution with tqdm
    seeds = range(args.n_reps)
    with ProcessPoolExecutor(max_workers=args.cpus) as exe:
        results = exe.map(
            simulate_once,
            seeds,
            [args.t_end]*args.n_reps,
            [args.n_points]*args.n_reps
        )
        all_data = np.stack(
            list(tqdm(results,
                      total=args.n_reps,
                      desc="Simulating trajectories")),
            axis=0
        )
    # all_data.shape == (n_reps, n_points+1, n_species)

    # Optionally filter out Y-extinct runs
    if args.filter_y_extinct:
        y_idx = species.index('Y')
        mask = ~np.any(all_data[:,:,y_idx] == 0, axis=1)
        removed = args.n_reps - mask.sum()
        all_data = all_data[mask]
        print(f"Filtered out {removed} trajectories where Y went extinct.")



    # Save everything
    np.savez_compressed(args.out,
                        times=times,
                        species=species,
                        all_data=all_data)
    print(f"Saved {args.n_reps} trajectories for species {species} into '{args.out}'")
    plot_all_species_trajectories(npz_file='results_all.npz', output_dir='plots')
    plot_VWY_trajectories(npz_file='results_all.npz', output_dir='plots')

if __name__ == "__main__":
    main()
    