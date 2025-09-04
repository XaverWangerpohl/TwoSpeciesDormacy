#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import tellurium as te
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl




# LaTeX: Times-like math (newtxmath) + Computer Modern text
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman', 'Times']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
# Legend appearance: slightly opaque background
mpl.rcParams['legend.framealpha'] = .8
# PGF export configuration (pdflatex + newtxmath)
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
mpl.rcParams['pgf.preamble'] = r'\usepackage{newtxmath}'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10



width_pt = 390
inches_per_pt = 1.0/72.27
golden_ratio = (5**.5 - 1) / 2  # aesthetic figure height

fig_width = width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_ratio # height in inches

# Allow imports from parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from invasion import compute_nontrivial_slice

# ── Model parameters & initial slicing ──
W_birth, W_death = 0.4, 0.1
Y_birth, Y_death = 0.9, 0.15
U_in, U_out      = 0.000001, 1
X_in, X_out      = 0.1, 0.005
# Compute equilibrium slice
Wss, Yss = compute_nontrivial_slice(W_birth, W_death, Y_birth, Y_death)
W0 = Wss / 2
V0 = Wss - W0
X0 = W0 / (X_out / X_in)
U0 = V0 / (U_out / U_in)
Y0 = Yss
size = 1000  # scaling to integer counts





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
    """Run one Gillespie SSA trajectory; return species counts over time."""
    rr = te.loadAntimonyModel(ANT_MODEL)
    rr.setIntegrator('gillespie')
    rr.resetAll()
    rr.setSeed(seed)
    traj = rr.gillespie(0, t_end, n_points)
    # drop the time column → returns array shape (n_points+1, n_species)
    return traj[:, 1:]

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
    plt.figure(figsize=(fig_width,fig_height))
    colors = {'V':'orange','W':'darkgreen','Y':'darkblue'}
    pct = {}
    # plot individual
    for sp,i in zip(('V','W','Y'), idxs):
        for rep in range(n_reps):
            plt.plot(times, all_data[rep,:,i], color=colors[sp], alpha=0.02, lw=0.5)
    # plot mean & compute extinction
    labels = (r'$\widetilde{W}^a$', r'$W^a$', r'$Y$')
    label_map = dict(zip(('V','W','Y'), labels))
    for sp,i in zip(('V','W','Y'), idxs):
        mean_traj = all_data[:,:,i].mean(axis=0)
        plt.plot(times, mean_traj, color=colors[sp], lw=1, label=label_map.get(sp, sp))
        extinct = np.sum((all_data[:,:,i]==0).any(axis=1))
        pct[sp] = extinct/n_reps*100

    title = ', '.join(f"{label_map.get(sp, sp)}: {pct[sp]:.1f}%" for sp in pct)
    title_text = fr"Trajectories for $K_{{pop}} = {size}$"
    plt.title(title_text)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='upper right'); plt.tight_layout()
    out = os.path.join(output_dir,f'VWY_trajectories_{size}.png')
    plt.savefig(out, dpi=300); plt.close()
    print(f"Saved VWY plot to {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-reps',   type=int,   default=100)
    p.add_argument('--t-end',    type=float, default=5000.0)
    p.add_argument('--n-points', type=int,   default=500)
    p.add_argument('--cpus',     type=int,   default=os.cpu_count())
    p.add_argument('--out',      type=str,   default='results_all.npz')
    args = p.parse_args()

    # Precompute times & species
    rr0 = te.loadAntimonyModel(ANT_MODEL)
    rr0.setIntegrator('gillespie')
    full = rr0.gillespie(0, args.t_end, args.n_points)
    times   = full[:,0]
    species = rr0.getFloatingSpeciesIds()  # ['U','V','W','X','Y']

    # Parallel SSA
    seeds = range(args.n_reps)
    with ProcessPoolExecutor(max_workers=args.cpus) as exe:
        it = exe.map(simulate_once,
                     seeds,
                     [args.t_end]*args.n_reps,
                     [args.n_points]*args.n_reps)
        all_data = np.stack(list(tqdm(it, total=args.n_reps)), axis=0)

    # Save results
    np.savez_compressed(args.out,
                        times=times,
                        species=species,
                        all_data=all_data)
    print(f"Saved {args.n_reps} trajectories to '{args.out}'")

    # Plot combined V/W/Y
    plot_VWY_trajectories(npz_file=args.out, output_dir='plots')

if __name__ == "__main__":
    main()
