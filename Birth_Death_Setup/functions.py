import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm, trange
import re
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor




''''
simulate_segment simulates the dynamics of the model from a given initial state

run_invasion simulates multiple segments iteratively, applying perturbations to the state variables in order to gauge the invasibility of the system.

local_invasability captures weather small changes of the parameters lead to invasion or not

global_invasibility captures which parameters lead to invasion given a fixed (test_point) inhabitant
'''


def compute_equilibrium(W_birth, W_death, Y_birth, Y_death):
    """
    Compute the positive nontrivial equilibrium (W_eq, Y_eq) by solving:
      Q1 = W_death / W_birth,   Q2 = Y_death / Y_birth
      W_eq = ½ [ (1 - Q1 + Q2) + sqrt((1 - Q1 + Q2)^2 - 4·Q2 ) ]
      Y_eq = ½ [ (1 - Q2 + Q1) + sqrt((1 - Q2 + Q1)^2 - 4·Q1 ) ]
    Returns (W_eq, Y_eq) if both lie in (0,1); otherwise (None, None).
    """
    Q1 = W_death / W_birth
    Q2 = Y_death / Y_birth

    disc_W = (1 - Q1 + Q2)**2 - 4 * Q2
    if disc_W < 0:
        return None, None
    sqrt_disc_W = np.sqrt(disc_W)
    W_equil = 0.5 * ((1 - Q1 + Q2) + sqrt_disc_W)
    if not (0.0 < W_equil < 1.0):
        return None, None

    disc_Y = (1 - Q2 + Q1)**2 - 4 * Q1
    if disc_Y < 0:
        return None, None
    sqrt_disc_Y = np.sqrt(disc_Y)
    Y_equil = 0.5 * ((1 - Q2 + Q1) + sqrt_disc_Y)
    if not (0.0 < Y_equil < 1.0):
        return None, None

    return W_equil, Y_equil

def simulate_segment(U0, V0, W0, X0, Y0, Z0,
                      W_birth, Y_birth,
                      W_death, Y_death,
                      X_in, X_out,
                      U_in, U_out, 
                      Z_in, Z_out, 
                      duration, dt,
                      use_X=True, use_Z=False,
                      tol=1e-7,
                      stop_at_eq=True):
    """
    Integrate from t=0 to t=duration with initial conditions
      V(0)=V0, W(0)=W0, Y(0)=Y0, X(0)=X0, Z(0)=Z0.
    If stop_at_eq=True, stops early when all |dV|,|dW|,|dY| (and |dX| if use_X, |dZ| if use_Z)
    fall below tol. Otherwise, always runs full duration.
    Returns:
      t_array,
      V_array, W_array, Y_array,
      X_array (unscaled), Z_array (unscaled),
      X_plot = X_array * X_scaler, Z_plot = Z_array * Z_scaler.
    """


    N = int(np.ceil(duration / dt)) + 1
    t = np.linspace(0.0, duration, N)

    U = np.zeros(N)
    V = np.zeros(N)
    W = np.zeros(N)
    X = np.zeros(N)
    Y = np.zeros(N)
    Z = np.zeros(N)

    U[0] = U0
    V[0] = V0
    W[0] = W0
    X[0] = X0
    Y[0] = Y0
    Z[0] = Z0

    final_index = N - 1
    for i in range(1, N):
        Vi = V[i-1]
        Wi = W[i-1]
        Yi = Y[i-1]
        Xi = X[i-1]
        Zi = Z[i-1]
        Ui = U[i-1]

        # dV/dt, dW/dt
        dV = W_birth * (1 - Wi - Vi) * Vi * Yi - W_death * Vi
        dW = W_birth * (1 - Wi - Vi) * Wi * Yi - W_death * Wi

        # dY/dt
        dY = Y_birth * (1 - Yi) * Yi * (Vi + Wi) - Y_death * Yi

        # X-coupling
        if use_X:
            dW += X_out * Xi - X_in * Wi
        # U-coupling
            dV += U_out * Ui - U_in * Vi
        # Z-coupling
        if use_Z:
            dY += Z_out * Zi - Z_in * Yi

        # seed bank dynamics 
        dX = - X_out * Xi + X_in * Wi
        dU = - U_out * Ui + U_in * Vi
        dZ = - Z_out * Zi + Z_in * Yi


        # If stop_at_eq=True, check for equilibrium
        if stop_at_eq:
            cond = (abs(dV) < tol and abs(dW) < tol and abs(dY) < tol)
            if use_X:
                cond &= abs(dX) < tol
            if use_Z:
                cond &= abs(dZ) < tol
            if cond:
                final_index = i - 1
                break

        # Euler update
        V[i] = Vi + dt * dV
        W[i] = Wi + dt * dW
        Y[i] = Yi + dt * dY
        X[i] = Xi + dt * dX
        Z[i] = Zi + dt * dZ
        U[i] = Ui + dt * dU

        # Enforce nonnegativity (just to be sure to not have rounding errors)
        V[i] = max(V[i], 0.0)
        W[i] = max(W[i], 0.0)
        Y[i] = max(Y[i], 0.0)
        X[i] = max(X[i], 0.0)
        Z[i] = max(Z[i], 0.0)
        U[i] = max(U[i], 0.0)

    

    # Truncate arrays if we stopped early
    t = t[: final_index + 1]
    V = V[: final_index + 1]
    W = W[: final_index + 1]
    Y = Y[: final_index + 1]
    X = X[: final_index + 1]
    Z = Z[: final_index + 1]
    U = U[: final_index + 1]



    # Compute scalers for plotting (so the seedbank and active populations are on the same scale)
        
    X_scaler = X_out / X_in 
    Z_scaler = Z_out / Z_in 
    U_scaler = U_out / U_in

    X_plot = X * X_scaler
    Z_plot = Z * Z_scaler
    U_plot = U * U_scaler



    return t, U, V, W, X, Y, Z, X_plot, Z_plot, U_plot

def plot_segment(U0, V0, W0, X0, Y0, Z0,
              W_birth, Y_birth,
              W_death, Y_death,
              X_in, X_out,
              U_in, U_out,
              Z_in, Z_out,
              Time=200.0, dt=0.1,
              use_X=True, use_Z=False,
              severity=0.5,
              perturb_W=False, perturb_Y=True,
              perturb_time=20.0,
              tol=1e-7, plot_Y=False):
    """
    Build a time-series plot for a fixed W0 using perturbation multiplier = (1 - severity).
    1) Compute (W_eq, Y_eq).
    2) Verify W0 ∈ [0, W_eq], then set V0 = W_eq - W0, X0, Z0.
    3) Stage A: simulate from t=0 → perturb_time (no stopping).
    4) At t=0 apply perturbation: V_mid=(1-severity)*V_eq_pre, etc.
    5) Stage B: simulate from t=0 → Time (no stopping).
    6) Concatenate and plot V, W, Y, X, Z over t ∈ [-perturb_time, Time].
    Returns results dict.
    """

    # Stage A, part to ensure, that we start at equilibrium
    t_pre, U_pre, V_pre, W_pre, X_pre, Y_pre, Z_pre, X_pre_plot, Z_pre_plot, U_pre_plot = simulate_segment(
        V0=V0, W0=W0, Y0=Y0, X0=X0, Z0=Z0, U0=U0,
        W_birth=W_birth, Y_birth=Y_birth,
        W_death=W_death, Y_death=Y_death,
        X_in=X_in, Z_in=Z_in,
        X_out=X_out, Z_out=Z_out,
        U_in=U_in, U_out=U_out,
        duration=perturb_time, dt=dt,
        use_X=use_X, use_Z=use_Z,
        tol=tol,
        stop_at_eq=False
    )

    # make the time coherent
    t_pre_shifted = t_pre - perturb_time

    V_eq_pre = V_pre[-1]
    W_eq_pre = W_pre[-1]
    Y_eq_pre = Y_pre[-1]
    X_eq_pre = X_pre_plot[-1]
    U_eq_pre = U_pre_plot[-1]
    Z_eq_pre = Z_pre_plot[-1]

    # (4) apply perturbation multiplier = (1 - severity)
    V_mid = ((1 - severity) * V_eq_pre) if perturb_W else V_eq_pre
    W_mid = ((1 - severity) * W_eq_pre) if perturb_W else W_eq_pre
    Y_mid = ((1 - severity) * Y_eq_pre) if perturb_Y else Y_eq_pre

    # Stage B
    t_post, U, V_post, W_post, X, Y_post, Z, X_post_plot, Z_post_plot, U_post_plot = simulate_segment(
        V0=V_mid, W0=W_mid, Y0=Y_mid, X0=X_pre[-1], Z0=Z_pre[-1], U0=U_pre[-1],
        W_birth=W_birth, Y_birth=Y_birth,
        W_death=W_death, Y_death=Y_death,
        X_in=X_in, Z_in=Z_in,
        X_out=X_out, Z_out=Z_out,
        U_in=U_in, U_out=U_out,
        duration=Time, dt=dt,
        use_X=use_X, use_Z=use_Z,
        tol=tol,
        stop_at_eq=False
    )

    t_full = np.concatenate((t_pre_shifted, t_post))
    U_full = np.concatenate((U_pre_plot, U_post_plot))
    V_full = np.concatenate((V_pre, V_post))
    W_full = np.concatenate((W_pre, W_post))
    Y_full = np.concatenate((Y_pre, Y_post))
    X_full = np.concatenate((X_pre_plot, X_post_plot))


    W_final = W_full[-1]

    delta_W_test = W_final - W_eq_pre

    X_scaler = X_out/X_in
    U_scaler = U_out/U_in


    # Time-series plot
    plt.figure(figsize=(8, 5))
    if use_X:
        plt.plot(t_full, X_full, label=f'{X_scaler:.1f} 'r'$X(t)$ (seedbank of W)', color='lime', linewidth=1.5)
        plt.plot(t_full, U_full, label=f'{U_scaler:.1f} 'r'$U(t)$ (seedbank of V)', color='gold', linewidth=1.5)
    if plot_Y:
        plt.plot(t_full, Y_full, label=r'$Y(t)$', color='darkblue', linewidth=1.5)
    plt.plot(t_full, V_full, label=r'$V(t)$', color='orange', linewidth=1.5)
    plt.plot(t_full, W_full, label=r'$W(t)$', color='darkgreen', linewidth=1.5)


    plt.axvline(x=0.0, color='gray', linestyle='--', lw=1)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.title(
        rf'Modeling of a {(severity*100):.0f}% Extinction Event on $Y$ (latent)' + '\n'
        + rf'$\Delta W = {delta_W_test:.4f}$',
        fontsize=14
    )
    plt.legend(loc='best', fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/Users/xaverwangerpohl/Documents/GitHub/master-code/SegmentPlots/segment.pdf', format='pdf')
    plt.show()
    print(delta_W_test)
    

    return 

def run_invasion(V0, W0, Y0,
               W_birth, Y_birth,
               W_death, Y_death,
               X_in, X_out,
               U_in, U_out,
               Z_in, Z_out,
               extinction_rate, dt,
               use_X, use_Z,
               severity,
               cycles=10000,
               perturb_W=False,
               perturb_Y=True,
               plot=False,
               stop=None,
               break_threshold=0.01, show_Y=False):
    """
    Run 'cycles' successive calls to simulate_segment, each time:
      1) simulate_segment(...) → (t, V_arr, W_arr, Y_arr, X_arr, Z_arr)
      2) record final V, W, Y
      3) if perturb_W: set W0_next = (1-severity)*W_final and
                         V0_next = (1-severity)*V_final
      4) if perturb_Y: set Y0_next = (1-severity)*Y_final
      5) X0_next = X_final, Z0_next = Z_final
    After all cycles, plot cycle index vs final W, V, and Y.
    Returns lists of final values [V_finals, W_finals, Y_finals].
    """

    #get equilibrium seedbank sizes

    X0 = W0 / (X_out / X_in)
    U0 = V0 / (U_out / U_in)
    Z0 = Y0 / (Z_out /Z_in)

    U_current = U0
    V_current = V0
    W_current = W0
    X_current = X0
    Y_current = Y0
    Z_current = Z0
    
    V_finals = []
    W_finals = []
    Y_finals = []

    for n in range(1, cycles+1):

        # 1) simulate one segment
        t, U, V, W, X, Y, Z, X_plot, Z_plot, U_plot = simulate_segment(
            V0=V_current, W0=W_current, Y0=Y_current, X0=X_current, Z0=Z_current, U0=U_current,
            W_birth=W_birth, Y_birth=Y_birth, W_death=W_death, Y_death=Y_death,
            X_in=X_in, Z_in=Z_in, X_out=X_out, Z_out=Z_out, U_in=U_in, U_out=U_out,
            duration=extinction_rate, dt=dt,
            use_X=use_X, use_Z=use_Z,
            tol=1e-7,
            stop_at_eq=True
        )

        # 2) record final values of the segment
        V_final = V[-1]
        W_final = W[-1]
        Y_final = Y[-1]

        V_finals.append(V_final)
        W_finals.append(W_final)
        Y_finals.append(Y_final)

        # burn in, if the extinction rate is faster then the return to equilibrium, the population values drop uniformly
        if n == 50:
            W0 = W_final

        # break threshold to accelerate the computation (if we grow by some amount, we expect it to grow fully)
        
        if (abs(W_final - W0) > break_threshold) and n > 50:
            break

        # 3) perturb for next cycle (extinction event)
        if perturb_W:
            V_current = (1 - severity) * V_final
            W_current = (1 - severity) * W_final
        else:
            V_current = V_final
            W_current = W_final

        if perturb_Y:
            Y_current = (1 - severity) * Y_final
        else:
            Y_current = Y_final

        # 4) carry over X, Z unchanged
        X_current = X[-1]
        Z_current = Z[-1]
        U_current = U[-1]


    if plot:
        # plot all three on one figure
        cycles_idx = np.arange(1, n+1)
        plt.figure(figsize=(8, 5))
        plt.plot(cycles_idx, W_finals, label='W', color='darkgreen')
        plt.plot(cycles_idx, V_finals, label='V', color='orange')
        if show_Y:
            plt.plot(cycles_idx, Y_finals, label='Y', color='darkblue')
        plt.xlabel('Cycle', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        titlestr = f'V, W, Y after each cycle\n(severity={severity}' 
        titlestr += ', W Perturbed, ' if perturb_W else ''
        titlestr += ', Y perturbed, ' if perturb_Y else ''
        titlestr += 'U[in,out]: ({:.2f}, {:.2f}), X:({:.2f}, {:.2f}))'.format(
            U_in, U_out, X_in, X_out)
        plt.title(titlestr, fontsize=14)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Saving the plot
        
        folder = "run_invasion"
        os.makedirs(folder, exist_ok=True)
        base = "run_invasion"
        pattern = os.path.join(folder, base + "*.pdf")
        existing = glob.glob(pattern)
        if not existing:
            pdf_name = base + ".pdf"
        else:
            taken = set(int(os.path.basename(p).replace(base,"").replace(".pdf","") or 0)
                        for p in existing if os.path.basename(p).replace(base,"").replace(".pdf","").isdigit() or p.endswith(base+".pdf"))
            k=0
            while k in taken:
                k+=1
            pdf_name=f"{base}{k}.pdf"
        path = os.path.join(folder, pdf_name)
        plt.savefig(path)
        print(f"Saved run_invasion plot to {path}")

        plt.show()

    return W_final - W0

def global_invasability(V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size,
    U_in=0.1, U_out=0.1,
    X_in_range = 0.1,
    X_out_range = 0.1,
    perturb_W=False, perturb_Y=True, break_threshold=0.01

):
    """
    Generate an invasion plot over X_in (x-axis) and X_out (y-axis).
    Uses the middle point of the grid as the resident

    """

    X_in_vals = np.linspace(U_in-X_in_range, U_in + X_in_range, grid_size)
    X_out_vals = np.linspace(U_out-X_out_range, U_out + X_out_range, grid_size)
    deltaW_matrix = np.zeros((grid_size, grid_size))

    for i, X_in in enumerate(tqdm(X_in_vals, desc="Scanning X_in")):
        for j, X_out in enumerate(X_out_vals):
            # Mirror diagonal
            if (j == i) and (i == grid_size // 2):
                deltaW_matrix[j, i] = 0.0
                continue

            # Compute ΔW via run_invasion
            deltaW = run_invasion(
                        V0=V0, W0=W0, Y0=Y0,
                        W_birth=W_birth, Y_birth=Y_birth,
                        W_death=W_death, Y_death=Y_death,
                        X_in=X_in, X_out=X_out,
                        U_in=U_in,U_out=U_out,
                        Z_in=Z_in, Z_out=Z_out,
                        extinction_rate=extinction_rate, dt=dt,
                        use_X=use_X, use_Z=use_Z,
                        cycles=cycles, severity=severity,
                        perturb_W=perturb_W, perturb_Y=perturb_Y,
                        plot=False, break_threshold=break_threshold)
            
            deltaW_matrix[j, i] = deltaW

    # make grid edges
    X_edges = np.linspace(U_in - X_in_range, U_in + X_in_range, grid_size + 1)
    Y_edges = np.linspace(U_out - X_out_range, U_out + X_out_range, grid_size + 1)


        # Prepare output folder
    folder = 'invasion_plots'
    os.makedirs(folder, exist_ok=True)

    # —— Binary plot —— #

    # Binary plot: determine next filename index
    bin_pattern = os.path.join(folder, 'pip*.pdf')
    existing_bin = glob.glob(bin_pattern)
    bin_idxs = []
    for p in existing_bin:
        m = re.match(r'.*pip(\d*)\.pdf$', os.path.basename(p))
        if m:
            idx = int(m.group(1)) if m.group(1) else 0
            bin_idxs.append(idx)
    next_bin = max(bin_idxs) + 1 if bin_idxs else 0
    bin_fname = f'pip{next_bin}.pdf'


    # Build category matrix (0,1,2)…
    category = np.zeros_like(deltaW_matrix, dtype=int)
    category[deltaW_matrix <  0] = 0
    category[deltaW_matrix == 0] = 1
    category[deltaW_matrix >  0] = 2
    cmap = ListedColormap(['white', 'black', 'gray'])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Again, use pcolormesh for perfect alignment:
    mesh2 = ax.pcolormesh(
        X_edges, Y_edges, category,
        shading='flat',
        cmap=cmap,
        vmin=0, vmax=2
    )

    # Gridlines:
    ax.set_xticks(X_edges, minor=True)
    ax.set_yticks(Y_edges, minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)

    # Major ticks centered:
    ax.set_xticks((X_edges[:-1] + X_edges[1:]) / 2)
    ax.set_yticks((Y_edges[:-1] + Y_edges[1:]) / 2)
    ax.set_xticklabels([f"{x:.2f}" for x in (X_edges[:-1] + X_edges[1:]) / 2], rotation=90)
    ax.set_yticklabels([f"{y:.2f}" for y in (Y_edges[:-1] + Y_edges[1:]) / 2])

    ax.set_xlabel('Mutant Seedbank in Rate')
    ax.set_ylabel('Mutant Seedbank out Rate')
    ax.set_title('Invasion (gray) and Extinction (white) of Mutant')

    plt.tight_layout()
    fig.savefig(os.path.join(folder, bin_fname), format='pdf')
    plt.show()

    return X_in_vals, X_out_vals, deltaW_matrix

def local_invasibility(V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size=5,
    U_in_min=0.01, U_in_max=0.4,
    U_out_min=0.01, U_out_max=0.4,
    folder='local_invasibility',
    break_threshold=0.01
):
    """
    For each interior gridpoint (i,j) on the plane U_in, U_out:
      • Evaluate run_invasion at its 4 nearest neighbors
      • Compute mean of their np.sign(deltaW)
      • Plot 
    """

    # 1) Prepare output directory
    os.makedirs(folder, exist_ok=True)

    # 2) Build the 2d grid
    U_in_vals = np.linspace(U_in_min, U_in_max, grid_size)
    U_out_vals = np.linspace(U_out_min, U_out_max, grid_size)

    # 3) Initialize the score matrix
    score = np.zeros((grid_size, grid_size))

    # 4) Offsets for the neighbors
    neighbor_offsets = [         (-1,0),       
                        ( 0,-1),         ( 0,1),
                                 ( 1,0),       ]
    
    # deltas is the dictionary, which stores all the considered combination of invasion parameters and weather they succeed
    deltas = {}
    argsdict = {}

    # 5) Loop over interior points (border does not have all the neighbours)
    for i in tqdm(range(1, grid_size-1), desc="Computing local invasibility"):
        for j in range(1, grid_size-1):
            # invasions counts, how many neighbours can invade
            invasions = 0
            # loop over the neighbours
            for di, dj in neighbor_offsets:
                X_in = U_in_vals[i+di]
                X_out = U_out_vals[j+dj]
                # check weather the invasion has already been computed (antisymmetric if W invades V, V is invaded by W)
                
                if ((i+di,j+dj), (i,j)) in deltas.keys():
                    if deltas[((i+di,j+dj), (i,j))] == -1:
                        invasions += 1
                else:
                
                    args = {
                    "V0": V0, "W0": W0, "Y0": Y0,
                    "W_birth": W_birth, "Y_birth": Y_birth,
                    "W_death": W_death, "Y_death": Y_death,
                    "X_in": X_in, "X_out": X_out,
                    "U_in": U_in_vals[i], "U_out": U_out_vals[j],
                    "Z_in": Z_in, "Z_out": Z_out,
                    "extinction_rate": extinction_rate, "dt": dt,
                    "use_X": use_X, "use_Z": use_Z,
                    "cycles": cycles, "severity": severity,
                    "perturb_W": False, "perturb_Y": True,
                    "plot": False, "break_threshold": break_threshold
                    }
                    argsdict[((i,j), (i+di,j+dj))] = args
                
                    deltaW = run_invasion(**args
                    )
                    # save findings to deltas
                    if deltaW> 0: # W can invade V  
                        invasions += 1
                        deltas[((i,j), (i+di,j+dj))] = 1
                        deltas[((i+di,j+dj), (i,j))] = -1
                    else:
                        deltas[((i,j), (i+di,j+dj))] = -1
                        deltas[((i+di,j+dj), (i,j))] = 1

                # store the mean sign ∈ [–1,+1]
            score[i, j] = invasions

    # 6) Plot the result 
    # mask out the border
    mask = np.zeros_like(score, dtype=bool)
    mask[ 0, :] = True   # top row
    mask[-1, :] = True   # bottom row
    mask[:,  0] = True   # left column
    mask[:, -1] = True   # right column
    score_masked = np.ma.array(score, mask=mask)

    cmap = plt.get_cmap('Greens', 5)
    cmap.set_bad(color='white') 

    # 2. Create a norm that bins values 0–4 into 5 discrete intervals.
    bounds = np.arange(-0.5, 5.5, 1)     
    norm   = BoundaryNorm(bounds, cmap.N)
    # 6) Plot the result 
    plt.figure(figsize=(9,8))
    im = plt.imshow(
        score_masked.T,
        origin='lower',
        extent=[U_in_min, U_in_max, U_out_min, U_out_max],
        aspect='auto',
        cmap=cmap,
        norm=norm,
        zorder=0
    )

    plt.xlabel('U_in')
    plt.ylabel('U_out')
    plt.title('Local invasibility (Number of invading neighbors)')

    # add gridlines that align with the coloured squares
    # compute the edges of the cells:
    x_edges = np.linspace(U_in_min, U_in_max, grid_size + 1)
    y_edges = np.linspace(U_out_min, U_out_max, grid_size + 1)
    # draw vertical grid lines
    for x in x_edges:
        plt.axvline(x, color='black', linewidth=1, linestyle='-',
                    zorder=1)
    # draw horizontal grid lines
    for y in y_edges:
        plt.axhline(y, color='black', linewidth=1, linestyle='-',
                    zorder=1)

    # now add the colorbar
    cbar = plt.colorbar(im, ticks=np.arange(0, 5, 1), boundaries=bounds)
    cbar.set_label('Count of invading neighbors')

    plt.tight_layout()

    # save a new file without overwriting
    existing = sorted([p for p in os.listdir(folder) 
                       if p.startswith('local_inv') and p.endswith('.pdf')])
    idx = int(re.search(r'\d+', existing[-1]).group())+1 if existing else 0
    fname = os.path.join(folder, f'local_inv{idx}.pdf')
    plt.savefig(fname)
    plt.show()
    return U_in_vals, U_out_vals, score, deltas, argsdict

def test_local_invasion(U_in_vals, U_out_vals, deltas, test_point,
                  
    V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
):
    X_vals = [
    (U_in_vals[k], U_out_vals[l])
    for ((i, j), (k, l)), v in deltas.items()
    if (i == test_point[0] and j == test_point[1])
    ]
    
    U_vals = [
    U_in_vals[test_point[0]], U_out_vals[test_point[1]]]

    U_in, U_out = U_vals

    for Xs in X_vals:
        X_in, X_out = Xs
        
        run_invasion(
            V0, W0, Y0,
            W_birth, Y_birth,
            W_death, Y_death,
            X_in, X_out,
            U_in, U_out,
            Z_in, Z_out,
            extinction_rate, dt,
            use_X, use_Z,
            severity,
            cycles=cycles,
            perturb_W=False,
            perturb_Y=True,
            plot=True
        )
  
    return X_vals, U_vals

def reconstruct_and_flow_map(
    signs,
    x_vals,
    y_vals,
    folder='surface_flow',
    arrow_scale=20,
    invert=False
):
    """
    Reconstruct the scalar field f from sign data on a custom grid,
    compute its gradient, plot a quiver (flow) map of the gradient vectors
    with adjustable arrow length, and overlay markers at every tested data point.

    Parameters
    ----------
    signs : dict
        Mapping edge tuples ((i,j),(ii,jj)) to sign values (+1 or -1).
        Indices i in [0, len(x_vals)), j in [0, len(y_vals)).
    x_vals : Sequence[float]
        1D array of x-coordinates (length Nx).
    y_vals : Sequence[float]
        1D array of y-coordinates (length Ny).
    folder : str
        Directory to save the PDF of the flow map.
    arrow_scale : float
        Controls arrow length (smaller → longer).
    invert : bool
        If True, flip arrow directions (for sign convention).

    Returns
    -------
    f : np.ndarray, shape (Nx, Ny)
        Reconstructed scalar field.
    fx, fy : np.ndarray, shape (Nx, Ny)
        Partial derivatives ∂f/∂x and ∂f/∂y on the grid.
    """
    os.makedirs(folder, exist_ok=True)
    Nx = len(x_vals)
    Ny = len(y_vals)
    M = Nx * Ny
    idx = lambda i, j: i * Ny + j

    # Build sparse Laplacian and RHS g
    rows, cols, data = [], [], []
    g = np.zeros(M)
    offsets = [(1,0), (-1,0), (0,1), (0,-1)]

    for i in trange(Nx, desc='Building Laplacian'):
        for j in range(Ny):
            n = idx(i, j)
            deg = 0
            for di, dj in offsets:
                ii, jj = i + di, j + dj
                if 0 <= ii < Nx and 0 <= jj < Ny:
                    deg += 1
                    rows.append(n); cols.append(idx(ii, jj)); data.append(-1)
                    g[n] += signs.get(((i, j), (ii, jj)), 0)
            rows.append(n); cols.append(n); data.append(deg)

    L = sp.coo_matrix((data, (rows, cols)), shape=(M, M)).tocsr()
    L[0, :] = 0; L[0, 0] = 1; g[0] = 0

    # Solve for f
    f_vec = spla.spsolve(L, g)
    f = f_vec.reshape((Nx, Ny))

    # Compute gradient with non-uniform spacing
    fx, fy = np.gradient(f, x_vals, y_vals, edge_order=2)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    # Normalize vectors and optionally invert
    speed = np.hypot(fx, fy)
    fx_n = fx / (speed + 1e-8)
    fy_n = fy / (speed + 1e-8)
    if invert:
        fx_n, fy_n = -fx_n, -fy_n

    # Collect every tested data point (nodes)
    tested_nodes = set()
    for (i_j, ii_jj) in signs.keys():
        i, j = i_j
        ii, jj = ii_jj
        if 0 <= i < Nx and 0 <= j < Ny:
            tested_nodes.add((i, j))
        if 0 <= ii < Nx and 0 <= jj < Ny:
            tested_nodes.add((ii, jj))

    tested_x = [x_vals[i] for (i, _) in tested_nodes]
    tested_y = [y_vals[j] for (_, j) in tested_nodes]

    # Plot quiver + tested data points
    fig, ax = plt.subplots(figsize=(10, 10))
    q = ax.quiver(
        X, Y, fx_n, fy_n,
        speed,
        scale=arrow_scale,
        cmap='inferno',
        pivot='mid'
    )
    #cbar = fig.colorbar(q, ax=ax)
    #cbar.set_label('|∇f| (vector magnitude)')

    ax.set_xlabel('U in')
    ax.set_ylabel('U out')
    ax.set_title('Flow Map with Tested Data Points')
    ax.legend()
    plt.tight_layout()

    # Save PDF
    pdf_path = os.path.join(folder, 'flow_map_tested_points.pdf')
    fig.savefig(pdf_path)
    plt.show()
    print(f"Saved flow map with tested data points to {pdf_path}")

    predicted_signs = {}
    for ((i, j), (ii, jj)), true_sign in signs.items():
        if 0 <= i < Nx and 0 <= j < Ny and 0 <= ii < Nx and 0 <= jj < Ny:
            if invert:
                df = -f[ii, jj] + f[i, j]
            else:
                df = f[ii, jj] - f[i, j]
            sign = np.sign(df)
            if sign != 0:  # optional: ignore zero gradients
                predicted_signs[((i, j), (ii, jj))] = int(sign)

    return predicted_signs
