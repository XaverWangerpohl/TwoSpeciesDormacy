import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import re
from operator import itemgetter

def compute_nontrivial_slice(W_birth, W_death, Y_birth, Y_death):
    """
    Compute the positive nontrivial equilibrium (W_eq, Y_eq) by solving:
      Q1 = W_death / W_birth,   Q2 = Y_death / Y_birth
      W_eq = ½ [ (1 − Q1 + Q2) + sqrt((1 − Q1 + Q2)^2 − 4·Q2 ) ]
      Y_eq = ½ [ (1 − Q2 + Q1) + sqrt((1 − Q2 + Q1)^2 − 4·Q1 ) ]
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

def simulate_segment(V0, W0, Y0, X0, Z0,
                     W_birth, Y_birth, W_death, Y_death,
                     X_in, Z_in, X_out, Z_out,
                     duration, dt,
                     use_X, use_Z,
                     tol=1e-6,
                     stop_at_eq=True):
    """
    Integrate from t=0 to t=duration with initial conditions
      V(0)=V0, W(0)=W0, Y(0)=Y0, X(0)=X0, Z(0)=Z0.
    If stop_at_eq=True, stops early when all |dV|,|dW|,|dY| (and |dX| if use_X, |dZ| if use_Z)
    fall below tol. Otherwise, always runs full duration.
    Returns:
      t_array,
      V_array, W_array, Y_array,
      X_raw_array (unscaled), Z_raw_array (unscaled),
      X_plot = X_raw_array * X_scaler, Z_plot = Z_raw_array * Z_scaler.
    """
    X_scaler = X_out / X_in if (use_X and X_in > 0) else 1.0
    Z_scaler = Z_out / Z_in if (use_Z and Z_in > 0) else 1.0

    N = int(np.ceil(duration / dt)) + 1
    t = np.linspace(0.0, duration, N)

    V = np.zeros(N)
    W = np.zeros(N)
    Y = np.zeros(N)
    X_raw = np.zeros(N)
    Z_raw = np.zeros(N)

    V[0] = V0
    W[0] = W0
    Y[0] = Y0
    X_raw[0] = X0
    Z_raw[0] = Z0

    final_index = N - 1
    for i in range(1, N):
        Vi = V[i-1]
        Wi = W[i-1]
        Yi = Y[i-1]
        Xi = X_raw[i-1]
        Zi = Z_raw[i-1]

        # dV/dt, dW/dt
        dV = W_birth * (1 - Wi - Vi) * Vi * Yi - W_death * Vi
        dW = W_birth * (1 - Wi - Vi) * Wi * Yi - W_death * Wi

        # dY/dt
        dY = Y_birth * (1 - Yi) * Yi * (Vi + Wi) - Y_death * Yi

        # X-coupling
        if use_X:
            dW += X_out * Xi - X_in * Wi
        # Z-coupling
        if use_Z:
            dY += Z_out * Zi - Z_in * Yi

        # dX/dt, dZ/dt
        dX = - X_out * Xi + X_in * Wi
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
        X_raw[i] = Xi + dt * dX
        Z_raw[i] = Zi + dt * dZ

        # Enforce nonnegativity
        V[i] = max(V[i], 0.0)
        W[i] = max(W[i], 0.0)
        Y[i] = max(Y[i], 0.0)
        X_raw[i] = max(X_raw[i], 0.0)
        Z_raw[i] = max(Z_raw[i], 0.0)

    # Truncate arrays
    t_trunc     = t[: final_index + 1]
    V_trunc     = V[: final_index + 1]
    W_trunc     = W[: final_index + 1]
    Y_trunc     = Y[: final_index + 1]
    X_raw_trunc = X_raw[: final_index + 1]
    Z_raw_trunc = Z_raw[: final_index + 1]

    X_plot = X_raw_trunc * X_scaler
    Z_plot = Z_raw_trunc * Z_scaler

    return t_trunc, V_trunc, W_trunc, Y_trunc, X_raw_trunc, Z_raw_trunc, X_plot, Z_plot

def simulate_segment2(U0, V0, W0, X0, Y0, Z0,
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
      X_raw_array (unscaled), Z_raw_array (unscaled),
      X_plot = X_raw_array * X_scaler, Z_plot = Z_raw_array * Z_scaler.
    """

    X_scaler = X_out / X_in if (use_X and X_in > 0) else 1.0
    Z_scaler = Z_out / Z_in if (use_Z and Z_in > 0) else 1.0

    N = int(np.ceil(duration / dt)) + 1
    t = np.linspace(0.0, duration, N)

    U_raw = np.zeros(N)
    V = np.zeros(N)
    W = np.zeros(N)
    X_raw = np.zeros(N)
    Y = np.zeros(N)
    Z_raw = np.zeros(N)

    U_raw[0] = U0
    V[0] = V0
    W[0] = W0
    X_raw[0] = X0
    Y[0] = Y0
    Z_raw[0] = Z0

    final_index = N - 1
    for i in range(1, N):
        Vi = V[i-1]
        Wi = W[i-1]
        Yi = Y[i-1]
        Xi = X_raw[i-1]
        Zi = Z_raw[i-1]
        Ui = U_raw[i-1]

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

        # dX/dt, dZ/dt
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
        X_raw[i] = Xi + dt * dX
        Z_raw[i] = Zi + dt * dZ
        U_raw[i] = Ui + dt * dU

        # Enforce nonnegativity
        V[i] = max(V[i], 0.0)
        W[i] = max(W[i], 0.0)
        Y[i] = max(Y[i], 0.0)
        X_raw[i] = max(X_raw[i], 0.0)
        Z_raw[i] = max(Z_raw[i], 0.0)
        U_raw[i] = max(U_raw[i], 0.0)

    # Truncate arrays
    t_trunc     = t[: final_index + 1]
    V_trunc     = V[: final_index + 1]
    W_trunc     = W[: final_index + 1]
    Y_trunc     = Y[: final_index + 1]
    X_raw_trunc = X_raw[: final_index + 1]
    Z_raw_trunc = Z_raw[: final_index + 1]
    U_raw_trunc = U_raw[: final_index + 1]

    X_plot = X_raw_trunc * X_scaler
    Z_plot = Z_raw_trunc * Z_scaler
    U_plot = U_raw_trunc * X_scaler

    return t_trunc, V_trunc, W_trunc, Y_trunc, X_raw_trunc, Z_raw_trunc, U_raw_trunc, X_plot, Z_plot, U_plot

def compute_deltaW_curve(W_birth, Y_birth, W_death, Y_death,
                         X_in, Z_in, X_out, Z_out,
                         Time, dt, use_X, use_Z,
                         num_points, severity,
                          perturb_W, perturb_Y,
                         tol):
    """
    Compute W0_values and corresponding ΔW for a given 'severity',
    then trim off every W0 whose (W_final + V_final) < 0.1·W_eq.

    Returns:
      - W0_surv       : 1D array of surviving W0 (starting at the minimal surviving W0)
      - DeltaW_surv   : corresponding ΔW = W_final − W0
      - integral_surv : ∫|ΔW| dW0 over that surviving range
      - W0_min_surv   : the smallest W0 that survives, or None if none survive
    """
    W_eq, Y_eq = compute_nontrivial_slice(W_birth, W_death, Y_birth, Y_death)
    if (W_eq is None) or (Y_eq is None):
        raise RuntimeError("No positive, nontrivial equilibrium exists.")

    W0_values    = np.linspace(0.0, W_eq, num_points)
    DeltaW       = np.zeros_like(W0_values)
    W_final_vals = np.zeros_like(W0_values)
    V_final_vals = np.zeros_like(W0_values)

    for idx, W0 in enumerate(W0_values):
        V0 = W_eq - W0
        X0 = (X_in / X_out) * W0 if use_X else 0.0
        Z0 = (Z_in / Z_out) * Y_eq if use_Z else 0.0

        M   = 1.0 - severity
        V0p = (M * V0)   if perturb_W else V0
        W0p = (M * W0)   if perturb_W else W0
        Y0p = (M * Y_eq) if perturb_Y else Y_eq

        _, V_arr, W_arr, Y_arr, _, _, _, _ = simulate_segment(
            V0=V0p,  W0=W0p,  Y0=Y0p,  X0=X0,  Z0=Z0,
            W_birth=W_birth, Y_birth=Y_birth,
            W_death=W_death, Y_death=Y_death,
            X_in=X_in, Z_in=Z_in,
            X_out=X_out, Z_out=Z_out,
            duration=Time, dt=dt,
            use_X=use_X, use_Z=use_Z,
            tol=tol,
            stop_at_eq=True
        )
        W_final = W_arr[-1]
        V_final = V_arr[-1]

        W_final_vals[idx] = W_final
        V_final_vals[idx] = V_final
        DeltaW[idx]       = W_final - W0

    threshold     = 0.1 * W_eq
    survival_mask = (W_final_vals + V_final_vals >= threshold)
    if not np.any(survival_mask):
        return np.array([]), np.array([]), 0.0, None

    first_surv_idx = np.argmax(survival_mask)
    W0_surv        = W0_values[first_surv_idx:]
    DeltaW_surv    = DeltaW[first_surv_idx:]
    integral_surv  = np.trapz(np.abs(DeltaW_surv), W0_surv)
    W0_min_surv    = W0_values[first_surv_idx]

    return W0_surv, DeltaW_surv, integral_surv, W0_min_surv

def compare_severities(W_birth, Y_birth, W_death, Y_death,
                       X_in, Z_in, X_out, Z_out,
                       Time=200.0, dt=0.01,
                       use_X=True, use_Z=True,
                       num_points=100,
                       severity_range=(0.2, 0.4), n_sev=5,
                       perturb_W=False, perturb_Y=True,
                       tol=1e-6,
                       verbose=False):
    """
    For severities in [severity_range[0], severity_range[1]] (n_sev points),
    compute ΔW vs W0—but only over the surviving subrange, defined by
    (W_final + V_final) ≥ 0.1·W_eq.  Plot each truncated curve in reversed viridis,
    draw a vertical dashed line at each threshold W0_min_surv (if > 0),
    and save a PDF of the figure into a single folder named 'compare_severities'.
    All PDFs go into that folder, with integer‐suffix filenames to avoid overwriting.
    If verbose=False, suppress all print statements.
    """
    severities = np.linspace(severity_range[0], severity_range[1], n_sev)
    print(severities) if verbose else None

    try:
        base_cmap = cm.colormaps['viridis']
    except AttributeError:
        base_cmap = cm.get_cmap('viridis')
    cmap = base_cmap.reversed()
    norm = mcolors.Normalize(vmin=severity_range[0], vmax=severity_range[1])

    fig, ax = plt.subplots(figsize=(8, 6))
    integrals = np.zeros(n_sev)
    curves   = []
    valid_idx = []

    # 1) Compute each ΔW‐curve (truncated) and record its W0_min_surv
    for i, sev in enumerate(severities):
        W0_surv, DeltaW_surv, integral_surv, W0_min_surv = compute_deltaW_curve(
            W_birth=W_birth, Y_birth=Y_birth,
            W_death=W_death, Y_death=Y_death,
            X_in=X_in, Z_in=Z_in, X_out=X_out, Z_out=Z_out,
            Time=Time, dt=dt, use_X=use_X, use_Z=use_Z,
            num_points=num_points,
            severity=sev,
             perturb_W=perturb_W, perturb_Y=perturb_Y,
            tol=tol
        )

        if W0_min_surv is None:
            if verbose:
                print(f"Warning: severity={sev:.3f} → no survivors. Skipping.")
            curves.append((None, None, None))
            integrals[i] = 0.0
            continue

        curves.append((W0_surv, DeltaW_surv, W0_min_surv))
        integrals[i] = integral_surv
        valid_idx.append(i)

    if len(valid_idx) == 0:
        raise RuntimeError("All severities lead to extinction; no plot generated.")

    max_int = np.max(integrals[valid_idx])
    rel_ints = integrals / max_int

    if verbose:
        print("Severity   Relative ∫|ΔW| dW₀")
        for sev, rel in zip(severities, rel_ints):
            print(f"{sev:.3f}      {rel:.4f}")

    # 2) Plot each truncated ΔW‐curve plus a vertical line at W0_min_surv (if > 0)
    for i in valid_idx:
        sev            = severities[i]
        W0_surv, DeltaW_surv, W0_min_surv = curves[i]
        color          = cmap(norm(sev))

        # 2a) the truncated curve
        ax.plot(
            W0_surv, DeltaW_surv,
            color=color, linewidth=1.8,
            label=f"sev={sev:.3f}, W₀₋min={W0_min_surv:.4f}"
        )

        # 2b) vertical line from y=0 up to ΔW_surv[0], only if W0_min_surv > 0
        if W0_min_surv > 0.0:
            delta_at_threshold = DeltaW_surv[0]
            ax.vlines(
                x=W0_min_surv,
                ymin=0.0,
                ymax=delta_at_threshold,
                color=color,
                linestyle='--',
                linewidth=1,
                alpha=0.7
            )

    if perturb_W:
        perturb = "W"
    else:
        perturb = "Y"
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(r'$W_{0}$', fontsize=12)
    ax.set_ylabel(r'$\Delta W$', fontsize=12)
    ax.set_title(f'ΔW vs W₀ ({perturb} extinction)', fontsize=14)
    ax.grid(True)

    # 3) Colorbar for severity
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('severity', fontsize=12)

    # 4) Legend (uncomment if desired)
    # ax.legend(loc='upper left', fontsize=8, framealpha=0.9, bbox_to_anchor=(1.2, 1))

    plt.tight_layout()

    # --------------------------
    # 5) Saving logic (all PDFs in the same folder)
    folder_name = "compare_severities"
    os.makedirs(folder_name, exist_ok=True)

    base_pdf = "compare_severities"
    pattern = os.path.join(folder_name, base_pdf + "*.pdf")
    existing_pdfs = glob.glob(pattern)
    if not existing_pdfs:
        pdf_name = base_pdf + ".pdf"
    else:
        taken = set()
        for p in existing_pdfs:
            stem = os.path.basename(p)
            tail = stem.replace(base_pdf, "").replace(".pdf", "")
            if tail == "":
                taken.add(0)
            elif tail.isdigit():
                taken.add(int(tail))
        k = 0
        while k in taken:
            k += 1
        pdf_name = f"{base_pdf}{k}.pdf"

    full_path = os.path.join(folder_name, pdf_name)
    plt.savefig(full_path)
    if verbose:
        print(f"Figure saved to: {full_path}")
    # --------------------------

    plt.show()

def test_plot(W_birth, Y_birth, W_death, Y_death,
              X_in, Z_in, X_out, Z_out,
              W0,
              Time=200.0, dt=0.01,
              use_X=True, use_Z=True,
              severity=0.5,
               perturb_W=False, perturb_Y=True,
              perturb_time=20.0,
              tol=1e-6):
    """
    Build a time‐series plot for a fixed W0 using perturbation multiplier = (1 - severity).
    1) Compute (W_eq, Y_eq).
    2) Verify W0 ∈ [0, W_eq], then set V0 = W_eq - W0, X0, Z0.
    3) Stage A: simulate from t=0 → perturb_time (no stopping).
    4) At t=0 apply perturbation: V_mid=(1−severity)*V_eq_pre, etc.
    5) Stage B: simulate from t=0 → Time (no stopping).
    6) Concatenate and plot V, W, Y, X, Z over t ∈ [−perturb_time, Time].
    Returns results dict.
    """
    # (1) equilibrium
    W_eq, Y_eq = compute_nontrivial_slice(W_birth, W_death, Y_birth, Y_death)
    if (W_eq is None) or (Y_eq is None):
        raise RuntimeError("No positive equilibrium exists.")

    if not (0.0 <= W0 <= W_eq):
        raise ValueError(f"W0 must lie in [0, {W_eq:.4f}].")

    X_scaler = X_out / X_in if (use_X and X_in > 0) else 1.0
    Z_scaler = Z_out / Z_in if (use_Z and Z_in > 0) else 1.0

    V0_eq = W_eq - W0
    X0_eq = (X_in / X_out) * W0 if use_X else 0.0
    Z0_eq = (Z_in / Z_out) * Y_eq if use_Z else 0.0

    # Stage A
    t_pre, V_pre, W_pre, Y_pre, X_pre_raw, Z_pre_raw, X_pre_plot, Z_pre_plot = simulate_segment(
        V0=V0_eq, W0=W0, Y0=Y_eq, X0=X0_eq, Z0=Z0_eq,
        W_birth=W_birth, Y_birth=Y_birth,
        W_death=W_death, Y_death=Y_death,
        X_in=X_in, Z_in=Z_in,
        X_out=X_out, Z_out=Z_out,
        duration=perturb_time, dt=dt,
        use_X=use_X, use_Z=use_Z,
        tol=tol,
        stop_at_eq=False
    )
    t_pre_shifted = t_pre - perturb_time

    V_eq_pre = V_pre[-1]
    W_eq_pre = W_pre[-1]
    Y_eq_pre = Y_pre[-1]
    X_eq_pre = X_pre_plot[-1]
    Z_eq_pre = Z_pre_plot[-1] if use_Z else None

    # (4) apply perturbation multiplier = (1 - severity)
    V_mid = ((1 - severity) * V_eq_pre) if perturb_W else V_eq_pre
    W_mid = ((1 - severity) * W_eq_pre) if perturb_W else W_eq_pre
    Y_mid = ((1 - severity) * Y_eq_pre) if perturb_Y else Y_eq_pre

    # Stage B
    t_post, V_post, W_post, Y_post, X_post_raw, Z_post_raw, X_post_plot, Z_post_plot = simulate_segment(
        V0=V_mid, W0=W_mid, Y0=Y_mid, X0=X0_eq, Z0=Z0_eq,
        W_birth=W_birth, Y_birth=Y_birth,
        W_death=W_death, Y_death=Y_death,
        X_in=X_in, Z_in=Z_in,
        X_out=X_out, Z_out=Z_out,
        duration=Time, dt=dt,
        use_X=use_X, use_Z=use_Z,
        tol=tol,
        stop_at_eq=False
    )

    t_full = np.concatenate((t_pre_shifted, t_post))
    V_full = np.concatenate((V_pre, V_post))
    W_full = np.concatenate((W_pre, W_post))
    Y_full = np.concatenate((Y_pre, Y_post))
    X_full = np.concatenate((X_pre_plot, X_post_plot))
    Z_full = np.concatenate((Z_pre_plot, Z_post_plot)) if use_Z else None

    V_final = V_full[-1]
    W_final = W_full[-1]
    Y_final = Y_full[-1]
    X_final = X_full[-1]
    Z_final = Z_full[-1] if use_Z else None
    delta_W_test = W_final - W_eq_pre

    # Time‐series plot
    plt.figure(figsize=(8, 5))
    if use_X:
        plt.plot(t_full, X_full, label=r'$X(t)$', color='gold', linewidth=1.5)
    if use_Z:
        plt.plot(t_full, Z_full, label=r'$Z(t)$', color='skyblue', linewidth=1.5)
    plt.plot(t_full, Y_full, label=r'$Y(t)$', color='darkblue', linewidth=1.5)
    plt.plot(t_full, V_full, label=r'$V(t)$', color='orange', linewidth=1.5)
    plt.plot(t_full, W_full, label=r'$W(t)$', color='darkgreen', linewidth=1.5)

    plt.axvline(x=0.0, color='gray', linestyle='--', lw=1)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.title(
        f'Time Series at $W_{0} = {W0:.4f}$ (severity={severity:.2f})\n'
        + rf'$\Delta W_{{\mathrm{{test}}}} = {delta_W_test:.4f}$',
        fontsize=14
    )
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        'W_eq'        : W_eq,
        'Y_eq'        : Y_eq,
        'W0'          : W0,
        'V_eq_pre'    : V_eq_pre,
        'W_eq_pre'    : W_eq_pre,
        'Y_eq_pre'    : Y_eq_pre,
        'X_eq_pre'    : X_eq_pre,
        'Z_eq_pre'    : Z_eq_pre,
        'V_mid'       : V_mid,
        'W_mid'       : W_mid,
        'Y_mid'       : Y_mid,
        't_full'      : t_full,
        'V_full'      : V_full,
        'W_full'      : W_full,
        'Y_full'      : Y_full,
        'X_full'      : X_full,
        'Z_full'      : Z_full,
        'delta_W_test': delta_W_test
    }

def test_plot2(U0, V0, W0, X0, Y0, Z0,
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
              tol=1e-7):
    """
    Build a time‐series plot for a fixed W0 using perturbation multiplier = (1 - severity).
    1) Compute (W_eq, Y_eq).
    2) Verify W0 ∈ [0, W_eq], then set V0 = W_eq - W0, X0, Z0.
    3) Stage A: simulate from t=0 → perturb_time (no stopping).
    4) At t=0 apply perturbation: V_mid=(1−severity)*V_eq_pre, etc.
    5) Stage B: simulate from t=0 → Time (no stopping).
    6) Concatenate and plot V, W, Y, X, Z over t ∈ [−perturb_time, Time].
    Returns results dict.
    """


    # Stage A
    t_pre, V_pre, W_pre, Y_pre, X_raw_trunc, Z_raw_trunc, U_raw_trunc, X_pre_plot, Z_plot, U_pre_plot = simulate_segment2(
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
    plt.figure(figsize=(8, 5))
    plt.plot(t_pre, V_pre, label=r'$V(t)$', color='orange', linewidth=1.5)
    plt.plot(t_pre, W_pre, label=r'$W(t)$', color='darkgreen', linewidth=1.5)
    plt.plot(t_pre, Y_pre, label=r'$Y(t)$', color='darkblue', linewidth=1.5)
    plt.show()
 
    t_pre_shifted = t_pre - perturb_time

    V_eq_pre = V_pre[-1]
    W_eq_pre = W_pre[-1]
    Y_eq_pre = Y_pre[-1]
    X_eq_pre = X_pre_plot[-1]
    U_eq_pre = U_pre_plot[-1]

    # (4) apply perturbation multiplier = (1 - severity)
    V_mid = ((1 - severity) * V_eq_pre) if perturb_W else V_eq_pre
    W_mid = ((1 - severity) * W_eq_pre) if perturb_W else W_eq_pre
    Y_mid = ((1 - severity) * Y_eq_pre) if perturb_Y else Y_eq_pre

    # Stage B
    t_post, V_post, W_post, Y_post, X_post_raw, Z_post_raw, U_raw_trunc, X_post_plot, Z_post_plot, U_post_plot = simulate_segment2(
        V0=V_mid, W0=W_mid, Y0=Y_mid, X0=X0, Z0=Z0, U0=U0,
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
    V_full = np.concatenate((V_pre, V_post))
    W_full = np.concatenate((W_pre, W_post))
    Y_full = np.concatenate((Y_pre, Y_post))
    X_full = np.concatenate((X_pre_plot, X_post_plot))

    V_final = V_full[-1]
    W_final = W_full[-1]
    Y_final = Y_full[-1]
    X_final = X_full[-1]
    delta_W_test = W_final - W_eq_pre

    # Time‐series plot
    plt.figure(figsize=(8, 5))
    if use_X:
        plt.plot(t_full, X_full, label=r'$X(t)$', color='gold', linewidth=1.5)
    plt.plot(t_full, Y_full, label=r'$Y(t)$', color='darkblue', linewidth=1.5)
    plt.plot(t_full, V_full, label=r'$V(t)$', color='orange', linewidth=1.5)
    plt.plot(t_full, W_full, label=r'$W(t)$', color='darkgreen', linewidth=1.5)

    plt.axvline(x=0.0, color='gray', linestyle='--', lw=1)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.title(
        f'Time Series at $W_{0} = {W0:.4f}$ (severity={severity:.2f})\n'
        + rf'$\Delta W_{{\mathrm{{test}}}} = {delta_W_test:.4f}$',
        fontsize=14
    )
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        'W0'          : W0,
        'V_eq_pre'    : V_eq_pre,
        'W_eq_pre'    : W_eq_pre,
        'Y_eq_pre'    : Y_eq_pre,
        'X_eq_pre'    : X_eq_pre,
        'V_mid'       : V_mid,
        'W_mid'       : W_mid,
        'Y_mid'       : Y_mid,
        't_full'      : t_full,
        'V_full'      : V_full,
        'W_full'      : W_full,
        'Y_full'      : Y_full,
        'X_full'      : X_full,
        'delta_W_test': delta_W_test
    }

def compare_scalers(W_birth, Y_birth, W_death, Y_death,
                       X_in, Z_in, X_out, Z_out, severity,
                       Time=200.0, dt=0.01,
                       use_X=True, use_Z=True,
                       scale_X=True,
                       num_points=100,
                       scaler_range=(0.5, 2), n_scaler=5,
                       perturb_W=False, perturb_Y=True,
                       tol=1e-6,
                       verbose=False):
    """
    For severities in [scaler_range[0], scaler_range[1]] (n_scaler points),
    compute ΔW vs W0—but only over the surviving subrange, defined by
    (W_final + V_final) ≥ 0.1·W_eq.  Plot each truncated curve in reversed viridis,
    draw a vertical dashed line at each threshold W0_min_surv (if > 0),
    and save a PDF of the figure into a single folder named 'compare_severities'.
    All PDFs go into that folder, with integersuffix filenames to avoid overwriting.
    If verbose=False, suppress all print statements.
    """
    scalers = np.linspace(scaler_range[0], scaler_range[1], n_scaler)
    print(scalers) if verbose else None

    try:
        base_cmap = cm.colormaps['viridis']
    except AttributeError:
        base_cmap = cm.get_cmap('viridis')
    cmap = base_cmap.reversed()
    norm = mcolors.Normalize(vmin=scaler_range[0], vmax=scaler_range[1])

    fig, ax = plt.subplots(figsize=(8, 6))
    integrals = np.zeros(n_scaler)
    curves   = []
    valid_idx = []

    # 1) Compute each ΔW‐curve (truncated) and record its W0_min_surv
    if scale_X:
        for i, scals in enumerate(scalers):
            W0_surv, DeltaW_surv, integral_surv, W0_min_surv = compute_deltaW_curve(
                W_birth=W_birth, Y_birth=Y_birth,
                W_death=W_death, Y_death=Y_death,
                X_in=X_in*scals, Z_in=Z_in, X_out=X_out*scals, Z_out=Z_out,
                Time=Time, dt=dt, use_X=use_X, use_Z=use_Z,
                num_points=num_points,
                severity=severity,
                perturb_W=perturb_W, perturb_Y=perturb_Y,
                tol=tol
            )

            if W0_min_surv is None:

                curves.append((None, None, None))
                integrals[i] = 0.0
                continue

            curves.append((W0_surv, DeltaW_surv, W0_min_surv))
            integrals[i] = integral_surv
            valid_idx.append(i)
    else:
        for i, scals in enumerate(scalers):
            W0_surv, DeltaW_surv, integral_surv, W0_min_surv = compute_deltaW_curve(
                W_birth=W_birth, Y_birth=Y_birth,
                W_death=W_death, Y_death=Y_death,
                X_in=X_in, Z_in=Z_in*scals, X_out=X_out, Z_out=Z_out*scals,
                Time=Time, dt=dt, use_X=use_X, use_Z=use_Z,
                num_points=num_points,
                severity=severity,
                perturb_W=perturb_W, perturb_Y=perturb_Y,
                tol=tol
            )

            if W0_min_surv is None:
                curves.append((None, None, None))
                integrals[i] = 0.0
                continue

            curves.append((W0_surv, DeltaW_surv, W0_min_surv))
            integrals[i] = integral_surv
            valid_idx.append(i)

    if len(valid_idx) == 0:
        raise RuntimeError("All scalers lead to extinction; no plot generated.")

    max_int = np.max(integrals[valid_idx])
    rel_ints = integrals / max_int

    if verbose:
        print("Scalers   Relative ∫|ΔW| dW₀")
        for scal, rel in zip(scalers, rel_ints):
            print(f"{scal:.3f}      {rel:.4f}")

    # 2) Plot each truncated ΔW‐curve plus a vertical line at W0_min_surv (if > 0)
    for i in valid_idx:
        scals            = scalers[i]
        W0_surv, DeltaW_surv, W0_min_surv = curves[i]
        color          = cmap(norm(scals))

        # 2a) the truncated curve
        ax.plot(
            W0_surv, DeltaW_surv,
            color=color, linewidth=1.8,
            label=f"scal={scals:.3f}, W₀₋min={W0_min_surv:.4f}"
        )

        # 2b) vertical line from y=0 up to ΔW_surv[0], only if W0_min_surv > 0
        if W0_min_surv > 0.0:
            delta_at_threshold = DeltaW_surv[0]
            ax.vlines(
                x=W0_min_surv,
                ymin=0.0,
                ymax=delta_at_threshold,
                color=color,
                linestyle='--',
                linewidth=1,
                alpha=0.7
            )

    if perturb_W:
        perturb = "W"
    else:
        perturb = "Y"
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(r'$W_{0}$', fontsize=12)
    ax.set_ylabel(r'$\Delta W$', fontsize=12)
    ax.set_title(f'ΔW vs W₀ ({perturb} extinction)', fontsize=14)
    ax.grid(True)

    # 3) Colorbar for Scalers
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Scalers', fontsize=12)

    # 4) Legend (uncomment if desired)
    # ax.legend(loc='upper left', fontsize=8, framealpha=0.9, bbox_to_anchor=(1.2, 1))

    plt.tight_layout()

    # --------------------------
    # 5) Saving logic (all PDFs in the same folder)
    folder_name = "compare_scalers"
    os.makedirs(folder_name, exist_ok=True)

    base_pdf = "compare_scalers"
    pattern = os.path.join(folder_name, base_pdf + "*.pdf")
    existing_pdfs = glob.glob(pattern)
    if not existing_pdfs:
        pdf_name = base_pdf + ".pdf"
    else:
        taken = set()
        for p in existing_pdfs:
            stem = os.path.basename(p)
            tail = stem.replace(base_pdf, "").replace(".pdf", "")
            if tail == "":
                taken.add(0)
            elif tail.isdigit():
                taken.add(int(tail))
        k = 0
        while k in taken:
            k += 1
        pdf_name = f"{base_pdf}{k}.pdf"

    full_path = os.path.join(folder_name, pdf_name)
    plt.savefig(full_path)
    if verbose:
        print(f"Figure saved to: {full_path}")
    # --------------------------

    plt.show()

def run_cycles_ext(V0, W0, Y0, X0, Z0,
                   W_birth, Y_birth, W_death, Y_death,
                   X_in, Z_in, X_out, Z_out,
                   extinction_rates, dt,
                   use_X, use_Z,
                   cycles,
                   severity,
                   perturb_W=False,
                   perturb_Y=False,
                   save_plot=True):
    """
    For each duration in `extinction_rates`, run `cycles` chained simulate_segment calls
    (starting from equilibrium‐based initials), record the final W each cycle,
    and plot all W_final vs. cycle curves on one axes, colored by duration via viridis.
    If save_plot=True, saves the figure into ./run_cycles_ext/run_cycles_ext[.pdf | N.pdf].
    """
    # Prepare colormap
    try:
        base_cmap = cm.colormaps['viridis']
    except AttributeError:
        base_cmap = cm.get_cmap('viridis')
    cmap = base_cmap
    norm = mcolors.Normalize(vmin=min(extinction_rates), vmax=max(extinction_rates))

    plt.figure(figsize=(8, 5))
    cycles_idx = np.arange(1, cycles + 1)

    for dur in extinction_rates:
        # equilibrium‐based initials
        W_eq, Y_eq = compute_nontrivial_slice(W_birth, W_death, Y_birth, Y_death)
        if W_eq is None:
            raise RuntimeError("No positive equilibrium exists.")
        V_curr, W_curr, Y_curr, X_curr, Z_curr = W_eq-W0, W0, Y_eq, W0, Y_eq

        W_finals = []
        for _ in range(cycles):
            _, V_arr, W_arr, Y_arr, X_arr, Z_arr, X_plot, Z_plot = simulate_segment(
                V_curr, W_curr, Y_curr, X_curr, Z_curr,
                W_birth, Y_birth, W_death, Y_death,
                X_in, Z_in, X_out, Z_out,
                duration=dur, dt=dt,
                use_X=use_X, use_Z=use_Z,
                tol=1e-6, stop_at_eq=False
            )
            V_final, W_final, Y_final = V_arr[-1], W_arr[-1], Y_arr[-1]
            W_finals.append(W_final)

            # perturb
            if perturb_W:
                V_curr, W_curr = (1-severity)*V_final, (1-severity)*W_final
            else:
                V_curr, W_curr = V_final, W_final
            Y_curr = (1-severity)*Y_final if perturb_Y else Y_final
            X_curr, Z_curr = X_arr[-1], Z_arr[-1]

        color = cmap(norm(dur))
        plt.plot(cycles_idx, W_finals, color=color, label=f"T={dur}")

    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel(r'$W_{\mathrm{final}}$', fontsize=12)
    plt.title(f'Final W vs Cycle for different extinction rates', fontsize=14)
    plt.grid(True)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, pad=0.02, aspect=30)
    cbar.set_label('Rates', fontsize=12)

    plt.legend(title='Rates', fontsize=9, title_fontsize=10)
    plt.tight_layout()

    # Saving
    if save_plot:
        folder = "run_cycles_ext"
        os.makedirs(folder, exist_ok=True)
        base = "run_cycles_ext"
        pattern = os.path.join(folder, base + "*.pdf")
        existing = glob.glob(pattern)
        if not existing:
            pdf_name = base + ".pdf"
        else:
            taken = set(int(os.path.basename(p).replace(base, "").replace(".pdf","") or 0)
                        for p in existing if os.path.basename(p).replace(base, "").replace(".pdf","").isdigit() or p.endswith(base+".pdf"))
            k = 0
            while k in taken:
                k += 1
            pdf_name = f"{base}{k}.pdf"
        path = os.path.join(folder, pdf_name)
        plt.savefig(path)
        print(f"Saved run_cycles_ext plot to {path}")

    plt.show()

def run_cycles(V0, W0, Y0, X0, Z0,
               W_birth, Y_birth, W_death, Y_death,
               X_in, Z_in, X_out, Z_out,
               extinction_rate, dt,
               use_X, use_Z,
               cycles,
               severity,
               perturb_W=False,
               perturb_Y=False,
               save_plot=True):
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
    V_curr, W_curr, Y_curr, X_curr, Z_curr = V0, W0, Y0, X0, Z0
    V_finals = []
    W_finals = []
    Y_finals = []

    for n in range(1, cycles+1):
        # 1) simulate one segment
        t_arr, V_arr, W_arr, Y_arr, X_arr, Z_arr, X_plot, Z_plot = simulate_segment(
            V_curr, W_curr, Y_curr, X_curr, Z_curr,
            W_birth, Y_birth, W_death, Y_death,
            X_in, Z_in, X_out, Z_out,
            extinction_rate, dt,
            use_X, use_Z,
            tol=1e-6,
            stop_at_eq=False
        )

        # 2) record finals
        V_final = V_arr[-1]
        W_final = W_arr[-1]
        Y_final = Y_arr[-1]
        V_finals.append(V_final)
        W_finals.append(W_final)
        Y_finals.append(Y_final)

        # 3) perturb for next cycle
        if perturb_W:
            V_curr = (1 - severity) * V_final
            W_curr = (1 - severity) * W_final
        else:
            V_curr = V_final
            W_curr = W_final

        if perturb_Y:
            Y_curr = (1 - severity) * Y_final
        else:
            Y_curr = Y_final

        # 4) carry over X, Z unchanged
        X_curr = X_arr[-1]
        Z_curr = Z_arr[-1]

    # plot all three on one figure
    cycles_idx = np.arange(1, cycles+1)
    plt.figure(figsize=(8, 5))
    plt.plot(cycles_idx, W_finals, label='W final', color='darkgreen')
    plt.plot(cycles_idx, V_finals, label='V final', color='orange')
    plt.plot(cycles_idx, Y_finals, label='Y final', color='darkblue')
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('Final Value', fontsize=12)
    plt.title(f'Final V, W, Y after each cycle\n(severity={severity}, perturb_W={perturb_W}, perturb_Y={perturb_Y})', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Saving
    if save_plot:
        folder = "run_cycles"
        os.makedirs(folder, exist_ok=True)
        base = "run_cycles"
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
        print(f"Saved run_cycles plot to {path}")

    plt.show()

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
               perturb_Y=False,
               plot=False,
               stop=None,
               break_threshold=0.01):
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
    X0 = W0 / (X_out / X_in)
    U0 = V0 / (U_out / U_in)
    Z0 = Y0 / (Z_out /Z_in)

    V_curr, W_curr, Y_curr, X_curr, Z_curr, U_curr = V0, W0, Y0, X0, Z0, U0

    V_finals = []
    W_finals = []
    Y_finals = []

    for n in range(1, cycles+1):

        # 1) simulate one segment
        t_arr, V_arr, W_arr, Y_arr, X_arr, Z_arr, U_arr, X_plot, Z_plot, U_plot = simulate_segment2(
            V0=V_curr, W0=W_curr, Y0=Y_curr, X0=X_curr, Z0=Z_curr, U0=U_curr,
            W_birth=W_birth, Y_birth=Y_birth, W_death=W_death, Y_death=Y_death,
            X_in=X_in, Z_in=Z_in, X_out=X_out, Z_out=Z_out, U_in=U_in, U_out=U_out,
            duration=extinction_rate, dt=dt,
            use_X=use_X, use_Z=use_Z,
            tol=1e-7,
            stop_at_eq=True
        )


        # 2) record finals
        V_final = V_arr[-1]
        W_final = W_arr[-1]
        Y_final = Y_arr[-1]
        
        V_finals.append(V_final)
        W_finals.append(W_final)
        Y_finals.append(Y_final)

        if n == 50:
            W0 = W_final
        
        if (abs(W_final - W0) > break_threshold) and n > 50:
            break

        # 3) perturb for next cycle
        if perturb_W:
            V_curr = (1 - severity) * V_final
            W_curr = (1 - severity) * W_final
        else:
            V_curr = V_final
            W_curr = W_final

        if perturb_Y:
            Y_curr = (1 - severity) * Y_final
        else:
            Y_curr = Y_final

        # 4) carry over X, Z unchanged
        X_curr = X_arr[-1]
        Z_curr = Z_arr[-1]
        U_curr = U_arr[-1]


    if plot:

        # plot all three on one figure
        cycles_idx = np.arange(1, n+1)
        plt.figure(figsize=(8, 5))
        plt.plot(cycles_idx, W_finals, label='W final', color='darkgreen')
        plt.plot(cycles_idx, V_finals, label='V final', color='orange')
        plt.plot(cycles_idx, Y_finals, label='Y final', color='darkblue')
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

        # Saving
        
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

def test_invasion(
    U_in_vals, U_out_vals, deltas, test_point,
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
    X_vals = sorted(X_vals, key=itemgetter(0, 1))
    
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

def pairwise_invasion_plot(
    V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size=50,
    U_min=0.01, U_max=0.99,
    X_min=0.01, X_max=0.99,
    perturb_W=False, perturb_Y=True, speedplot=False

):
    """
    Generate a pairwise invasion plot (PIP) over U_in (x-axis) and X_in (y-axis).
    Uses X_out = X_in and U_out = U_in for each grid point.
    Only computes for X_in >= U_in and mirrors symmetry.
    Saves both continuous ΔW heatmap and binary invasion/failure plot without overwriting.
    """
    # Prepare output folder
    folder = 'pip_plots'
    os.makedirs(folder, exist_ok=True)

    U_vals = np.linspace(U_min, U_max, grid_size)
    X_vals = np.linspace(X_min, X_max, grid_size)
    deltaW_matrix = np.zeros((grid_size, grid_size))

    for i, U_in in enumerate(tqdm(U_vals, desc="Scanning U_in")):
        for j, X_in in enumerate(X_vals):
            # Mirror diagonal
            if j == i:
                deltaW_matrix[j, i] = 0.0
                continue
            if X_in < U_in:
                deltaW_matrix[j, i] = -deltaW_matrix[i, j]
                continue

            # Compute ΔW via run_invasion
            deltaW = run_invasion(
             V0, W0, Y0,
               W_birth, Y_birth,
               W_death, Y_death,
               X_in, X_in,
               U_in, U_in,
               Z_in, Z_out,
                extinction_rate, dt,
                use_X, use_Z,
                severity,
                cycles,
                perturb_W,
                perturb_Y,
                plot=False
            )
            deltaW_matrix[j, i] = deltaW
    if speedplot:
        # Continuous heatmap: determine next filename index
        cont_pattern = os.path.join(folder, 'pip_speed*.pdf')
        existing_cont = glob.glob(cont_pattern)
        cont_idxs = []
        for p in existing_cont:
            m = re.match(r'.*pip_speed(\d*)\.pdf$', os.path.basename(p))
            if m:
                idx = int(m.group(1)) if m.group(1) else 0
                cont_idxs.append(idx)
        next_cont = max(cont_idxs) + 1 if cont_idxs else 0
        cont_fname = f'pip_speed{next_cont}.pdf'

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            deltaW_matrix,
            origin='lower',
            extent=[U_min, U_max, X_min, X_max],
            aspect='auto'
        )
        plt.xlabel('U_in')
        plt.ylabel('X_in')
        plt.title(f'Pairwise Invasion Plot: ΔW after {cycles} cycles')
        cbar = plt.colorbar(im)
        cbar.set_label('ΔW (positive: invasion; negative: failure)')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, cont_fname))
        plt.show()

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

    category = np.zeros_like(deltaW_matrix, dtype=int)
    category[deltaW_matrix <  0] = 0
    category[deltaW_matrix == 0] = 1
    category[deltaW_matrix >  0] = 2

    # 2. Create a colormap that maps:
    #    0 → white, 1 → black, 2 → 0.5 gray
    cmap = ListedColormap(['white', 'black', 'gray'])

    # 3. Plot with imshow, ensuring no interpolation and correct extent:
    plt.figure(figsize=(8, 8))
    plt.imshow(
        category,
        origin='lower',
        extent=[U_min, U_max, X_min, X_max],
        aspect='auto',
        cmap=cmap,
        vmin=0, vmax=2
    )
    plt.xlabel('Resident Seedbank Rate')
    plt.ylabel('Mutant Seedbank Rate')
    plt.title('Invasion (gray), Extinction (white), Neutral (black)')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, bin_fname))
    plt.show()

    return U_vals, X_vals, deltaW_matrix

def piplot(
    V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size=50,
    U_min=0.01, U_max=0.99,
    X_min=0.01, X_max=0.99, U_out_baseline=.07,
    perturb_W=False, perturb_Y=True, speedplot=False

):
    """
    Generate a pairwise invasion plot (PIP) over U_in (x-axis) and X_in (y-axis).
    Uses X_out = X_in and U_out = U_in for each grid point.
    Only computes for X_in >= U_in and mirrors symmetry.
    Saves both continuous ΔW heatmap and binary invasion/failure plot without overwriting.
    """
    # Prepare output folder
    folder = 'pip2_plots'
    os.makedirs(folder, exist_ok=True)

    U_vals = np.linspace(U_min, U_max, grid_size)
    X_vals = np.linspace(X_min, X_max, grid_size)
    deltaW_matrix = np.zeros((grid_size, grid_size))

    for i, U_in in enumerate(tqdm(U_vals, desc="Scanning U_in")):
        for j, X_in in enumerate(X_vals):
            # Mirror diagonal
            if j == i:
                deltaW_matrix[j, i] = 0.0
                continue
            if X_in < U_in:
                deltaW_matrix[j, i] = -deltaW_matrix[i, j]
                continue

            # Compute ΔW via run_invasion
            deltaW = run_invasion(
             V0, W0, Y0,
               W_birth, Y_birth,
               W_death, Y_death,
               X_in, U_out_baseline,
               U_in, U_out_baseline,
               Z_in, Z_out,
                extinction_rate, dt,
                use_X, use_Z,
                severity,
                cycles,
                perturb_W,
                perturb_Y,
                plot=False
            )
            deltaW_matrix[j, i] = deltaW
    if speedplot:
        # Continuous heatmap: determine next filename index
        cont_pattern = os.path.join(folder, 'pip_speed*.pdf')
        existing_cont = glob.glob(cont_pattern)
        cont_idxs = []
        for p in existing_cont:
            m = re.match(r'.*pip_speed(\d*)\.pdf$', os.path.basename(p))
            if m:
                idx = int(m.group(1)) if m.group(1) else 0
                cont_idxs.append(idx)
        next_cont = max(cont_idxs) + 1 if cont_idxs else 0
        cont_fname = f'pip_speed{next_cont}.pdf'

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            deltaW_matrix,
            origin='lower',
            extent=[U_min, U_max, X_min, X_max],
            aspect='auto'
        )
        plt.xlabel('U_in')
        plt.ylabel('X_in')
        plt.title(f'Pairwise Invasion Plot: ΔW after {cycles} cycles')
        cbar = plt.colorbar(im)
        cbar.set_label('ΔW (positive: invasion; negative: failure)')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, cont_fname))
        plt.show()

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

    category = np.zeros_like(deltaW_matrix, dtype=int)
    category[deltaW_matrix <  0] = 0
    category[deltaW_matrix == 0] = 1
    category[deltaW_matrix >  0] = 2

    # 2. Create a colormap that maps:
    #    0 → white, 1 → black, 2 → 0.5 gray
    cmap = ListedColormap(['white', 'black', 'gray'])

    # 3. Plot with imshow, ensuring no interpolation and correct extent:
    plt.figure(figsize=(8, 8))
    plt.imshow(
        category,
        origin='lower',
        extent=[U_min, U_max, X_min, X_max],
        aspect='auto',
        cmap=cmap,
        vmin=0, vmax=2
    )
    plt.xlabel('Resident Seedbank Rate')
    plt.ylabel('Mutant Seedbank Rate')
    plt.title('Invasion (gray) and Extinction (white) of Mutant')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, bin_fname))
    plt.show()

    return U_vals, X_vals, deltaW_matrix

def piplot2(
    V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size=50,
    U_min=0.01, U_max=0.99,
    X_min=0.01, X_max=0.99, U_in_baseline=.07,
    perturb_W=False, perturb_Y=True, speedplot=False

):
    """
    Generate a pairwise invasion plot (PIP) over U_in (x-axis) and X_in (y-axis).
    Uses X_out = X_in and U_out = U_in for each grid point.
    Only computes for X_in >= U_in and mirrors symmetry.
    Saves both continuous ΔW heatmap and binary invasion/failure plot without overwriting.
    """
    # Prepare output folder
    folder = 'pip3_plots'
    os.makedirs(folder, exist_ok=True)

    U_vals = np.linspace(U_min, U_max, grid_size)
    X_vals = np.linspace(X_min, X_max, grid_size)
    deltaW_matrix = np.zeros((grid_size, grid_size))

    for i, U_out in enumerate(tqdm(U_vals, desc="Scanning U_out")):
        for j, X_out in enumerate(X_vals):
            # Mirror diagonal
            if j == i:
                deltaW_matrix[j, i] = 0.0
                continue
            if X_out < U_out:
                deltaW_matrix[j, i] = -deltaW_matrix[i, j]
                continue

            # Compute ΔW via run_invasion
            deltaW = run_invasion(
             V0, W0, Y0,
               W_birth, Y_birth,
               W_death, Y_death,
               X_out, U_in_baseline,
               U_out, U_in_baseline,
               Z_in, Z_out,
                extinction_rate, dt,
                use_X, use_Z,
                severity,
                cycles,
                perturb_W,
                perturb_Y,
                plot=False
            )
            deltaW_matrix[j, i] = deltaW
    if speedplot:
        # Continuous heatmap: determine next filename index
        cont_pattern = os.path.join(folder, 'pip_speed*.pdf')
        existing_cont = glob.glob(cont_pattern)
        cont_idxs = []
        for p in existing_cont:
            m = re.match(r'.*pip_speed(\d*)\.pdf$', os.path.basename(p))
            if m:
                idx = int(m.group(1)) if m.group(1) else 0
                cont_idxs.append(idx)
        next_cont = max(cont_idxs) + 1 if cont_idxs else 0
        cont_fname = f'pip_speed{next_cont}.pdf'

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            deltaW_matrix,
            origin='lower',
            extent=[U_min, U_max, X_min, X_max],
            aspect='auto'
        )
        plt.xlabel('Resident out')
        plt.ylabel('Mutant out')
        plt.title(f'Pairwise Invasion Plot: ΔW after {cycles} cycles')
        cbar = plt.colorbar(im)
        cbar.set_label('ΔW (positive: invasion; negative: failure)')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, cont_fname))
        plt.show()

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

    category = np.zeros_like(deltaW_matrix, dtype=int)
    category[deltaW_matrix <  0] = 0
    category[deltaW_matrix == 0] = 1
    category[deltaW_matrix >  0] = 2

    # 2. Create a colormap that maps:
    #    0 → white, 1 → black, 2 → 0.5 gray
    cmap = ListedColormap(['white', 'black', 'gray'])

    # 3. Plot with imshow, ensuring no interpolation and correct extent:
    plt.figure(figsize=(8, 8))
    plt.imshow(
        category,
        origin='lower',
        extent=[U_min, U_max, X_min, X_max],
        aspect='auto',
        cmap=cmap,
        vmin=0, vmax=2
    )
    plt.xlabel('Resident Seedbank Rate')
    plt.ylabel('Mutant Seedbank Rate')
    plt.title('Invasion (gray) and Extinction (white) of Mutant')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, bin_fname))
    plt.show()

    return U_vals, X_vals, deltaW_matrix

def global_invasability(
    V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size,
    U_in=0.1, U_out=0.1,
    X_in_range = 0.1,
    X_out_range = 0.1,
    perturb_W=False, perturb_Y=True, speedplot=False, break_threshold=0.01

):
    """
    Generate a pairwise invasion plot (PIP) over U_in (x-axis) and X_in (y-axis).
    Uses X_out = X_in and U_out = U_in for each grid point.
    Only computes for X_in >= U_in and mirrors symmetry.
    Saves both continuous ΔW heatmap and binary invasion/failure plot without overwriting.
    """
    # Prepare output folder
    folder = 'invasion_plots'
    os.makedirs(folder, exist_ok=True)

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
                        plot=False)
            deltaW_matrix[j, i] = deltaW
    # Compute the mesh edges so that each cell of pcolormesh
    # spans exactly between successive sample points.
    X_edges = np.linspace(U_in - X_in_range, U_in + X_in_range, grid_size + 1)
    Y_edges = np.linspace(U_out - X_out_range, U_out + X_out_range, grid_size + 1)

    if speedplot:
        # Determine next index and filename as before...
        cont_pattern = os.path.join(folder, 'pip_speed*.pdf')
        existing_cont = glob.glob(cont_pattern)
        cont_idxs = [int(m.group(1)) if (m := re.match(r'.*pip_speed(\d*)\.pdf$', os.path.basename(p))) else 0
                     for p in existing_cont]
        next_cont = max(cont_idxs) + 1 if cont_idxs else 0
        cont_fname = f'pip_speed{next_cont}.pdf'

        fig, ax = plt.subplots(figsize=(8, 6))

        # Use pcolormesh so that grid cells align with sample points:
        mesh = ax.pcolormesh(
            X_edges, Y_edges, deltaW_matrix,
            shading='flat',      # no smoothing
            cmap='RdBu_r',       # diverging colormap, for example
            vmin=np.min(deltaW_matrix),
            vmax=np.max(deltaW_matrix)
        )

        # Draw gridlines between cells:
        ax.set_xticks(X_edges, minor=True)
        ax.set_yticks(Y_edges, minor=True)
        ax.grid(which='minor', color='black', linewidth=0.5)

        # Place major ticks at the center of each cell:
        ax.set_xticks((X_edges[:-1] + X_edges[1:]) / 2)
        ax.set_yticks((Y_edges[:-1] + Y_edges[1:]) / 2)
        ax.set_xticklabels([f"{x:.2f}" for x in (X_edges[:-1] + X_edges[1:]) / 2], rotation=90)
        ax.set_yticklabels([f"{y:.2f}" for y in (Y_edges[:-1] + Y_edges[1:]) / 2])

        ax.set_xlabel('U_in')
        ax.set_ylabel('X_in')
        ax.set_title(f'Pairwise Invasion Plot: ΔW after {cycles} cycles')
        cbar = fig.colorbar(mesh, ax=ax, label='ΔW (positive: invasion; negative: failure)')

        plt.tight_layout()
        fig.savefig(os.path.join(folder, cont_fname), format='pdf')
        plt.show()

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

def local_invasibility_heatmap(
    V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size=5,
    U_in_min=0.01, U_in_max=0.4,
    U_out_min=0.01, U_out_max=0.4,
    folder='total_invasibility',
):
    """
    For each interior gridpoint (i,j) on [0,1]^2 in U_in, X_in:
      • Evaluate run_invasion at its 8 nearest neighbors
      • Compute mean of their np.sign(deltaW)
      • Plot that in grayscale (1=black…+1=white)
    """

    # 1) Prepare output directory
    os.makedirs(folder, exist_ok=True)

    # 2) Build the [0,1] grid
    U_in_vals = np.linspace(U_in_min, U_in_max, grid_size)
    U_out_vals = np.linspace(U_out_min, U_out_max, grid_size)

    # 3) Initialize the score matrix
    score = np.zeros((grid_size, grid_size))

    # 4) Offsets for the eight neighbors
    neighbor_offsets = [         (-1,0),       
                        ( 0,-1),         ( 0,1),
                                 ( 1,0),       ]
    deltas = {}
    # 5) Loop over interior points
    for i in tqdm(range(1, grid_size-1), desc="Computing local invasibility"):
        for j in range(1, grid_size-1):
            invasions = 0
            for di, dj in neighbor_offsets:
                X_in = U_in_vals[i+di]
                X_out = U_out_vals[j+dj]

                if ((i+di,j+dj), (i,j)) in deltas.keys():
                    if deltas[((i+di,j+dj), (i,j))] == -1:
                        invasions += 1
                else:
                    # Here we keep X_out = X_in and U_out = U_in for simplicity
                    deltaW = run_invasion(

                         V0=V0, W0=W0, Y0=Y0,
                        W_birth=W_birth, Y_birth=Y_birth,
                        W_death=W_death, Y_death=Y_death,
                        X_in=X_in, X_out=X_out,
                        U_in=U_in_vals[i],U_out=U_in_vals[j],
                        Z_in=Z_in, Z_out=Z_out,
                        extinction_rate=extinction_rate, dt=dt,
                        use_X=use_X, use_Z=use_Z,
                        cycles=cycles, severity=severity,
                        perturb_W=False, perturb_Y=True,
                        plot=False,

    
                    )
                    if deltaW> 0:
                        invasions += 1
                        deltas[((i,j), (i+di,j+dj))] = 1
                        deltas[((i+di,j+dj), (i,j))] = -1
                    else:
                        deltas[((i,j), (i+di,j+dj))] = -1
                        deltas[((i+di,j+dj), (i,j))] = 1

                # store the mean sign ∈ [–1,+1]
            score[i, j] = invasions

    # 6) Plot the result as a grayscale PDF
    #    –1 → black; +1 → white
    mask = np.zeros_like(score, dtype=bool)
    mask[ 0, :] = True   # top row
    mask[-1, :] = True   # bottom row
    mask[:,  0] = True   # left column
    mask[:, -1] = True   # right column
    score_masked = np.ma.array(score, mask=mask)

    cmap = plt.get_cmap('Greens', 5)
    cmap.set_bad(color='white') 

    # 2. Create a norm that bins values 0–8 into 9 discrete intervals.
    bounds = np.arange(-0.5, 5.5, 1)     # edges at -0.5, 0.5, 1.5, …, 8.5
    norm   = BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(9,8))
    im = plt.imshow(
        score_masked,
        origin='lower',
        extent=[U_in_min, U_in_max,U_out_min,U_out_max],
        aspect='auto',
        cmap=cmap,
        norm=norm
    )
    plt.xlabel('U_in')
    plt.ylabel('U_out')
    plt.title('Local invasibility (Number if invading neighbors)')
    cbar = plt.colorbar(im, ticks=np.arange(0, 5, 1), boundaries=bounds)
    cbar.set_label('Count of invading neighbors')
    plt.tight_layout()

    # save a new file without overwriting
    existing = sorted([p for p in os.listdir(folder) if p.startswith('local_inv') and p.endswith('.pdf')])
    idx = int(re.search(r'\d+', existing[-1]).group())+1 if existing else 0
    fname = os.path.join(folder, f'local_inv{idx}.pdf')
    plt.savefig(fname)
    plt.show()

    # 7) Return for further use
    return U_in_vals, U_out_vals, score, deltas

def local_invasibility_path(
    V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size=5,
    U_in_min=0.01, U_in_max=0.4,
    U_out_min=0.01, U_out_max=0.4,
    folder='total_invasibility',
):
    """
    For each interior gridpoint (i,j) on [0,1]^2 in U_in, X_in:
      • Evaluate run_invasion at its 8 nearest neighbors
      • Compute mean of their np.sign(deltaW)
      • Plot that in grayscale (1=black…+1=white)
    """

    # 1) Prepare output directory
    os.makedirs(folder, exist_ok=True)

    # 2) Build the [0,1] grid
    U_in_vals = np.linspace(U_in_min, U_in_max, grid_size)
    U_out_vals = np.linspace(U_out_min, U_out_max, grid_size)

    # 3) Initialize the score matrix
    score = np.zeros((grid_size, grid_size))

    # 4) Offsets for the eight neighbors
    neighbor_offsets = [         (1,0),       
                        ( 0,1),         ( 0,-1),
                                 ( -1,0),       ]
    deltas = {}

    steps = 0

    current = (1, 1)
    path_coords = []


    while steps < grid_size:
        steps += 1
        i, j = current
        path_coords.append((U_in_vals[i], U_out_vals[j]))
        for di, dj in neighbor_offsets:
                X_in = U_in_vals[i+di]
                X_out = U_out_vals[j+dj]

                if ((i+di,j+dj), (i,j)) in deltas.keys():
                    if deltas[((i+di,j+dj), (i,j))] == 0:
                        losses += 1
                        current[0] += di
                        current[1] += dj
                        break
                    else:
                        continue   
                else:
                    deltaW = run_invasion(

                         V0=V0, W0=W0, Y0=Y0,
                        W_birth=W_birth, Y_birth=Y_birth,
                        W_death=W_death, Y_death=Y_death,
                        X_in=X_in, X_out=X_out,
                        U_in=U_in_vals[i],U_out=U_in_vals[j],
                        Z_in=Z_in, Z_out=Z_out,
                        extinction_rate=extinction_rate, dt=dt,
                        use_X=use_X, use_Z=use_Z,
                        cycles=cycles, severity=severity,
                        perturb_W=False, perturb_Y=True,
                        plot=False,

    
                    )
                    if deltaW> 0:
                        deltas[((i,j), (i+di,j+dj))] = 1
                        deltas[((i+di,j+dj), (i,j))] = 0
                        current[0] += di
                        current[1] += dj
                        break
                    else:
                        deltas[((i,j), (i+di,j+dj))] = 0
                        deltas[((i+di,j+dj), (i,j))] = 1
                       

    return U_in_vals, U_out_vals, score, deltas








