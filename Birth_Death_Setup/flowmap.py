#!/usr/bin/env python3
"""
Generate paired flow maps (side-by-side) for W extinction and Y extinction
using the local_invasibility sign reconstruction in the Birth_Death_Setup.

Baseline parameters are chosen to match the usual values seen in the repo.
You can run this script directly to produce surface_flow/flow_map_pair.pdf.
"""

import os
import numpy as np
from sbsize import (
    compute_equilibrium,
    local_invasibility,
    flow_map_pair,
)


def build_signs_for_mode(
    V0, W0, Y0,
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_size,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size=7,
    U_in_min=0.01, U_in_max=0.4,
    U_size_min=0.01, U_size_max=0.4,
    perturb_W=False, perturb_Y=True,
):
    """
    Compute local invasibility on a small grid to obtain the sign dictionary
    for the requested perturbation mode (W or Y), then return (signs, x_vals, y_vals).
    """
    U_in_vals, U_size_vals, _score, deltas, _args = local_invasibility(
        V0, W0, Y0,
        W_birth, Y_birth, W_death, Y_death,
        Z_in, Z_size,
        extinction_rate, dt,
        use_X, use_Z,
        cycles, severity,
        grid_size=grid_size,
        U_in_min=U_in_min, U_in_max=U_in_max,
        U_size_min=U_size_min, U_size_max=U_size_max,
        folder='local_invasibility',
        break_threshold=0.01,
    )
    # deltas already maps edge pairs to ±1
    return deltas, U_in_vals, U_size_vals


def main():
    # Baseline parameters (mirroring typical values in test.py)
    W_birth, W_death = 0.4, 0.1
    Y_birth, Y_death = 0.9, 0.15

    use_X, use_Z = True, False
    Z_in, Z_size = 0.1, 0.1
    dt = 0.01
    Time = 400.0
    cycles = 2000
    severity = 0.8
    extinction_rate = 100

    # Initial conditions from equilibrium slice
    W_eq, Y_eq = compute_equilibrium(W_birth, W_death, Y_birth, Y_death)
    W0 = W_eq / 2.0
    V0 = W_eq - W0
    Y0 = Y_eq

    # Build sign maps for Y-extinction and W-extinction
    signs_Y, x_vals, y_vals = build_signs_for_mode(
        V0, W0, Y0,
        W_birth, Y_birth, W_death, Y_death,
        Z_in, Z_size,
        extinction_rate, dt,
        use_X, use_Z,
        cycles, severity,
        grid_size=7,
        U_in_min=0.01, U_in_max=0.4,
        U_size_min=0.01, U_size_max=0.4,
        perturb_W=False, perturb_Y=True,
    )

    signs_W, _, _ = build_signs_for_mode(
        V0, W0, Y0,
        W_birth, Y_birth, W_death, Y_death,
        Z_in, Z_size,
        extinction_rate, dt,
        use_X, use_Z,
        cycles, severity,
        grid_size=5,
        U_in_min=0.01, U_in_max=0.4,
        U_size_min=0.01, U_size_max=0.4,
        perturb_W=True, perturb_Y=False,
    )

    # Compose figure title and create paired flow map
    title = r'Flow Fields: $W^a$ vs $Y$ extinction'
    out_pdf = flow_map_pair(
        signs_W, signs_Y,
        x_vals, y_vals,
        folder='surface_flow',
        arrow_scale=20,
        invert=False,
        titles=(r'$W^a$ extinction', r'$Y$ extinction'),
        figure_title=title,
    )
    print(f"Saved paired flow map to: {out_pdf}")


if __name__ == '__main__':
    main()

