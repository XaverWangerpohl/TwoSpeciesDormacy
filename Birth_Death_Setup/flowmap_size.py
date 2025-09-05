#!/usr/bin/env python3
"""
Generate paired flow maps (side-by-side) on (U_in, ξ_W) where ξ_W is the seedbank size,
for W extinction and Y extinction, using utilities from sbsize.py.

Baseline parameters are chosen to match the usual values; running this script
produces surface_flow/flow_map_pair.pdf with the Y-axis labeled as ξ_W.
"""

import os
import numpy as np
from sbsize import (
    compute_equilibrium,
    flow_map_size,
)


def main():
    # Baseline parameters
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

    title = r'Flow Fields on $(U^{in}, \\xi_W)$: $W^a$ vs $Y$ extinction'
    out_pdf = flow_map_size(
        V0, W0, Y0,
        W_birth, Y_birth, W_death, Y_death,
        Z_in, Z_size,
        extinction_rate, dt,
        use_X, use_Z,
        cycles, severity,
        grid_size=7,
        U_in_min=0.01, U_in_max=0.4,
        U_size_min=0.01, U_size_max=0.4,
        folder='surface_flow',
        break_threshold=0.01,
        arrow_scale=20,
        figure_title=title,
    )
    print(f"Saved size-parameter flow map to: {out_pdf}")


if __name__ == '__main__':
    main()

