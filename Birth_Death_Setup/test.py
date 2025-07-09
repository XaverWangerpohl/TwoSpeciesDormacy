import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import re

from functions import (compute_equilibrium,
                       simulate_segment,
                       plot_segment,
                       run_invasion,
                       global_invasability,
                       local_invasibility,
                       )

W_birth = 0.4
W_death = 0.1
Y_birth = 0.9
Y_death = 0.15


U_in= 0.2
U_out = 0.1
X_in = 0.1
X_out = 0.2
Z_in = 0.1
Z_out = 0.1

Time = 600.0
dt = 0.1

use_X = True
use_Z = False

num_points = 100
severity = 0.5
cycles = 2000
extinction_rate = 50

W,Y = compute_equilibrium(W_birth, W_death, Y_birth, Y_death)
W0 = W/2

V0 = W - W0

Y0 = Y
X0 = W0 / (X_out / X_in)
U0 = V0 / (U_out / U_in)
Z0 = Y0 / (Z_out /Z_in)

dt = 0.1


if __name__ == '__main__':

    run_invasion(V0, W0, Y0,
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
               plot=True,
               stop=None,
               break_threshold=0.5)

'''
    global_invasability(V0, W0, Y0, 
    W_birth, Y_birth, W_death, Y_death,
    Z_in, Z_out,
    extinction_rate, dt,
    use_X, use_Z,
    cycles, severity,
    grid_size = 15,
    U_in=0.1, U_out=0.1,
    X_in_range = 0.1,
    X_out_range = 0.1,
    perturb_W=False, perturb_Y=True, break_threshold=0.01

     )
'''


'''
    plot_segment(U0, V0, W0, X0, Y0, Z0,
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
              tol=1e-7)


    '''
'''
    local_invasibility(
        V0, W0, Y0,
        W_birth, Y_birth, W_death, Y_death,
        Z_in, Z_out,
        extinction_rate, dt,
        use_X, use_Z,
        cycles, severity,
        grid_size=10,
        U_in_min  = 0.01,
        U_in_max  = 0.5, 
        U_out_min  = 0.01, 
        U_out_max  = 0.5

    )
    '''

















