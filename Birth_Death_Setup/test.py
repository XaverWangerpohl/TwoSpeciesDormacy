import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import re

from sbsize import (compute_equilibrium,
                       simulate_segment,
                       plot_segment,
                       run_invasion,
                       global_invasability,
                       local_invasibility,
                       plot_segment_deriv,
                       piplot,
                       piplot_pair
                       
                       )

W_birth = 0.4
W_death = 0.1
Y_birth = 0.9
Y_death = 0.15


U_in= 0.2
U_size = 0.1
X_in = 1
X_size = 10
Z_in = 0.1
Z_size = 0.1

Time = 400.0

use_X = True
use_Z = False

num_points = 200
severity = 0.8
cycles = 2000
extinction_rate = 100


# calculate starting values

W,Y = compute_equilibrium(W_birth, W_death, Y_birth, Y_death)
W0 = W/2

V0 = W - W0

Y0 = Y
X0 = W0 * X_size
U0 = V0 * U_size
Z0 = Y0 * Z_size

dt = 0.01

U_size_baseline = 10
X_in_min = .01

X_in_max = .2

U_in_min = .01
U_in_max = .2

if __name__ == '__main__':

    # test of run_segment function

    #plot_segment(U0=U0, V0=V0, W0=W0, X0=X0, Y0=Y0, Z0=Z0,W_birth=W_birth, Y_birth=Y_birth,W_death=W_death, Y_death=Y_death,X_in=X_in, X_size=X_size,U_in=U_in, U_size=U_size, Z_in=Z_in, Z_size=Z_size,Time=Time, dt=dt,use_X=True, use_Z=False,severity=0.5,perturb_W=True, perturb_Y=False,perturb_time=20.0,tol=1e-7)

    # test of run invasion function

    U_vals, X_vals, m1, m2, out = piplot_pair(V0, W0, Y0, W_birth, Y_birth, W_death, Y_death, Z_in, Z_size, extinction_rate, dt, use_X, use_Z, cycles, severity, grid_size=4, U_in_min=0.01, U_in_max=0.99, X_in_min=0.01, X_in_max=0.99, U_size_pair=(1, 10), perturb_W=False, perturb_Y=True, mode='binary')

    #piplot(V0=V0, W0=W0,Y0=Y0,W_birth=W_birth, Y_birth=Y_birth,W_death=W_death, Y_death=Y_death,X_in_min=X_in_min,X_in_max=X_in_max,U_in_min=U_in_min,U_in_max=U_in_max, U_size_baseline=U_size_baseline, Z_in=Z_in, Z_size=Z_size,cycles=cycles,extinction_rate=extinction_rate, use_X=use_X, use_Z=use_Z,severity=severity,perturb_W=False, perturb_Y=True, dt=dt, grid_size=4)

    #run_invasion(V0, W0, Y0, W_birth, Y_birth, W_death, Y_death, X_in, X_out,U_in, U_out, Z_in, Z_out,extinction_rate, dt,use_X, use_Z,severity,cycles=1000,perturb_W=False,perturb_Y=True ,plot=True,stop=None,break_threshold=np.inf)

    # test of global fitness (adnust grid size for bigger picture)

    #global_invasability(V0, W0, Y0, W_birth, Y_birth, W_death, Y_death,Z_in, Z_out,extinction_rate,dt,use_X, use_Z,cycles, severity,grid_size = 8,U_in=0.1, U_out=0.1,X_in_range = 0.1,X_out_range = 0.1,perturb_W=False, perturb_Y=True, break_threshold=0.01)

    # test of local fitness

    #local_invasibility(V0, W0, Y0,W_birth, Y_birth, W_death, Y_death,Z_in, Z_out,extinction_rate, dt,use_X, use_Z,cycles, severity,grid_size=10,U_in_min  = 0.01,U_in_max  = 0.3, U_out_min  = 0.01, U_out_max  = 0.5)