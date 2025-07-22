import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import re

from functions_death import (compute_equilibrium,
                       simulate_segment,
                       plot_segment,
                       run_invasion,
                       global_invasability,
                       local_invasibility,
                       )

W_birth = 0.7
W_death = 0.1
Y_birth = 0.9
Y_death = 0.15

X_death = 0.0000001


U_in= 0.2
U_out = 0.1
X_in = 1
X_out = 0.1
Z_in = 0.1
Z_out = 0.1

Time = 1000.0
dt = 0.1

use_X = True
use_Z = False

num_points = 100
severity = 0.8
cycles = 2000
extinction_rate = 200
U_in, U_out = (10, 1)
X_in, X_out = (1,0.001)

# calculate starting values

W_death_eq = W_death + (X_in*X_death) / (X_out + X_death)

V_death_eq = W_death + (U_in*X_death) / (U_out + X_death)

W,Y = compute_equilibrium(W_birth, W_death_eq, Y_birth, Y_death)
V,Y = compute_equilibrium(W_birth, V_death_eq, Y_birth, Y_death)


start_val = .5

W0 = W * start_val

V0 = V * (1-start_val)

Y0 = Y
X0 = W0 / ((X_out + X_death) / X_in)
U0 = V0 / ((U_out + X_death) / U_in)
Z0 = Y0 / (Z_out /Z_in)

dt = 0.1


if __name__ == '__main__':
    print(W_death_eq,V_death_eq)

    # test of run_segment function

    #plot_segment(U0=U0, V0=V0, W0=W0, X0=X0, Y0=Y0, Z0=Z0, W_birth=W_birth, Y_birth=Y_birth, W_death=W_death, Y_death=Y_death,X_in=X_in, X_out=X_out, U_in=U_in, U_out=U_out, Z_in=Z_in, Z_out=Z_out, X_death=X_death,Time=200.0, dt=0.1,use_X=True, use_Z=False,severity=0.5,perturb_W=False, perturb_Y=True,perturb_time=200.0,tol=1e-7)

    # test of run invasion function


    run_invasion(V0, W0, Y0, W_birth, Y_birth, W_death, Y_death, X_in, X_out,U_in, U_out, Z_in, Z_out,X_death, extinction_rate, dt,use_X, use_Z,severity,cycles=1000,perturb_W=False,perturb_Y=True ,plot=True,stop=None,break_threshold=1)

    # test of global fitness (adnust grid size for bigger picture)

    #global_invasability(V0, W0, Y0, W_birth, Y_birth, W_death, Y_death,Z_in, Z_out,extinction_rate,dt,use_X, use_Z,cycles, severity,grid_size = 8,U_in=0.1, U_out=0.1,X_in_range = 0.1,X_out_range = 0.1,perturb_W=False, perturb_Y=True, break_threshold=0.01)

    # test of local fitness

    #local_invasibility(V0, W0, Y0,W_birth, Y_birth, W_death, Y_death,Z_in, Z_out,extinction_rate, dt,use_X, use_Z,cycles, severity,grid_size=10,U_in_min  = 0.01,U_in_max  = 0.3, U_out_min  = 0.01, U_out_max  = 0.5)