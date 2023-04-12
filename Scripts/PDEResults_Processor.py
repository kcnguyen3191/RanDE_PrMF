import sys, importlib
# "../" to go back one director
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from scipy import interpolate


def Process_PDEResults(load_folder, num_PDE, num_run):
    
    total_time = 0
    best_error = 1e64
    
    for irun in range(num_run):
        path = load_folder+'/PDEFitting_N01_'+str(num_PDE)+'P/PDEFitting_N01_R'+str(irun+1).zfill(2)+'.npy'
        PDE_results = np.load(path, allow_pickle=True).item()
    
        total_time += PDE_results['elapsed_time']

        if PDE_results['error'] < best_error:
            best_error = PDE_results['error']
            optimized_params = PDE_results['optimized_params']
            
    return optimized_params, best_error, total_time/3600
