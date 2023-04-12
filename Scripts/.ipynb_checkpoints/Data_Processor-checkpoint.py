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


def ProcessData(data_path, 
                t_end_idx=None, 
                load_noisy=False, 
                plot=False, 
                save_path=None):
    
    data = np.load(data_path, allow_pickle=True).item()
    
    x = data['x']
    t = data['t']
    
    if load_noisy:
        U = data['u_noise']
    else:
        U = data['u_true']
        
    if t_end_idx is not None:
        t = t[:t_end_idx+1]
        U = U[:,:t_end_idx+1]
    
    X, T = np.meshgrid(x, t, indexing='ij')
    shape = U.shape
    
    # flatten for MLP
    inputs = np.concatenate([X.reshape(-1)[:, None],
                             T.reshape(-1)[:, None]], axis=1)
    outputs = U.reshape(-1)[:, None]
    
    if plot:
        # plot surface
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X, T, U, cmap=cm.coolwarm, alpha=1)
        ax.scatter(X.reshape(-1), T.reshape(-1), U.reshape(-1), s=5, c='k')
        plt.title('Data')
        ax.set_xlabel('Position')
        ax.set_ylabel('Time')
        ax.set_zlabel('Cell density')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        if save_path is not None:
            plt.savefig(save_path,bbox_inches='tight')
        plt.show()
    
    return inputs, outputs, X, T, U, shape, data
