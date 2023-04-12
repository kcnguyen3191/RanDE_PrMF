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
import seaborn as sns
from scipy import interpolate
import pandas as pd

def Plot_PDF(param1_mesh, param2_mesh, weights_mesh, 
             x_label, y_label, title,
             map_type='contourf',
             save_path=None):
    
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(2, 1, 1)
    # plot heatmap of residuals
    
    if map_type == 'contourf':
        num_param1 = len(np.unique(param1_mesh))
        num_param2 = len(np.unique(param2_mesh))
        
        ax.contourf(param1_mesh.reshape((num_param1,num_param2)), param2_mesh.reshape((num_param1,num_param2)), weights_mesh.reshape((num_param1,num_param2)))
    else:
        param1_min = np.min(np.unique(param1_mesh))
        param1_max = np.max(np.unique(param1_mesh))
        param2_min = np.min(np.unique(param2_mesh))
        param2_max = np.max(np.unique(param2_mesh))

        param1_mesh = np.round_(param1_mesh, decimals=4)
        param2_mesh = np.round_(param2_mesh, decimals=4)

        df = {x_label.split(" ")[0]: param1_mesh.reshape((1,-1))[0],
              y_label.split(" ")[0]: param2_mesh.reshape((1,-1))[0],
              'Weight':weights_mesh.reshape((1,-1))[0]}
        df = pd.DataFrame(df, columns=[x_label.split(" ")[0],
                                       y_label.split(" ")[0],
                                       'Weight'])
        df = df.pivot(y_label.split(" ")[0], x_label.split(" ")[0], 'Weight')
        
        
        ax = sns.heatmap(df, cbar=False, rasterized=True)
        ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=20)
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    
    return fig, ax

def Plot_SurfaceFitting(X, T, U_fit, U_data,
                        title=None,
                        error=None,
                        time=None,
                        save_path=None):
    
    # giving a title to my graph 
    if title is not None:
        title = title
    else:
        title = 'Fitting Result'
    
    if error is not None:
        title += ' \n '
        title += f'Error = {error:.{4}} '
    if time is not None:
        title += ' \n '
        title += f'Time = {time:.{2}} (hours)'
    
    # plot surface
    fig = plt.figure()#(figsize=(11,11))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, T, U_fit, cmap=cm.coolwarm, alpha=1)
    ax.scatter(X.reshape(-1), T.reshape(-1), U_data.reshape(-1), s=5, c='k')
    plt.title(title, fontsize=20)
    ax.set_xlabel('Position', fontsize=14)
    ax.set_ylabel('Time', fontsize=14)
    ax.set_zlabel('Cell density', fontsize=14)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad=2)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    
    return None

def Plot_UvsX(X, T, U_fit, U_data, 
              t_plot_idcs=None,
              title=None,
              error=None,
              time=None,
              legend_list=None,
              save_path=None):
    
    x = np.unique(X)
    t = np.unique(T)
    
    # giving a title to my graph 
    if title is not None:
        title = title
    else:
        title = 'Fitting Result'
    
    if error is not None:
        title += ' \n '
        title += f'Error = {error:.{4}} '
    if time is not None:
        title += ' \n '
        title += f'Time = {time:.{2}} (hours)'
    
    if t_plot_idcs is None:
        t_plot_idcs = np.arange(0, len(t)+1, 5)
    
    if legend_list is None:
        legend_list = []
        for iL in t_plot_idcs:
            time_val = f't= {t[iL]:.{2}}'
            legend_list.append(time_val)
            
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ['x', 'o', 's', 'd', '^', 'p', '1']
    
    plt.figure()#(figsize=(14.6,7))
    for t_idx in t_plot_idcs:
        plt.plot(x, U_fit[:,t_idx], '-', c=colors[t_idx//10], linewidth=2)
    for t_idx in t_plot_idcs:
        plt.plot(x, U_data[:,t_idx], '--', c=colors[t_idx//10], linewidth=2)
#         plt.plot(x, U_data[:,t_idx], marker=markers[t_idx//5], c=colors[t_idx//5], linestyle='--')
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Cell density', fontsize=14)
    plt.legend(legend_list)
    plt.grid()
    plt.title(title, fontsize=20)
    
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    
    return None

def Plot_246PDE(param1_mesh, param2_mesh, weights_mesh, 
                param1_2PDE, param2_2PDE,
                param1_4PDE, param2_4PDE,
                param1_6PDE, param2_6PDE,
                x_label, y_label, title,
                save_path=None):
    
    # plot heatmap of residuals
    fig = plt.figure()#(figsize=(11,11))
    ax = fig.add_subplot(2, 1, 1)
    # plot heatmap of residuals
    ax.contourf(param1_mesh, param2_mesh, weights_mesh)
    ax.scatter(param1_2PDE, param2_2PDE, c ="slategrey",
            linewidths = 0.5,
            marker ="s",
            edgecolor ="k",
            s = 150)
    ax.scatter(param1_4PDE, param2_4PDE, c ="forestgreen",
                linewidths = 0.5,
                marker ="^",
                edgecolor ="k",
                s = 150)
    ax.scatter(param1_6PDE, param2_6PDE, c ="blueviolet",
                linewidths = 0.5,
                marker ="o",
                edgecolor ="k",
                s = 150)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.legend(['2-PDE', '4-PDE', '6-PDE'],loc='upper right')
    ax.set_title(title, fontsize=20)
    
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    
    return None

def Plot_ErrorComparison(nums_PDEs,
                         PDE_errors,
                         PMF_error,
                         figure_mods,
                         fig=None,
                         ax=None,
                         title=None,
                         save_path=None):
    
    # giving a title to my graph 
    if title is not None:
        title = title
    else:
        title = 'Error Result'
        
    if (fig is None) or (ax is None):
        fig = plt.figure(figsize=(11,11))
        ax = fig.add_subplot(3, 1, 2)
        
    # make a plot
    ax.plot(nums_PDEs,
            PDE_errors,
            color=figure_mods['color'], 
            marker=figure_mods['marker'])
    ax.axhline(y = PMF_error, 
               color = figure_mods['color'], 
               linestyle='dashed')
    ax.set_xlabel("Number of PDEs", fontsize = 14)
    ax.set_xticks(nums_PDEs)
    ax.set_ylabel(figure_mods['ylabel'],
                  #color=figure_mods['color'],
                  fontsize=14)
    ax.set_title(title)
    
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    
    return fig, ax

def Plot_WaveSpeedComparison(densities,
                             data_wave_speed,
                             PMF_wave_speed,
                             PDE2_wave_speed,
                             PDE4_wave_speed,
                             PDE6_wave_speed,
                             ylim=None,
                             title=None,
                             save_path=None):
    
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(3, 1, 1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ['x', 'o', 's', 'd', '^']

    # make a plot
    ax.plot(densities,
            data_wave_speed,
            color=colors[0],
            marker=markers[0])
    ax.plot(densities,
            PMF_wave_speed,
            color=colors[1],
            marker=markers[1])
    ax.plot(densities,
            PDE2_wave_speed,
            color=colors[2],
            marker=markers[2])
    ax.plot(densities,
            PDE4_wave_speed,
            color=colors[3],
            marker=markers[3])
    ax.plot(densities,
            PDE6_wave_speed,
            color=colors[4],
            marker=markers[4])
    ax.set_xlabel("Density", fontsize = 14)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(['True','RanDE','2-PDE', '4-PDE', '6-PDE'],loc='upper right')
    # set y-axis label
    ax.set_ylabel("Wave speed",
                  fontsize=14)
    ax.set_title(title, fontsize=20)
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    
    return fig, ax


def Plot_CentersOnPDF(PDF_fig, PDF_ax,
                      param1_center,
                      param2_center,
                      param1_vec,
                      param2_vec,
                      figure_mods,
                      save_path=None):
    
#     param1_vec = np.unique(param1_vec)
#     param2_vec = np.unique(param2_vec)
    
#     param1_vec = np.sort(param1_vec)
#     param2_vec = np.sort(param2_vec)
    
#     delta_param1 = param1_vec[1]-param1_vec[0]
#     delta_param2 = param2_vec[1]-param2_vec[0]
    
#     param1_center = (param1_center - np.min(param1_vec))/(np.max(param1_vec) - np.min(param1_vec))*len(param1_vec)
#     param2_center = (param2_center - np.min(param2_vec))/(np.max(param2_vec) - np.min(param2_vec))*len(param2_vec)
    
    # adjusted centers
    #delta_param1 = (delta_param1 - np.min(param1_vec))/(np.max(param1_vec) - np.min(param1_vec))*len(param1_vec)
    #delta_param2 = (delta_param2 - np.min(param2_vec))/(np.max(param2_vec) - np.min(param2_vec))*len(param2_vec)
    
    #param1_center = param1_center + 0.5*delta_param1
    #param2_center = param2_center + 0.5*delta_param2
    
    PDF_ax.scatter(param1_center, param2_center,c=figure_mods['color'],
            linewidths = figure_mods['linewidths'],
            marker =figure_mods['marker'],
            edgecolor =figure_mods['edgecolor'],
            s = figure_mods['size'])
            
#     PDF_ax.invert_yaxis()

    if figure_mods['legend_list'] is not None:
        PDF_ax.legend(figure_mods['legend_list'],loc='upper right')
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    
    return PDF_fig, PDF_ax