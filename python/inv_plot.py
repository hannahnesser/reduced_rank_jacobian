import yaml
import numpy as np
import xarray as xr 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy

# matplotlib.rcParams['font.serif'] = "Cambria Math"
# matplotlib.rcParams['font.family'] = "serif"

AVOGADRO   = 6.02214129 * 10**(23)  #molecules/mol
CH4_MOLMASS   = 16.04 #g/mol

def cm_transparent_at_zero(cmap):
    my_cmap = plt.cm.get_cmap(cmap,lut=3000)
    my_cmap._init()
    slopen = 200
    alphas_slope = np.abs(np.linspace(0, 1.0, slopen))
    alphas_stable = np.ones(3003-slopen)
    alphas = np.concatenate((alphas_slope, alphas_stable))
    my_cmap._lut[:,-1] = alphas
    my_cmap.set_under('gray', alpha=1)
    return my_cmap

def cm_transparent_at_one(cmap='RdBu_r'):
    my_cmap = plt.cm.get_cmap(cmap,lut=3000)
    my_cmap._init()
    slopen = 1501
    alphas_slope = np.abs(np.linspace(-50, 0, slopen))
    alphas_stable = np.abs(np.linspace(0, 50, 3003-slopen))
    alphas = np.concatenate((alphas_slope, alphas_stable))
    alphas[alphas>1] = 1
    my_cmap._lut[:,-1] = alphas
    return my_cmap

def match_data_to_clusters(data, clusters, default_value=0):
    result = clusters.copy()
    c_array = result.values
    c_idx = np.where(c_array > 0)
    c_val = c_array[c_idx]
    row_idx = [r for _, r, _ in sorted(zip(c_val, c_idx[0], c_idx[1]))]
    col_idx = [c for _, _, c in sorted(zip(c_val, c_idx[0], c_idx[1]))]
    idx = (row_idx, col_idx)

    d_idx = np.where(c_array == 0)

    c_array[c_idx] = data
    c_array[d_idx] = default_value
    result.values = c_array

    return result

def plot_state(state_vector, clusters, ax=None, default_value=0, **plot_options):
    state_data = match_data_to_clusters(state_vector, clusters, default_value)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(20,8),
                               subplot_kw={'projection': ccrs.PlateCarree()})

    if 'cmap' not in plot_options.keys():
        if np.any(state_vector < 0):
            plot_options = {'cmap' : cm_transparent_at_one('RdBu_r'),
                            'vmin' : 0,
                            'vmax' : 2,
                            'cbar_kwargs' : {'label' : 'Scaling Factors'}}
        else:
            state_data = state_data/float(10**6 * AVOGADRO) * (366 * 24 * 60 * 60) * CH4_MOLMASS * float(1e10)
            plot_options = {'cmap' : cm_transparent_at_zero('plasma'),
                            'vmin' : -10**-5,
                            'vmax' : 20,
                            'cbar_kwargs' : {'label' : 'CH$_4$ emissions (Mg a$^{-1}$ km$^{-2}$)'}}
        
    state_data.plot(ax=ax, 
                    snap=True,
                    **plot_options)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='0.98')
    ax.add_feature(cartopy.feature.LAND, facecolor='0.98')
    ax.coastlines(color='grey')
    ax.gridlines(linestyle=':', draw_labels=True, color='grey')

    return ax

def plot_evecs(evecs, clusters, labels=None, n=None):
    # look at plots of eigenvectors
    if n is None:
        n = 2 # number of eigenvectors
    else:
        n = n/2

    fig, ax = plt.subplots(2,n,figsize=(2*n*5.5,2*6.5),
                           subplot_kw={'projection' : ccrs.PlateCarree()})
    cax = fig.add_axes([0.95, 0.25/2, 0.01, 0.75])
    j = 0
    plot_options_evecs = {'cmap' : 'RdBu_r',
                          'vmin' : -0.1,
                          'vmax' : 0.1,
                          'add_colorbar' : False}
    #                       'cbar_kwargs' : {'label' : 'Eigenvector Value'}}

    if labels is None:
        labels = ['Eigenvector 1', 'Eigenvector 2',
                  'Eigenvector 3', 'Eigenvector 4']

    for i, axis in enumerate(ax.flatten()):
        if i == (n-1):
            pos_evecs = plot_options_evecs.copy()
            del(pos_evecs['add_colorbar'])
            pos_evecs['cbar_ax'] = cax
            pos_evecs['cbar_kwargs'] = {'ticks': [pos_evecs['vmin'], 0, pos_evecs['vmax']]}
            cax.tick_params(labelsize=16)
        else:
            pos_evecs = plot_options_evecs.copy()

        axis = plot_state(evecs[:,i], clusters, ax=axis, **pos_evecs)
        axis.set_title(labels[i], y = 1.07, fontsize=24)
    #         ax[j,i].text(-,0,rf)
    
    return fig, ax

def plot_eval_spectra(evals_pph=None, evals_pi=None, **kw):
    if evals_pph is not None:
        evals = evals_pph/(1+evals_pph)
    elif evals_pi is not None:
        evals = evals_pi
    else:
        print('Incorrect arguments supplied.')

    if type(evals) != list:
        evals = [evals]

    fig, ax = kw.pop('figax', plt.subplots(figsize=(10, 5)))
    labels = kw.pop('labels', [i for i in range(len(evals))])
    cmap = kw.pop('cmap', plt.cm.get_cmap('inferno', lut=2*len(evals)))
    ls = kw.pop('ls', ['--' for i in range(len(evals))])

    ax.set_facecolor('0.98')

    for i, ev in enumerate(evals):
        ax.plot(ev, label=labels[i], c=cmap(2*i), ls=ls)

    ax.legend(frameon=False, fontsize=22)
    ax.set_xlabel('Eigenvalue Index', fontsize=22)
    ax.set_ylabel(r'Q$_{DOF}$ Eigenvalues', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=18)

    return fig, ax

# def plot_evecs_compare()


