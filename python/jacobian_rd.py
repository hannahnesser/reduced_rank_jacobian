import copy
import os
from os.path import join
import sys
import math
import itertools

sys.path.append('./python/')
import inversion as inv
import jacobian as j

import xarray as xr
import numpy as np

import pandas as pd
from scipy.sparse import diags, identity
from scipy import linalg, stats

from matplotlib import colorbar, colors, rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import matplotlib.table as tbl
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.patch import geos_to_path

sys.path.append('../Python/')
import format_plots as fp
import config

#########################
### PLOTTING DEFAULTS ###
#########################

# rcParams default
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.size'] = config.LABEL_FONTSIZE*config.SCALE
rcParams['text.usetex'] = True
# rcParams['mathtext.fontset'] = 'stixsans'
rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

# Colormaps
c = plt.cm.get_cmap('inferno', lut=10)
plasma_trans = fp.cmap_trans('plasma')
plasma_trans_r = fp.cmap_trans('plasma_r')

# Small (i.e. non-default) figure settings
small_fig_kwargs = {'max_width' : config.BASE_WIDTH,
                    'max_height' : config.BASE_HEIGHT}
small_map_kwargs = {'draw_labels' : False}

######################
### FILE LOCATIONS ###
######################
main = '/Users/hannahnesser/Documents/Harvard/Research/Reduced_Rank_Jacobian'
plots = join(main, 'plots')
code = join(main, 'python')
inputs = join(main, 'input')

#################
### LOAD DATA ###
#################

# Import clusters
clusters = xr.open_dataarray(join(inputs, 'clusters_1x125.nc'))
clusters_plot = xr.open_dataarray(join(inputs, 'clusters_1x125_plot.nc'))

# Load estimated and true Jacobian
k_est = xr.open_dataarray(join(inputs, 'k_est.nc'))
k_est_sparse = xr.open_dataarray(join(inputs, 'k_est_sparse.nc'))
k_true = xr.open_dataarray(join(inputs, 'k_true.nc'))

# Load prior and error
xa = xr.open_dataarray(join(inputs, 'xa.nc'))
xa_abs = xr.open_dataarray(join(inputs, 'xa_abs.nc'))
# sa_vec = xr.open_dataarray(join(inputs, 'sa_vec.nc'))
sa_vec = 0.25*np.ones(xa.shape[0])

# Load observations and error
y = xr.open_dataarray(join(inputs, 'y.nc'))
y_base = xr.open_dataarray(join(inputs, 'y_base.nc'))
so_vec = xr.open_dataarray(join(inputs, 'so_vec.nc'))

# Load the gridded observations
obs = pd.read_csv(join(inputs, 'sat_obs.gosat.00.m'),
                  delim_whitespace=True,
                  header=0)
obs['GOSAT'] *= 1e9
obs['model'] *= 1e9
obs = obs[obs['GLINT'] == False]
obs = obs[['NNN', 'LON', 'LAT', 'GOSAT', 'model']]
obs = obs.rename(columns={'LON' : 'lon',
                          'LAT' : 'lat',
                          'NNN' : 'Nobs'})

#####################
### SET CONSTANTS ###
#####################

RF = 5
threshold = 0

############
### TRUE ###
############

# Create a true Reduced Rank Jacobian object
true = inv.ReducedRankJacobian(k_true.values,
                               xa.values,
                               sa_vec,
                               y.values,
                               y_base.values,
                               so_vec.values)
true.model_runs = true.nstate
true.xa_abs = xa_abs*1e3
true.rf = RF

# Complete an eigendecomposition of the prior pre-
# conditioned Hessian, filling in the eigenvalue
# and eigenvector attributes of true.
true.edecomp()

# Solve the inversion, too.
true.solve_inversion()

########################
### INITIAL ESTIMATE ###
########################

est0 = inv.ReducedRankJacobian(k_est.values,
                               xa.values,
                               sa_vec,
                               y.values,
                               y_base.values,
                               so_vec.values)
est0.xa_abs = xa_abs*1e3
est0.rf = RF
est0.edecomp()
est0.solve_inversion()

########################################
### BUILD REDUCED DIMENSION JACOBIAN ###
########################################

# Reduced dimension
fig, ax = fp.get_figax(aspect=4, rows=2, cols=1)
# plt.subplots_adjust(hspace=1)
ax[0] = fp.add_labels(ax[0],
                      xlabel='Number of Native-Resolution Grid Cells',
                      ylabel='DOFS per Cluster')

est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                  clusters_plot,
                                  n_cells=[2098],
                                  n_cluster_size=[2098])

## Initialize the variables that we will update throughout our while loop
# A record of the number of state vector elements in the grid
nstate_record = [est1_ms.nstate]

# A record of the DOFS per cluster
dpc_record = [est1_ms.dofs.sum()/est1_ms.nstate]
dofs_record = [est1_ms.dofs.sum()]

# A record of the cluster size and the number of native-resolution grid
# cells that are used for that cluster size
cell_base = 25
cluster_size = [1]
cells = [cell_base]

# A record of the state vector indices at which the cluster size increases
state_thresholds = [0]

# Initialize our while loop variables
dpc_check = True
size_count = 1
iteration_count = 1

# While the DOFS per cluster are less than the threshold set at the top
# of this script
while dpc_check:
    print('-'*100)
    print('Iteration : ', iteration_count)
    print('Max. Cluster Size: ', cluster_size[-1])

    # Update the state vector, construct the Jacobian, and solve the
    # inversion
    est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                       clusters_plot,
                                       n_cells=cells,
                                       n_cluster_size=cluster_size)

    # Add the number of state vector elements and the DOFS per cluster
    # to the records
    nstate_record.append(np.sum(cells))
    dpc_record.append(est1_ms.dofs.sum()/est1_ms.nstate)
    dofs_record.append(est1_ms.dofs.sum())

    # Check whether the change in DOFS per cluster from the previous
    # estimate is greater than the threshold
    dpc_check = ((dpc_record[-1] - dpc_record[-2]) > threshold)
                 # or (iteration_count <= 1))
    sv_check = (np.sum(cells) < est0.nstate)

    # If all of the state vector elements are allocated to the state
    # vector, stop the loop
    if not sv_check:
        print('ALL NATIVE-RESOLUTION STATE VECTOR ELEMENTS ASSIGNED.')
        print(f'ITERATIONS : {iteration_count}')
        state_thresholds.append(np.sum(cells))
        dpc_check = False
    # If it is less than the threshold, increase the cluster size
    elif not dpc_check:
        print('Updating cluster size.')
        # Increase the cluster size and add more native-resolution
        # state vector elements to be allocated to the state vector
        # Add to state_thresholds
        state_thresholds.append(np.sum(cells))
        cluster_size.append(2**size_count)
        cells.append(cluster_size[-1]*cell_base)

        # Update dpc_check
        dpc_check = True

        # Up the counts
        size_count += 1
    else:
        # Add more native-resolution state vector elements to be
        # allocated
        cells[-1] += cluster_size[-1]*cell_base

    # Up the iteration count
    iteration_count += 1

# Second update
est2_ms = est1_ms.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                     clusters_plot, n_cells=[50],
                                     dofs_e=est0.dofs)

ax[0].plot(nstate_record, dpc_record, color=fp.color(0))
for i in range(len(state_thresholds)-1):
    ax[0].axvspan(state_thresholds[i], state_thresholds[i+1],
                  color=fp.color(2*i), alpha=0.2, zorder=-100)
ax[0].set_ylim(0.3, 0.5)

ax1 = ax[0].twinx()
ax1.plot(nstate_record, dofs_record, color=fp.color(0), ls=':')

ax[0].set_xlim(0, est0.nstate)


fp.save_fig(fig, plots, 'rd_scheme_test')

# Save outputs
np.savetxt(join(inputs, 'state_vector_rd.csv'), est2_ms.state_vector,
           delimiter=',')
np.savetxt(join(inputs, 'k_rd.csv'), est2_ms.k, delimiter=',')
np.savetxt(join(inputs, 'sa_vec_rd.csv'), est2_ms.sa_vec, delimiter=',')
np.savetxt(join(inputs, 'xa_rd.csv'), est2_ms.xa, delimiter=',')
np.savetxt(join(inputs, 'xa_abs_rd.csv'), est2_ms.xa_abs, delimiter=',')
np.savetxt(join(inputs, 'xhat_long_rd.csv'), est2_ms.xhat_long, delimiter=',')
np.savetxt(join(inputs, 'dofs_long_rd.csv'), est2_ms.dofs_long, delimiter=',')
np.savetxt(join(inputs, 'shat_err_long_rd.csv'),
           est2_ms.shat_err_long, delimiter=',')

print('Complete.')
