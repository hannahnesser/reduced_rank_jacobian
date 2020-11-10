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
sa_vec = xr.open_dataarray(join(inputs, 'sa_vec.nc'))

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

RF = 20

############
### TRUE ###
############

# Create a true Reduced Rank Jacobian object
true = inv.ReducedRankJacobian(k_true.values,
                               xa.values,
                               sa_vec.values,
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
                               sa_vec.values,
                               y.values,
                               y_base.values,
                               so_vec.values)
est0.xa_abs = xa_abs *1e3
est0.rf = RF
est0.edecomp()
est0.solve_inversion()

########################################
### BUILD REDUCED DIMENSION JACOBIAN ###
########################################

# Reduced dimension
fig, ax = fp.get_figax(aspect=4, rows=2, cols=1)
ax[0] = fp.add_labels(ax[0],
                      xlabel='Number of Native-Resolution Grid Cells',
                      ylabel='DOFS per Cluster')

est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                  clusters_plot,
                                  n_cells=[2098],
                                  n_cluster_size=[2098])

nstate_record_01 = [est1_ms.nstate]
dpc_record_01 = [est1_ms.dofs.sum()/est1_ms.nstate]
indices = np.arange(10, 160, 10)
for i in indices:
    est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                       clusters_plot,
                                       n_cells=[i],
                                       n_cluster_size=[1])
    nstate_record_01.append(est1_ms.nstate)
    dpc_record_01.append(est1_ms.dofs.sum()/est1_ms.nstate)

threshold01_idx = np.where(dpc_record_01 == max(dpc_record_01[1:]))[0][0]
threshold01 = int(indices[threshold01_idx])
n01 = nstate_record_01
cols = np.array([1, threshold01, n01[threshold01_idx]-1, n01[-1]])

ax[0].plot(np.concatenate(([0], indices))[:threshold01_idx+1],
           dpc_record_01[:threshold01_idx+1],
           c=fp.color(0), lw=3, ls='-',
           label='Clusters of ~1 grid cell')
ax[0].plot(np.concatenate(([0], indices))[threshold01_idx:],
           dpc_record_01[threshold01_idx:],
           c=fp.color(0), lw=3, ls=':')
ax[0].axvspan(0, threshold01, color=fp.color(0), alpha=0.2, zorder=-100)

# Second update
nstate_record_02 = []
dpc_record_02 = []
indices = np.arange(10, 200, 10)
for i in indices:
    est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                       clusters_plot,
                                       n_cells=[threshold01, i],
                                       n_cluster_size=[1, 2])
    nstate_record_02.append(est1_ms.nstate)
    dpc_record_02.append(est1_ms.dofs.sum()/est1_ms.nstate)

x02 = threshold01 + np.concatenate(([0], indices))
# x02 = np.concatenate(([0], indices))
y02 = np.concatenate(([dpc_record_01[threshold01_idx]],
                       dpc_record_02))
# y02 = dpc_record_02
n02 = np.concatenate(([n01[threshold01_idx]], nstate_record_02))
# n02 = nstate_record_02
# threshold02_idx = np.where(x02 > threshold02)[0][0]
threshold02_idx = np.where(y02 == max(dpc_record_02[1:]))[0][-1]
# threshold02_idx = -1
threshold02 = int(x02[threshold02_idx])

cols02 = np.array([2, threshold02-threshold01,
                   n02[threshold02_idx]-n01[threshold01_idx],
                   n02[-1]-n01[threshold01_idx]])
# cols = cols02.reshape((-1, 1))
cols = np.append(cols.reshape((-1,1)), cols02.reshape((-1,1)), axis=1)

ax[0].plot(x02[:threshold02_idx+1], y02[:threshold02_idx+1],
           c=fp.color(2), lw=3, ls='-',
           label='Clusters of ~2 grid cells')
ax[0].plot(x02[threshold02_idx:], y02[threshold02_idx:],
        c=fp.color(2), lw=3, ls=':')
ax[0].axvspan(threshold01, threshold02,
              color=fp.color(2), alpha=0.2, zorder=-100)

nstate_record_03 = []
dpc_record_03 = []
indices = np.arange(10, 600, 10)
for i in indices:
    est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                       clusters_plot,
                                       # n_cells=[threshold02, i],
                                       # n_cluster_size=[2, 4])
                                       n_cells=[threshold01,
                                                threshold02-threshold01,
                                                i],
                                       n_cluster_size=[1, 2, 4])
    nstate_record_03.append(est1_ms.nstate)
    dpc_record_03.append(est1_ms.dofs.sum()/est1_ms.nstate)

x03 = threshold02 + np.concatenate(([0], indices))
y03 = np.concatenate(([y02[threshold02_idx]], dpc_record_03))
n03 = np.concatenate(([n02[threshold02_idx]], nstate_record_03))
# threshold03_idx = np.where(y03 == max(dpc_record_03))[0][-1]
threshold03_idx = -1
threshold03 = int(x03[threshold03_idx])
cols = np.append(cols,
                 values=[[4],
                         [threshold03-threshold02],
                         [n03[threshold03_idx]-n02[threshold02_idx]],
                         [n03[-1]-n02[threshold02_idx]]],
                 axis=1)
# ax[0].plot(x03[:threshold03_idx+1], y03[:threshold03_idx+1],
ax[0].plot(x03, y03,
           c=fp.color(4), lw=3, ls='-',
           label='Clusters of ~4 grid cells')
# ax[0].plot(x03[threshold03_idx:], y03[threshold03_idx:],
#         c=fp.color(4), lw=3, ls=':')
ax[0].axvspan(threshold02, threshold03,
           color=fp.color(4), alpha=0.2, zorder=-100)

nstate_record_04 = []
dpc_record_04 = []
indices = np.arange(20, 200, 20)
for i in indices:
    est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                      clusters_plot,
                                       n_cells=[threshold01,
                                                threshold02-threshold01,
                                                threshold03-threshold02,
                                                i],
                                       n_cluster_size=[1, 2, 4, 8])
    nstate_record_04.append(est1_ms.nstate)
    dpc_record_04.append(est1_ms.dofs.sum()/est1_ms.nstate)

x04 = threshold03 + np.concatenate(([0], indices))
y04 = np.concatenate(([y03[threshold03_idx]], dpc_record_04))
n04 = np.concatenate([[n03[threshold03_idx]], nstate_record_04])
# threshold04_idx = np.where(y04 == max(dpc_record_04))[0][-1]
threshold04_idx = -1
threshold04 = int(x04[threshold04_idx])

cols = np.append(cols,
                 values=[[8],
                         [threshold04-threshold03],
                         [n04[threshold04_idx]-n03[threshold03_idx]],
                         [n04[-1]-n03[threshold03_idx]]],
                 axis=1)

# ax[0].plot(x, dpc_record_04, c=fp.color(8), lw=3, ls='-')
# ax[0].plot(x04[:threshold04_idx+1], y04[:threshold04_idx+1],
ax[0].plot(x04, y04,
           c=fp.color(6), lw=3, ls='-',
           label='Clusters of ~8 grid cells')
# ax[0].plot(x04[threshold04_idx:], y04[threshold04_idx:],
#         c=fp.color(5), lw=3, ls=':')
ax[0].axvspan(threshold03, threshold04,
           color=fp.color(6), alpha=0.2, zorder=-100)

nstate_record_05 = []
dpc_record_05 = []
indices = np.arange(20, 400, 50)
for i in indices:
    est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                      clusters_plot,
                                       n_cells=[threshold01,
                                                threshold02-threshold01,
                                                threshold03-threshold02,
                                                threshold04-threshold03,
                                                i],
                                       n_cluster_size=[1, 2, 4, 8, 16])
    nstate_record_05.append(est1_ms.nstate)
    dpc_record_05.append(est1_ms.dofs.sum()/est1_ms.nstate)

x05 = threshold04 + np.concatenate(([0], indices))
y05 = np.concatenate(([y04[threshold04_idx]], dpc_record_05))
n05 = np.concatenate([[n04[threshold04_idx]], nstate_record_05])
# threshold05_idx = np.where(y05 == max(dpc_record_05))[0][-1]
threshold05_idx = -1
threshold05 = int(x05[threshold05_idx])
cols = np.append(cols,
                 values=[[16],
                         [threshold05-threshold04],
                         [n05[threshold05_idx]-n04[threshold04_idx]],
                         [n05[-1]-n04[threshold04_idx]]],
                 axis=1)
# ax[0].plot(x, dpc_record_05, c=fp.color(10), lw=3, ls='-')
# ax[0].plot(x05[:threshold05_idx+1], y05[:threshold05_idx+1],
ax[0].plot(x05, y05,
           c=fp.color(7), lw=3, ls='-',
           label='Clusters of ~16 grid cells')
# ax[0].plot(x05[threshold05_idx:], y05[threshold05_idx:],
#         c=fp.color(6), lw=3, ls=':')
ax[0].axvspan(threshold04, threshold05,
           color=fp.color(7), alpha=0.2, zorder=-100)

nstate_record_06 = []
dpc_record_06 = []
indices = np.arange(50, 900, 100)
for i in indices:
    est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                       clusters_plot,
                                       n_cells=[threshold01,
                                                threshold02-threshold01,
                                                threshold03-threshold02,
                                                threshold04-threshold03,
                                                threshold05-threshold04,
                                                i],
                                       n_cluster_size=[1, 2, 4, 8, 16, 32])
    nstate_record_06.append(est1_ms.nstate)
    dpc_record_06.append(est1_ms.dofs.sum()/est1_ms.nstate)

x06 = threshold05 + np.concatenate(([0], indices))
y06 = np.concatenate(([y05[threshold05_idx]], dpc_record_06))
n06 = np.concatenate([[n05[threshold05_idx]], nstate_record_06])
# threshold06_idx = np.where(y06 == max(dpc_record_06))[0][-1]
threshold06_idx = -1
threshold06 = int(x06[threshold06_idx])
cols = np.append(cols,
                 values=[[32],
                         [threshold06-threshold05],
                         [n06[threshold06_idx]-n05[threshold05_idx]],
                         [n06[-1]-n05[threshold05_idx]]],
                 axis=1)

# ax[0].plot(x, dpc_record_06, c=fp.color(10), lw=3, ls='-')
# ax[0].plot(x06[:threshold06_idx+1], y06[:threshold06_idx+1],
ax[0].plot(x06, y06,
           c=fp.color(8), lw=3, ls='-',
           label='Clusters of ~32 grid cells')
# ax[0].plot(x06[threshold06_idx:], y06[threshold06_idx:],
#         c=fp.color(7), lw=3, ls=':')
ax[0].axvspan(threshold05, threshold06,
           color=fp.color(8), alpha=0.2, zorder=-100)

# nstate_record_07 = []
# dpc_record_07 = []
# indices = np.arange(100, 1300, 100)
# for i in indices:
#     est1_ms = est0.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
#                                        clusters_plot,
#                                        n_cells=[threshold01,
#                                                 threshold02-threshold01,
#                                                 threshold03-threshold02,
#                                                 threshold04-threshold03,
#                                                 threshold05-threshold04,
#                                                 threshold06-threshold05,
#                                                 i],
#                                        n_cluster_size=[1, 2, 4, 8, 16, 32, 64])
#     nstate_record_07.append(est1_ms.nstate)
#     dpc_record_07.append(est1_ms.dofs.sum()/est1_ms.nstate)

# x07 = threshold06 + np.concatenate(([0], indices))
# y07 = np.concatenate(([y06[threshold06_idx]], dpc_record_07))
# n07 = np.concatenate(([n06[threshold06_idx]], nstate_record_07))

# cols = np.append(cols,
#                  values=[[64],
#                          [true.nstate-threshold06],
#                          [n07[-1]-n06[threshold06_idx]],
#                          [n07[-1]-n06[-1]]],
#                  axis=1)
# ax[0].plot(x07, y07, c=fp.color(8), lw=3, ls='-',
#           label='Clusters of ~64 grid cells')
# # ax[0].plot(x[:threshold_idx+1], dpc_record_07[:threshold_idx+1],
# #         c=fp.color(6), lw=3, ls='-')
# # ax[0].plot(x[threshold_idx:], dpc_record_07[threshold_idx:],
# #         c=fp.color(6), lw=3, ls=':')
# ax[0].axvspan(threshold06, true.nstate,
#            color=fp.color(8), alpha=0.2, zorder=-100)

ax[0].set_xlim(0, true.nstate)
ax[0].set_ylim(0.15, 0.25)
# ax[0].set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.05)

# Code up a table from scratch
ax[1].set_axis_off()

# cols = np.append(cols, cols.sum(axis=0))
# cell_text = cols.astype(str)
# row_labels = np.array(['Cluster size', 'Number of N.R. cells',
#                        'Number of clusters', 'Model Runs']).reshape((-1, 1))
# cell_text = np.append(row_labels, cell_text)
# cell_text[0, -1] = 'Total'


# colors = ['white'] + [fp.color(i) for i in [0, 2, 4, 5, 6, 7, 8]] + ['white']

# for i in range(9):
#     ax[1].axvline(i)
#     ax[1].axvspan(i, i+1, color=colors[i], alpha=0.2, zorder=-100)


colors = [fp.color(0), fp.color(2), fp.color(4),
          fp.color(6), fp.color(7), fp.color(8),
          # fp.color(8),
          'white']*3
colors = np.array(colors).reshape((3, -1))
cellLoc = 'center'
# colWidths = [threshold01,
#              threshold02-threshold01,
#              threshold03-threshold02,
#              threshold04-threshold03,
#              threshold05-threshold04,
#              threshold06-threshold05,
#              true.nstate-threshold06]
# colWidths = [w/true.nstate for w in colWidths]
colWidths = [1/colors.shape[1]]*colors.shape[1]
rowLabels = [r'Cluster size', #r'\# of native resolution cells',
             r'\# of clusters', r'\# model runs']
rowLoc = 'right'

# cell_text = list(cols.flatten().astype(str))
cols = np.append(cols, cols.sum(axis=1).reshape((-1, 1)), axis=1)
cell_text = cols.astype(str)
cell_text[0, -1] = 'Total'
cell_text = cell_text[[0,2,3],:]
est1_ms.model_runs = int(cell_text[-1, -1])

table = tbl.table(ax[1],
                  cellText=cell_text, cellLoc=cellLoc,
                  cellColours=colors, colWidths=colWidths,
                  rowLabels=rowLabels, rowLoc=rowLoc,
                  fontsize=(config.TICK_FONTSIZE-2)*config.SCALE,
                  loc='center')
table.scale(1, 3)

for cell in table._cells:
    table._cells[cell].set_alpha(0.1)

ax[1].add_table(table)
plt.subplots_adjust(hspace=0.4)

# print(cols)

ax[0] = fp.add_title(ax[0],
                     title='DOFS per Cluster Accumulation in Multiscale Grids')

fp.save_fig(fig, plots, 'rd_scheme')

# Second update
est2_ms = est1_ms.update_jacobian_rd(true.k, true.xa_abs, true.sa_vec,
                                     clusters_plot, n_cells=[60],
                                     dofs_e=est0.dofs)
print(est2_ms.dofs.sum())

# Save outputs
np.savetxt(join(inputs, 'state_vector_rd.csv'), est2_ms.state_vector,
           delimiter=',')
np.savetxt(join(inputs, 'k_rd.csv'), est2_ms.k, delimiter=',')
np.savetxt(join(inputs, 'sa_vec_rd.csv'), est2_ms.sa_vec, delimiter=',')
np.savetxt(join(inputs, 'xa_rd.csv'), est2_ms.xa, delimiter=',')
np.savetxt(join(inputs, 'xa_abs_rd.csv'), est2_ms.xa_abs, delimiter=',')
np.savetxt(join(inputs, 'xhat_long_rd.csv'), est2_ms.xhat_long, delimiter=',')
np.savetxt(join(inputs, 'dofs_long_rd.csv'), est2_ms.dofs_long, delimiter=',')

print('Complete.')
