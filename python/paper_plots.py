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
rcParams['font.family'] = 'serif'
rcParams['font.size'] = config.LABEL_FONTSIZE*config.SCALE
rcParams['text.usetex'] = True

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

#----------------------------------------------------------------------------#
#  Data Processing                                                           #
#----------------------------------------------------------------------------#

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

#########################
### UPDATED ESTIMATES ###
#########################

## Reduced rank

# First set of updates
est1 = est0.update_jacobian(true.k,
                            snr=2)

# Second set
est2 = est1.update_jacobian(true.k,
                            rank=est0.get_rank(pct_of_info=0.98))

# Filter
mask = np.diag(est2.a) > 0.01
# mask = ~np.isclose(est2.xhat, np.ones(est2.nstate), atol=1e-2)
est2_f, true_f = est2.filter(true, mask)

print('-----------------------')
print('MODEL RUNS: ', est2.model_runs)
print('CONSTRAINED CELLS: ', len(est2_f.xhat))
print('Filtered DOFS: ', np.trace(est2_f.a))
print('Total DOFS: ', np.trace(est2.a))
print('-----------------------\n')

# Reduced dimension

test = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
                               clusters_plot,
                               n_cells=[200], n_cluster_size=[1])

# print(test.xhat)
# print(true.xhat[np.argsort(test.state_vector)][-200:])

# Posterior emissions
test.plot_state('xhat_long', clusters_plot, cbar=True, title='Ugh',
                vmin=0, vmax=2, cmap='RdBu_r',
                default_value=1)
plt.show()
# print(test.)

# fig, ax = fp.get_figax(aspect=4, rows=2, cols=1)
# ax[0] = fp.add_labels(ax[0],
#                       xlabel='Number of Native Resolution Grid Cells',
#                       ylabel='DOFS per Cluster')

# # est1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
# #                                   clusters_plot,
# #                                   n_cells=[2098],
# #                                   n_cluster_size=[2098])

# # nstate_record_01 = [est1_ms.nstate]
# # dpc_record_01 = [est1_ms.dofs.sum()/est1_ms.nstate]
# nstate_record_01 = []
# dpc_record_01 = []
# indices = np.arange(10, 150, 10)
# for i in indices:
#     est1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
#                                        clusters_plot,
#                                        n_cells=[i],
#                                        n_cluster_size=[1])
#     nstate_record_01.append(est1_ms.nstate)
#     dpc_record_01.append(est1_ms.dofs.sum()/est1_ms.nstate)

# threshold01_idx = np.where(dpc_record_01 == max(dpc_record_01[1:]))[0][0]
# threshold01 = int(indices[threshold01_idx])
# n01 = nstate_record_01
# cols = np.array([1, threshold01, n01[threshold01_idx]-1, n01[-1]])

# ax[0].plot(indices[:threshold01_idx+1],
#            dpc_record_01[:threshold01_idx+1],
#            c=fp.color(0), lw=3, ls='-',
#            label='Clusters of ~1 grid cell')
# ax[0].plot(indices[threshold01_idx:],
#            dpc_record_01[threshold01_idx:],
#            c=fp.color(0), lw=3, ls=':')
# ax[0].axvspan(0, threshold01, color=fp.color(0), alpha=0.2, zorder=-100)

# # Second update
# nstate_record_02 = []
# dpc_record_02 = []
# indices = np.arange(10, 200, 10)
# for i in indices:
#     est1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
#                                        clusters_plot,
#                                        n_cells=[threshold01, i],
#                                        n_cluster_size=[1, 2])
#     nstate_record_02.append(est1_ms.nstate)
#     dpc_record_02.append(est1_ms.dofs.sum()/est1_ms.nstate)

# x02 = threshold01 + np.concatenate(([0], indices))
# # x02 = np.concatenate(([0], indices))
# y02 = np.concatenate(([dpc_record_01[threshold01_idx]],
#                        dpc_record_02))
# # y02 = dpc_record_02
# n02 = np.concatenate(([n01[threshold01_idx]], nstate_record_02))
# # n02 = nstate_record_02
# # threshold02_idx = np.where(x02 > threshold02)[0][0]
# threshold02_idx = np.where(y02 == max(dpc_record_02[1:]))[0][-1]
# # threshold02_idx = -1
# threshold02 = int(x02[threshold02_idx])

# cols02 = np.array([2, threshold02-threshold01,
#                    n02[threshold02_idx]-n01[threshold01_idx],
#                    n02[-1]-n01[threshold01_idx]])
# cols = np.append(cols.reshape((-1,1)), cols02.reshape((-1,1)), axis=1)

# ax[0].plot(x02[:threshold02_idx+1], y02[:threshold02_idx+1],
#            c=fp.color(2), lw=3, ls='-',
#            label='Clusters of ~2 grid cells')
# ax[0].plot(x02[threshold02_idx:], y02[threshold02_idx:],
#         c=fp.color(2), lw=3, ls=':')
# ax[0].axvspan(threshold01, threshold02,
#               color=fp.color(2), alpha=0.2, zorder=-100)

# nstate_record_03 = []
# dpc_record_03 = []
# indices = np.arange(10, 600, 10)
# for i in indices:
#     est1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
#                                        clusters_plot,
#                                        # n_cells=[threshold02, i],
#                                        # n_cluster_size=[2, 4])
#                                        n_cells=[threshold01,
#                                                 threshold02-threshold01,
#                                                 i],
#                                        n_cluster_size=[1, 2, 4])
#     nstate_record_03.append(est1_ms.nstate)
#     dpc_record_03.append(est1_ms.dofs.sum()/est1_ms.nstate)

# x03 = threshold02 + np.concatenate(([0], indices))
# y03 = np.concatenate(([y02[threshold02_idx]], dpc_record_03))
# n03 = np.concatenate(([n02[threshold02_idx]], nstate_record_03))
# # threshold03_idx = np.where(y03 == max(dpc_record_03))[0][-1]
# threshold03_idx = -1
# threshold03 = int(x03[threshold03_idx])
# cols = np.append(cols,
#                  values=[[4],
#                          [threshold03-threshold02],
#                          [n03[threshold03_idx]-n02[threshold02_idx]],
#                          [n03[-1]-n02[threshold02_idx]]],
#                  axis=1)
# # ax[0].plot(x03[:threshold03_idx+1], y03[:threshold03_idx+1],
# ax[0].plot(x03, y03,
#            c=fp.color(4), lw=3, ls='-',
#            label='Clusters of ~4 grid cells')
# # ax[0].plot(x03[threshold03_idx:], y03[threshold03_idx:],
# #         c=fp.color(4), lw=3, ls=':')
# ax[0].axvspan(threshold02, threshold03,
#            color=fp.color(4), alpha=0.2, zorder=-100)

# nstate_record_04 = []
# dpc_record_04 = []
# indices = np.arange(20, 200, 20)
# for i in indices:
#     est1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
#                                       clusters_plot,
#                                        n_cells=[threshold01,
#                                                 threshold02-threshold01,
#                                                 threshold03-threshold02,
#                                                 i],
#                                        n_cluster_size=[1, 2, 4, 8])
#     nstate_record_04.append(est1_ms.nstate)
#     dpc_record_04.append(est1_ms.dofs.sum()/est1_ms.nstate)

# x04 = threshold03 + np.concatenate(([0], indices))
# y04 = np.concatenate(([y03[threshold03_idx]], dpc_record_04))
# n04 = np.concatenate([[n03[threshold03_idx]], nstate_record_04])
# # threshold04_idx = np.where(y04 == max(dpc_record_04))[0][-1]
# threshold04_idx = -1
# threshold04 = int(x04[threshold04_idx])

# cols = np.append(cols,
#                  values=[[8],
#                          [threshold04-threshold03],
#                          [n04[threshold04_idx]-n03[threshold03_idx]],
#                          [n04[-1]-n03[threshold03_idx]]],
#                  axis=1)

# # ax[0].plot(x, dpc_record_04, c=fp.color(8), lw=3, ls='-')
# # ax[0].plot(x04[:threshold04_idx+1], y04[:threshold04_idx+1],
# ax[0].plot(x04, y04,
#            c=fp.color(6), lw=3, ls='-',
#            label='Clusters of ~8 grid cells')
# # ax[0].plot(x04[threshold04_idx:], y04[threshold04_idx:],
# #         c=fp.color(5), lw=3, ls=':')
# ax[0].axvspan(threshold03, threshold04,
#            color=fp.color(6), alpha=0.2, zorder=-100)

# nstate_record_05 = []
# dpc_record_05 = []
# indices = np.arange(20, 400, 50)
# for i in indices:
#     est1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
#                                       clusters_plot,
#                                        n_cells=[threshold01,
#                                                 threshold02-threshold01,
#                                                 threshold03-threshold02,
#                                                 threshold04-threshold03,
#                                                 i],
#                                        n_cluster_size=[1, 2, 4, 8, 16])
#     nstate_record_05.append(est1_ms.nstate)
#     dpc_record_05.append(est1_ms.dofs.sum()/est1_ms.nstate)

# x05 = threshold04 + np.concatenate(([0], indices))
# y05 = np.concatenate(([y04[threshold04_idx]], dpc_record_05))
# n05 = np.concatenate([[n04[threshold04_idx]], nstate_record_05])
# # threshold05_idx = np.where(y05 == max(dpc_record_05))[0][-1]
# threshold05_idx = -1
# threshold05 = int(x05[threshold05_idx])
# cols = np.append(cols,
#                  values=[[16],
#                          [threshold05-threshold04],
#                          [n05[threshold05_idx]-n04[threshold04_idx]],
#                          [n05[-1]-n04[threshold04_idx]]],
#                  axis=1)
# # ax[0].plot(x, dpc_record_05, c=fp.color(10), lw=3, ls='-')
# # ax[0].plot(x05[:threshold05_idx+1], y05[:threshold05_idx+1],
# ax[0].plot(x05, y05,
#            c=fp.color(7), lw=3, ls='-',
#            label='Clusters of ~16 grid cells')
# # ax[0].plot(x05[threshold05_idx:], y05[threshold05_idx:],
# #         c=fp.color(6), lw=3, ls=':')
# ax[0].axvspan(threshold04, threshold05,
#            color=fp.color(7), alpha=0.2, zorder=-100)

# nstate_record_06 = []
# dpc_record_06 = []
# indices = np.arange(50, 800, 100)
# for i in indices:
#     est1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
#                                        clusters_plot,
#                                        n_cells=[threshold01,
#                                                 threshold02-threshold01,
#                                                 threshold03-threshold02,
#                                                 threshold04-threshold03,
#                                                 threshold05-threshold04,
#                                                 i],
#                                        n_cluster_size=[1, 2, 4, 8, 16, 32])
#     nstate_record_06.append(est1_ms.nstate)
#     dpc_record_06.append(est1_ms.dofs.sum()/est1_ms.nstate)

# x06 = threshold05 + np.concatenate(([0], indices))
# y06 = np.concatenate(([y05[threshold05_idx]], dpc_record_06))
# n06 = np.concatenate([[n05[threshold05_idx]], nstate_record_06])
# # threshold06_idx = np.where(y06 == max(dpc_record_06))[0][-1]
# threshold06_idx = -1
# threshold06 = int(x06[threshold06_idx])
# cols = np.append(cols,
#                  values=[[32],
#                          [threshold06-threshold05],
#                          [n06[threshold06_idx]-n05[threshold05_idx]],
#                          [n06[-1]-n05[threshold05_idx]]],
#                  axis=1)

# # ax[0].plot(x, dpc_record_06, c=fp.color(10), lw=3, ls='-')
# # ax[0].plot(x06[:threshold06_idx+1], y06[:threshold06_idx+1],
# ax[0].plot(x06, y06,
#            c=fp.color(8), lw=3, ls='-',
#            label='Clusters of ~32 grid cells')
# # ax[0].plot(x06[threshold06_idx:], y06[threshold06_idx:],
# #         c=fp.color(7), lw=3, ls=':')
# ax[0].axvspan(threshold05, threshold06,
#            color=fp.color(8), alpha=0.2, zorder=-100)

# # nstate_record_07 = []
# # dpc_record_07 = []
# # indices = np.arange(100, 1300, 100)
# # for i in indices:
# #     est1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
# #                                        clusters_plot,
# #                                        n_cells=[threshold01,
# #                                                 threshold02-threshold01,
# #                                                 threshold03-threshold02,
# #                                                 threshold04-threshold03,
# #                                                 threshold05-threshold04,
# #                                                 threshold06-threshold05,
# #                                                 i],
# #                                        n_cluster_size=[1, 2, 4, 8, 16, 32, 64])
# #     nstate_record_07.append(est1_ms.nstate)
# #     dpc_record_07.append(est1_ms.dofs.sum()/est1_ms.nstate)

# # x07 = threshold06 + np.concatenate(([0], indices))
# # y07 = np.concatenate(([y06[threshold06_idx]], dpc_record_07))
# # n07 = np.concatenate(([n06[threshold06_idx]], nstate_record_07))

# # cols = np.append(cols,
# #                  values=[[64],
# #                          [true.nstate-threshold06],
# #                          [n07[-1]-n06[threshold06_idx]],
# #                          [n07[-1]-n06[-1]]],
# #                  axis=1)
# # ax[0].plot(x07, y07, c=fp.color(8), lw=3, ls='-',
# #           label='Clusters of ~64 grid cells')
# # # ax[0].plot(x[:threshold_idx+1], dpc_record_07[:threshold_idx+1],
# # #         c=fp.color(6), lw=3, ls='-')
# # # ax[0].plot(x[threshold_idx:], dpc_record_07[threshold_idx:],
# # #         c=fp.color(6), lw=3, ls=':')
# # ax[0].axvspan(threshold06, true.nstate,
# #            color=fp.color(8), alpha=0.2, zorder=-100)

# ax[0].set_xlim(0, true.nstate)
# # ax[0].set_ylim(0.15, 0.25)
# # ax[0].set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.05)

# # Code up a table from scratch
# ax[1].set_axis_off()

# # cols = np.append(cols, cols.sum(axis=0))
# # cell_text = cols.astype(str)
# # row_labels = np.array(['Cluster size', 'Number of N.R. cells',
# #                        'Number of clusters', 'Model Runs']).reshape((-1, 1))
# # cell_text = np.append(row_labels, cell_text)
# # cell_text[0, -1] = 'Total'


# # colors = ['white'] + [fp.color(i) for i in [0, 2, 4, 5, 6, 7, 8]] + ['white']

# # for i in range(9):
# #     ax[1].axvline(i)
# #     ax[1].axvspan(i, i+1, color=colors[i], alpha=0.2, zorder=-100)


# colors = [fp.color(0), fp.color(2), fp.color(4),
#           fp.color(6), fp.color(7), fp.color(8),
#           # fp.color(8),
#           'white']*3
# colors = np.array(colors).reshape((3, -1))
# cellLoc = 'center'
# # colWidths = [threshold01,
# #              threshold02-threshold01,
# #              threshold03-threshold02,
# #              threshold04-threshold03,
# #              threshold05-threshold04,
# #              threshold06-threshold05,
# #              true.nstate-threshold06]
# # colWidths = [w/true.nstate for w in colWidths]
# colWidths = [1/colors.shape[1]]*colors.shape[1]
# rowLabels = [r'Cluster size', #r'\# of native resolution cells',
#              r'\# of clusters', r'\# model runs']
# rowLoc = 'right'

# # cell_text = list(cols.flatten().astype(str))
# cols = np.append(cols, cols.sum(axis=1).reshape((-1, 1)), axis=1)
# cell_text = cols.astype(str)
# cell_text[0, -1] = 'Total'
# cell_text = cell_text[[0,2,3],:]

# table = tbl.table(ax[1],
#                   cellText=cell_text, cellLoc=cellLoc,
#                   cellColours=colors, colWidths=colWidths,
#                   rowLabels=rowLabels, rowLoc=rowLoc,
#                   fontsize=(config.TICK_FONTSIZE-2)*config.SCALE,
#                   loc='center')
# table.scale(1, 3)

# for cell in table._cells:
#     table._cells[cell].set_alpha(0.1)

# ax[1].add_table(table)
# plt.subplots_adjust(hspace=0.4)

# # print(cols)

# ax[0] = fp.add_title(ax[0],
#                      title='DOFS per Cluster Accumulation in Multiscale Grids')


# fp.save_fig(fig, plots, 'fig03_msg_scheme')

# #######################################
# ### SENSITIVITY TESTS: REDUCED RANK ###
# #######################################
# # DON'T RERUN UNLESS ABSOLUTELY NECESSARY
# # n = 41
# # r2_summ = np.zeros((n, n))
# # nc_summ = np.zeros((n, n))
# # nm_summ = np.zeros((n, n))
# # dofs_summ = np.zeros((n, n))
# # indices = np.concatenate(([1], np.arange(25, 1025, 25)))
# # for col, first_update in enumerate(indices):
# #     for row, second_update in enumerate(indices):
# #         test1 = est0.update_jacobian(true.k, rank=first_update)
# #         test2 = test1.update_jacobian(true.k, rank=second_update)
# #         mask = np.diag(test2.a) > 0.01
# #         test2_f, true_f = test2.filter(true, mask)
# #         _, _, r = test2_f.calc_stats(true_f.xhat, test2_f.xhat)

# #         r2_summ[row, col] = r**2
# #         nc_summ[row, col] = len(test2_f.xhat)
# #         nm_summ[row, col] = test2_f.model_runs
# #         dofs_summ[row, col] = np.trace(test2.a)

# # np.save(join(inputs, 'r2_summary'), r2_summ)
# # np.save(join(inputs, 'nc_summary'), nc_summ)
# # np.save(join(inputs, 'nm_summary'), nm_summ)
# # np.save(join(inputs, 'dofs_summary'), dofs_summ)

# # open summary files
# r2_summ = np.load(join(inputs, 'r2_summary_R3.npy'))
# nc_summ = np.load(join(inputs, 'nc_summary_R3.npy'))
# nm_summ = np.load(join(inputs, 'nm_summary_R3.npy'))
# dofs_summ = np.load(join(inputs, 'dofs_summary_R3.npy'))

# #----------------------------------------------------------------------------#
# #  Figures                                                                   #
# #----------------------------------------------------------------------------#

# ################################################
# ### FIGURE 01: RANK AND DIMENSION FLOW CHART ###
# ################################################

# def flow_chart_settings(ax):
#     ax.add_feature(cartopy.feature.OCEAN, facecolor='white', zorder=2)
#     ax.coastlines(color='grey', zorder=5)
#     ax.outline_patch.set_visible(False)
#     return ax

# # # Original dimension
# # fig01a, ax = est0.plot_multiscale_grid(clusters_plot, colors='0.5', zorder=3,
# #                                        fig_kwargs=small_fig_kwargs,
# #                                        map_kwargs=small_map_kwargs)
# # ax = flow_chart_settings(ax)
# # fp.save_fig(fig01a, loc=plots, name='fig01a_dimn_rankn')

# # # Reduced rank
# # true.evec_sum = true.evecs[:, :3].sum(axis=1)
# # fig01b, ax, c = true.plot_state('evec_sum', clusters_plot,
# #                                 title='', cbar=False,  cmap='RdBu_r',
# #                                 vmin=-0.1, vmax=0.1, default_value=0,
# #                                 fig_kwargs=small_fig_kwargs,
# #                                 map_kwargs=small_map_kwargs)
# # ax = flow_chart_settings(ax)
# # fp.save_fig(fig01b, loc=plots, name='fig01b_dimn_rankk')

# # # Reduced rank and dimension (not aggregate)
# # for i in range(3):
# #     fig01c, ax, c = true.plot_state(('evecs', i), clusters_plot,
# #                                     title='', cbar=False, cmap='RdBu_r',
# #                                     vmin=-0.1, vmax=0.1,default_value=0,
# #                                     fig_kwargs=small_fig_kwargs,
# #                                     map_kwargs=small_map_kwargs)
# #     ax = flow_chart_settings(ax)
# #     fp.save_fig(fig01c, loc=plots, name='fig01c_evec' + str(i))

# # Reduced dimension (aggregate)
# fig01d, ax = est1_ms.plot_multiscale_grid(clusters_plot,
#                                           colors='0.5', zorder=3,
#                                           fig_kwargs=small_fig_kwargs,
#                                           map_kwargs=small_map_kwargs)
# ax = flow_chart_settings(ax)
# fp.save_fig(fig01d, loc=plots, name='fig01d_dimk_rankk_ms')

# #########################################################################
# ### FIGURE 02: AVERAGING KERNEL SENSITIVITY TO PRIOR AND OBSERVATIONS ###
# #########################################################################

# # True averaging kernel
# title = 'Native Resolution\nAveraging Kernel Sensitivities'
# avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
# avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
#                 'cbar_kwargs' : avker_cbar_kwargs,
#                 'fig_kwargs' : small_fig_kwargs,
#                 'map_kwargs' : small_map_kwargs}
# fig02a, ax, c = true.plot_state('dofs', clusters_plot, title=title,
#                                 **avker_kwargs)
# ax.text(0.025, 0.05, 'DOFS = %d' % np.trace(true.a),
#         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#         transform=ax.transAxes)
# fp.save_fig(fig02a, plots, 'fig02a_true_averaging_kernel')

# # Initial estimate averaging kernel
# title = 'Initial Estimate\nAveraging Kernel Sensitivities'
# # avker_kwargs['cbar_kwargs'] = avker_cbar_kwargs
# fig02b, ax, c = est0.plot_state('dofs', clusters_plot, title=title,
#                                 **avker_kwargs)
# ax.text(0.025, 0.05, 'DOFS = %d' % np.trace(est0.a),
#         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#         transform=ax.transAxes)
# fp.save_fig(fig02b, plots, 'fig02b_est0_averaging_kernel')

# # Prior error
# true.sd_vec = true.sa_vec**0.5
# true.sd_vec_abs = true.sd_vec*true.xa_abs
# cbar_kwargs = {'title' : 'Tg/month'}
# fig02c, ax, c = true.plot_state('sd_vec_abs', clusters_plot,
#                                 title='Prior Error Standard Deviation',
#                                 cmap=fp.cmap_trans('viridis'),
#                                 vmin=0, vmax=15,
#                                 fig_kwargs=small_fig_kwargs,
#                                 cbar_kwargs=cbar_kwargs,
#                                 map_kwargs=small_map_kwargs)
# fp.save_fig(fig02c, plots, 'fig02c_prior_error')

# # Observational density
# lat_res = np.diff(clusters_plot.lat)[0]
# lat_edges = np.append(clusters_plot.lat - lat_res/2,
#                       clusters_plot.lat[-1] + lat_res/2)
# # lat_edges = lat_edges[::2]
# obs['lat_edges'] = pd.cut(obs['lat'], lat_edges, precision=4)

# lon_res = np.diff(clusters_plot.lon)[0]
# lon_edges = np.append(clusters_plot.lon - lon_res/2,
#                       clusters_plot.lon[-1] + lon_res/2)
# # lon_edges = lon_edges[::2]
# obs['lon_edges'] = pd.cut(obs['lon'], lon_edges, precision=4)

# obs_density = obs.groupby(['lat_edges', 'lon_edges']).count()
# obs_density = obs_density['Nobs'].reset_index()
# obs_density['lat'] = obs_density['lat_edges'].apply(lambda x: x.mid)
# obs_density['lon'] = obs_density['lon_edges'].apply(lambda x: x.mid)
# obs_density = obs_density.set_index(['lat', 'lon'])['Nobs']
# obs_density = obs_density.to_xarray()

# title = 'GOSAT Observation Density\n(July 2009)'
# viridis_trans_long = fp.cmap_trans('viridis', nalpha=90, ncolors=300)
# cbar_kwargs = {'ticks' : np.arange(0, 25, 5),
#                'title' : 'Count'}
# fig02d, ax, c = true.plot_state_format(obs_density, title=title,
#                                        vmin=0, vmax=10, default_value=0,
#                                        cmap=viridis_trans_long,
#                                        fig_kwargs=small_fig_kwargs,
#                                        cbar_kwargs=cbar_kwargs,
#                                        map_kwargs=small_map_kwargs)
# fp.save_fig(fig02d, plots, 'fig02d_gosat_obs_density')

# #########################################
# ### FIGURE 03: MULTISCALE GRID SCHEME ###
# #########################################
# # This is coded in the multiscale grid update.

# ##################################
# ### FIGURE 04: MULTISCALE GRID ###
# ##################################
# fig04, ax = est1_ms.plot_multiscale_grid(clusters_plot,
#                                           colors='0.5', zorder=3,
#                                           title='Multiscale Grid',
#                                           fig_kwargs=small_fig_kwargs,
#                                           map_kwargs=small_map_kwargs)
# fp.save_fig(fig04, loc=plots, name='fig04_est2_ms_grid')

# ###############################################
# ### FIGURE 05 : CONSOLIDATED POSTERIOR PLOT ###
# ###############################################
# fig05a, ax05a = fp.get_figax(rows=1, cols=3, maps=True,
#                              lats=clusters_plot.lat, lons=clusters_plot.lon)
# fig05b, ax05b = fp.get_figax(rows=1, cols=3, maps=True,
#                              lats=clusters_plot.lat, lons=clusters_plot.lon)

# def add_dofs_subtitle(inversion_object, ax,
#                       state_vector_element_string='cell'):
#     subtitle = ('%d DOFS (%.2f/%s)\n%d model simulations'
#                 % (np.trace(inversion_object.a),
#                   (np.trace(inversion_object.a)/inversion_object.nstate),
#                   state_vector_element_string,
#                   inversion_object.model_runs))
#     ax.text(0.5, 1.125, subtitle,
#             fontsize=config.SUBTITLE_FONTSIZE*config.SCALE,
#             ha='center', transform=ax.transAxes)
#     return ax

# title_kwargs = {'y' : 1.45}
# state_cbar_kwargs = {'ticks' : np.arange(-1, 4, 1)}
# dofs_cbar_kwargs = {'ticks' : np.arange(0, 1.1, 0.25)}
# state_kwargs = {'default_value' : 1, 'cmap' : 'RdBu_r',
#                 'vmin' : -1, 'vmax' : 3,
#                 'cbar' : False, 'cbar_kwargs' : state_cbar_kwargs,
#                 'title_kwargs' : title_kwargs, 'map_kwargs' : small_map_kwargs}
# dofs_kwargs =  {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
#                 'cbar' : False, 'cbar_kwargs' : dofs_cbar_kwargs,
#                 'title_kwargs' : title_kwargs, 'map_kwargs' : small_map_kwargs}
# titles = ['Native Resolution', 'Multiscale Grid', 'Reduced Rank']
# sve_string = ['cell', 'cluster', 'cell']
# quantities = ['', '_long', '']

# for i, inv in enumerate([true, est1_ms, est2]):
#     state_kwargs['title'] = titles[i]
#     state_kwargs['fig_kwargs'] = {'figax' : [fig05a, ax05a[i]]}
#     dofs_kwargs['title'] = titles[i]
#     dofs_kwargs['fig_kwargs'] = {'figax' : [fig05b, ax05b[i]]}

#     # Posterior emissions
#     fig05a, ax05a[i], ca = inv.plot_state('xhat' + quantities[i],
#                                          clusters_plot, **state_kwargs)
#     ax05a[i] = add_dofs_subtitle(inv, ax05a[i], sve_string[i])

#     # Averaging kernel sensitivities
#     fig05b, ax05b[i], cb = inv.plot_state('dofs' + quantities[i],
#                                          clusters_plot, **dofs_kwargs)
#     ax05b[i] = add_dofs_subtitle(inv, ax05b[i], sve_string[i])

# # Polishing posterior emissions
# # Colorbar
# cax = fp.add_cax(fig05a, ax05a)
# cbar = fig05a.colorbar(ca, cax=cax, **state_cbar_kwargs)
# cbar = fp.format_cbar(cbar, cbar_title='Scaling Factors')

# # Label
# ax05a[0].text(-0.4, 0.5, 'Posterior\nScaling\nFactors',
#               fontsize=config.TITLE_FONTSIZE*config.SCALE,
#               rotation=90, ha='center', va='center',
#               transform=ax05a[0].transAxes)
# # Save
# fp.save_fig(fig05a, plots, 'fig05a_posterior_mean_summary')

# # Polishing averaging kernel sensitivities
# # Colorbar
# cax = fp.add_cax(fig05b, ax05b)
# cbar = fig05b.colorbar(cb, cax=cax, **dofs_cbar_kwargs)
# cbar = fp.format_cbar(cbar, cbar_title=r'$\partial\hat{x}/\partial x$')

# # Label
# ax05b[0].text(-0.4, 0.5, 'Averaging\nKernel\nSensitivities',
#               fontsize=config.TITLE_FONTSIZE*config.SCALE,
#               rotation=90, ha='center', va='center',
#               transform=ax05b[0].transAxes)

# # Save
# fp.save_fig(fig05b, plots, 'fig05b_averaging_kernel_summary')

# ############################################################
# ### FIGURE 06: EST2_F POSTERIOR COMPARSION SCATTER PLOTS ###
# ############################################################
# fig06, _, _ = est2_f.full_analysis(true_f, clusters_plot)
# fp.save_fig(fig06, plots, 'fig06_posterior_scattter_comparison')

# #################################################
# ### FIGURE 07: REDUCED RANK SENSITIVITY TESTS ###
# #################################################

# min_mr = 0
# max_mr = 1000
# increment = 25
# n = int((max_mr - min_mr)/increment) + 1

# def mr2n(model_runs,
#          min_mr=min_mr, max_mr=max_mr, max_n=n):
#     '''This function converts a number of model runs
#     to an index along an axis from 0 to n'''
#     x0 = min_mr
#     y0 = 0
#     slope = (max_n - 1)/(max_mr - min_mr)
#     func = lambda mr : slope*(mr - x0) + y0
#     return func(model_runs)

# # R2 plot
# fig07, ax = fp.get_figax()
# cax = fp.add_cax(fig07, ax)
# # ax = fp.add_title(ax, r'Posterior Emissions r$^2$'))
# ax = fp.add_title(ax, 'DOFS')
# # ax = fp.add_title(ax, 'Number of Constrained Grid Cells')

# # cf = ax.contourf(r2_summ, levels=np.linspace(0, 1, 25), vmin=0, vmax=1)
# cf = ax.contourf(dofs_summ,
#                  levels=np.linspace(0, true.dofs.sum(), 50),
#                  vmin=0, vmax=200, cmap='plasma')
# # cf = ax.contourf(nc_summ, levels=np.arange(0, 2000, 100),
# #                  vmin=0, vmax=2000)

# locs = [(n/4, n/4),
#         (n/2, n/2)]
# # nm_summ[0, :] -= 1
# # nm_summ[1:, 0] -= 1
# cl = ax.contour(nm_summ, levels=[500, 1000], colors=fp.color(3),
#                 linestyles='dotted')
# ax.clabel(cl, cl.levels, inline=True, manual=locs, fmt='%d',
#           fontsize=config.TICK_FONTSIZE*config.SCALE)
# cl = ax.contour(nm_summ, levels=[est2.model_runs], colors=fp.color(3),
#                  linestyles='dashed')
# # ax.clabel(cl, cl.levels, inline=True, fmt='%d',
# #           fontsize=config.TICK_FONTSIZE*config.SCALE)

# ax.scatter(mr2n(est1.model_runs),
#            mr2n(est2.model_runs-est1.model_runs),
#            zorder=10,
#            c=fp.color(3),
#            s=200, marker='*')

# # cbar = fig07.colorbar(cf, cax=cax, ticks=np.linspace(0, 1, 6))
# cbar = fig07.colorbar(cf, cax=cax,
#                     ticks=np.arange(0, true.dofs.sum(), 50))
# # cbar = fig07.colorbar(cf, cax=cax, ticks=np.arange(0, 2000, 500))

# # cbar = fp.format_cbar(cbar, r'r$^2$')
# cbar = fp.format_cbar(cbar, 'DOFS')
# # cbar = fp.format_cbar(cbar, 'Number of Constrained Cells')

# # Axis and tick labels
# ax = fp.add_labels(ax, 'First Update Model Runs', 'Second Update Model Runs')

# # ....This is still hard coded
# ax.set_xticks(np.arange(4, n, (n-4)/9))
# ax.set_xticklabels(np.arange(100, 1100, 100), fontsize=15)
# ax.set_xlim(0, (n-1)*750/975)

# ax.set_yticks(np.arange(4, n, (n-4)/9))
# ax.set_yticklabels(np.arange(100, 1100, 100), fontsize=15)
# ax.set_ylim(0, (n-1)*750/975)

# fp.save_fig(fig07, plots, 'fig07_dofs_comparison')
