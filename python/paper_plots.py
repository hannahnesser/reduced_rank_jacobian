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
# First set of updates
est1 = est0.update_jacobian(true.k,
                            snr=1.25)

# Second set
est2 = est1.update_jacobian(true.k,
                            rank=est0.get_rank(pct_of_info=0.975))

# Filter
mask = np.diag(est2.a) > 0.01
# mask = ~np.isclose(est2.xhat, np.ones(est2.nstate), atol=1e-2)
est2_f, true_f = est2.filter(true, mask)

print('-----------------------')
print('MODEL RUNS: ', est2.model_runs)
print('CONSTRAINED CELLS: ', len(est2_f.xhat))
print('DOFS: ', np.trace(est2_f.a))
print('-----------------------\n')

# And multiscale grid results
est1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
                                  clusters_plot,
                                  n_cells=[98, 300, 500, 1200],
                                  n_cluster_size=[1, 3, 5, 8])
est2_ms = est1_ms.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
                                     clusters_plot, n_cells=[100],
                                     n_cluster_size=[1])

print(est2_ms.rf)

#######################################
### SENSITIVITY TESTS: REDUCED RANK ###
#######################################
# DON'T RERUN UNLESS ABSOLUTELY NECESSARY
# n = 41
# r2_summ = np.zeros((n, n))
# nc_summ = np.zeros((n, n))
# nm_summ = np.zeros((n, n))
# dofs_summ = np.zeros((n, n))
# indices = np.concatenate(([1], np.arange(25, 1025, 25)))
# for col, first_update in enumerate(indices):
#     for row, second_update in enumerate(indices):
#         test1 = est0.update_jacobian(true.k, rank=first_update)
#         test2 = test1.update_jacobian(true.k, rank=second_update)
#         mask = np.diag(test2.a) > 0.01
#         test2_f, true_f = test2.filter(true, mask)
#         _, _, r = test2_f.calc_stats(true_f.xhat, test2_f.xhat)

#         r2_summ[row, col] = r**2
#         nc_summ[row, col] = len(test2_f.xhat)
#         nm_summ[row, col] = test2_f.model_runs
#         dofs_summ[row, col] = np.trace(test2.a)

# np.save(join(inputs, 'r2_summary'), r2_summ)
# np.save(join(inputs, 'nc_summary'), nc_summ)
# np.save(join(inputs, 'nm_summary'), nm_summ)
# np.save(join(inputs, 'dofs_summary'), dofs_summ)

# open summary files
r2_summ = np.load(join(inputs, 'r2_summary_R3.npy'))
nc_summ = np.load(join(inputs, 'nc_summary_R3.npy'))
nm_summ = np.load(join(inputs, 'nm_summary_R3.npy'))
dofs_summ = np.load(join(inputs, 'dofs_summary_R3.npy'))

##########################################
### SENSITIVITY TESTS: MULTISCALE GRID ###
##########################################

# fig, ax = plt.subplots(figsize=(20, 4))
# ax2 = ax.twinx()

# # test1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
# #                                    clusters_plot, n_cells=2098,
# #                                    n_cluster_size=[50])
# # test1_ms_k = test1_ms.disaggregate_k_ms()

# # test2_ms = inv.ReducedRankJacobian(test1_ms_k,
# #                                    xa.values,
# #                                    sa_vec.values,
# #                                    y.values,
# #                                    y_base.values,
# #                                    so_vec.values)
# # test2_ms.xa_abs = xa_abs *1e3
# # test2_ms.rf = RF
# # test2_ms.edecomp()
# # test2_ms.solve_inversion()

# # test2_ms.full_analysis(true, clusters_plot)

# # plt.show()

# n_record = []
# nstate_record = []
# dofs_record = []
# dofs_per_cell_record = []
# indices = np.concatenate((np.arange(10, 200, 10),
#                           np.arange(200, 500, 50),
#                           np.arange(500, 1000, 100),
#                           np.arange(1000, 2098, 200)))

# for i in indices:
#     test1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
#                                        clusters_plot,
#                                        n_cells=[i],
#                                        n_cluster_size=[1])
#     nstate_record.append(test1_ms.nstate)
#     dofs_record.append(test1_ms.dofs.sum())
#     dofs_per_cell_record.append(test1_ms.dofs.sum()/test1_ms.nstate)

# dofs_per_cell_diff = np.diff(dofs_per_cell_record)/np.diff(indices)
# cum_dy = (dofs_per_cell_record - dofs_per_cell_record[0])[1:]
# cum_dx = (indices - indices[0])[1:]
# cum_slope = cum_dy/cum_dx
# cum_slope_diff = np.diff(cum_slope)/np.diff(indices[1:])
# # threshold = np.where(cum_slope > 0)[0][-1] + 1
# # threshold = np.where(cum_slope_ diff > 0)[0][0]

# # ax.plot(indices[:threshold+1], dofs_per_cell_record[:threshold+1],
# #         c=fp.color(2), lw=3, ls='-')
# # ax.plot(indices[threshold+1:], dofs_per_cell_record[threshold+1:],
# #         c=fp.color(2), lw=3, ls=':')
# ax.plot(indices[1:-1] + np.diff(indices[1:])/2,
#          cum_slope_diff,
#          c=fp.color(2), lw=3, ls='-')
# ax.plot(indices[:-1] + np.diff(indices)/2,
#          dofs_per_cell_diff,
#          c=fp.color(2), lw=3, ls=':')
# ax2.plot(indices[1:], cum_dy, c=fp.color(2), lw=3, ls='--')
# ax2.axhline(0, c='grey', ls=':')

# plt.show()

# # n_record = []
# # dofs_record = []
# # dofs_per_cell_record = []
# # indices = np.concatenate((np.arange(10, 110, 10),
# #                           np.arange(150, 700, 50),
# #                           np.arange(800, 1000, 100),
# #                           np.arange(1200, 2000, 200)))
# # for i in indices:
# #     test1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
# #                                        clusters_plot,
# #                                        n_cells=[110, i],
# #                                        n_cluster_size=[1, 2])
# #     n_record.append(i)
# #     dofs_record.append(test1_ms.dofs.sum())
# #     dofs_per_cell_record.append(test1_ms.dofs.sum()/test1_ms.nstate)

# # plt.plot(n_record, dofs_per_cell_record, c='black')

# # n_record = []
# # dofs_record = []
# # dofs_per_cell_record = []
# # indices = np.concatenate((np.arange(10, 110, 10),
# #                           np.arange(150, 700, 50),
# #                           np.arange(800, 1000, 100),
# #                           np.arange(1200, 2000, 200)))
# # for i in indices:
# #     test1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
# #                                        clusters_plot,
# #                                        n_cells=[110, i],
# #                                        n_cluster_size=[1, 4])
# #     n_record.append(i)
# #     dofs_record.append(test1_ms.dofs.sum())
# #     dofs_per_cell_record.append(test1_ms.dofs.sum()/test1_ms.nstate)

# # plt.plot(n_record, dofs_per_cell_record, c='blue')

# # n_record = []
# # dofs_record = []
# # dofs_per_cell_record = []
# # indices = np.concatenate((np.arange(20, 110, 10),
# #                           np.arange(150, 700, 50),
# #                           np.arange(800, 1000, 100),
# #                           np.arange(1200, 1338, 200)))
# # for i in indices:
# #     test1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
# #                                        clusters_plot,
# #                                        n_cells=[110, 650, i],
# #                                        n_cluster_size=[1, 4, 16])
# #     n_record.append(i)
# #     dofs_record.append(test1_ms.dofs.sum())
# #     dofs_per_cell_record.append(test1_ms.dofs.sum()/test1_ms.nstate)

# # plt.plot(n_record, dofs_per_cell_record, c='green')

# # n_record = []
# # dofs_record = []
# # dofs_per_cell_record = []
# # indices = np.concatenate((np.arange(40, 110, 10),
# #                           np.arange(150, 700, 50),
# #                           np.arange(800, 1000, 100),
# #                           np.arange(1200, 1338, 200)))
# # for i in indices:
# #     test1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
# #                                        clusters_plot,
# #                                        n_cells=[110, 650, i],
# #                                        n_cluster_size=[1, 4, 32])
# #     n_record.append(i)
# #     dofs_record.append(test1_ms.dofs.sum())
# #     dofs_per_cell_record.append(test1_ms.dofs.sum()/test1_ms.nstate)

# # plt.plot(n_record, dofs_per_cell_record, c='orange')

# # n_record = []
# # dofs_record = []
# # dofs_per_cell_record = []
# # indices = np.concatenate((np.arange(50, 110, 10),
# #                           np.arange(150, 700, 50),
# #                           np.arange(800, 1000, 100),
# #                           np.arange(1200, 1338, 200)))
# # for i in indices:
# #     test1_ms = est0.update_jacobian_ms(true.k, true.xa_abs, true.sa_vec,
# #                                        clusters_plot,
# #                                        n_cells=[110, 650, i],
# #                                        n_cluster_size=[1, 4, 50])
# #     n_record.append(i)
# #     dofs_record.append(test1_ms.dofs.sum())
# #     dofs_per_cell_record.append(test1_ms.dofs.sum()/test1_ms.nstate)

# # plt.plot(n_record, dofs_per_cell_record, c='purple')

# plt.show()

#----------------------------------------------------------------------------#
#  Figures                                                                   #
#----------------------------------------------------------------------------#

################################################
### FIGURE 01: RANK AND DIMENSION FLOW CHART ###
################################################

# def flow_chart_settings(ax):
#     ax.add_feature(cartopy.feature.OCEAN, facecolor='white', zorder=2)
#     ax.coastlines(color='grey', zorder=5)
#     ax.outline_patch.set_visible(False)
#     return ax

# # Original dimension
# fig01a, ax = est0.plot_multiscale_grid(clusters_plot, colors='0.5', zorder=3,
#                                        fig_kwargs=small_fig_kwargs,
#                                        map_kwargs=small_map_kwargs)
# ax = flow_chart_settings(ax)
# fp.save_fig(fig01a, loc=plots, name='fig01a_dimn_rankn')

# # Reduced rank
# true.evec_sum = true.evecs[:, :3].sum(axis=1)
# fig01b, ax, c = true.plot_state('evec_sum', clusters_plot,
#                                 title='', cbar=False,  cmap='RdBu_r',
#                                 vmin=-0.1, vmax=0.1, default_value=0,
#                                 fig_kwargs=small_fig_kwargs,
#                                 map_kwargs=small_map_kwargs)
# ax = flow_chart_settings(ax)
# fp.save_fig(fig01b, loc=plots, name='fig01b_dimn_rankk')

# # Reduced rank and dimension (not aggregate)
# for i in range(3):
#     fig01c, ax, c = true.plot_state(('evecs', i), clusters_plot,
#                                     title='', cbar=False, cmap='RdBu_r',
#                                     vmin=-0.1, vmax=0.1,default_value=0,
#                                     fig_kwargs=small_fig_kwargs,
#                                     map_kwargs=small_map_kwargs)
#     ax = flow_chart_settings(ax)
#     fp.save_fig(fig01c, loc=plots, name='fig01c_evec' + str(i))

# # Reduced dimension (aggregate)
# fig01d, ax = est1_ms.plot_multiscale_grid(clusters_plot,
#                                           colors='0.5', zorder=3,
#                                           fig_kwargs=small_fig_kwargs,
#                                           map_kwargs=small_map_kwargs)
# ax = flow_chart_settings(ax)
# fp.save_fig(fig01d, loc=plots, name='fig01d_dimk_rankk_ms')

#########################################################################
### FIGURE 02: AVERAGING KERNEL SENSITIVITY TO PRIOR AND OBSERVATIONS ###
#########################################################################

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
# avker_kwargs['cbar_kwargs'] = avker_cbar_kwargs
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

##################################
### FIGURE 03: MULTISCALE GRID ###
##################################
# fig03, ax = est2_ms.plot_multiscale_grid(clusters_plot, colors='0.5', zorder=3,
#                                          title='Multiscale Grid',
#                                          fig_kwargs=small_fig_kwargs)
# fp.save_fig(fig03, loc=plots, name='fig03_est2_ms_grid')

###############################################
### FIGURE 04 : CONSOLIDATED POSTERIOR PLOT ###
###############################################
# fig04a, ax04a = fp.get_figax(rows=1, cols=3, maps=True,
#                              lats=clusters_plot.lat, lons=clusters_plot.lon)
# fig04b, ax04b = fp.get_figax(rows=1, cols=3, maps=True,
#                              lats=clusters_plot.lat, lons=clusters_plot.lon)

# def add_dofs_subtitle(inversion_object, ax):
#     subtitle = ('%d DOFS (%.2f/cell)'
#                 % (np.trace(inversion_object.a),
#                   (np.trace(inversion_object.a)/inversion_object.nstate)))
#     ax.text(0.5, 1.15, subtitle,
#             fontsize=config.SUBTITLE_FONTSIZE*config.SCALE,
#             ha='center', transform=ax.transAxes)
#     return ax

# title_kwargs = {'y' : 1.3}
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
# quantities = ['', '_long', '']

# for i, inv in enumerate([true, est2_ms, est2]):
#     state_kwargs['title'] = titles[i]
#     state_kwargs['fig_kwargs'] = {'figax' : [fig04a, ax04a[i]]}
#     dofs_kwargs['title'] = titles[i]
#     dofs_kwargs['fig_kwargs'] = {'figax' : [fig04b, ax04b[i]]}

#     # Posterior emissions
#     fig04a, ax04a[i], ca = inv.plot_state('xhat' + quantities[i],
#                                          clusters_plot, **state_kwargs)
#     ax04a[i] = add_dofs_subtitle(inv, ax04a[i])

#     # Averaging kernel sensitivities
#     fig04b, ax04b[i], cb = inv.plot_state('dofs' + quantities[i],
#                                          clusters_plot, **dofs_kwargs)
#     ax04b[i] = add_dofs_subtitle(inv, ax04b[i])

# # Polishing posterior emissions
# # Colorbar
# cax = fp.add_cax(fig04a, ax04a)
# cbar = fig04a.colorbar(ca, cax=cax, **state_cbar_kwargs)
# cbar = fp.format_cbar(cbar, cbar_title='Scaling Factors')

# # Label
# ax04a[0].text(-0.3, 0.5, 'Posterior\nEmissions',
#               fontsize=config.TITLE_FONTSIZE*config.SCALE,
#               rotation=90, ha='center', va='center',
#               transform=ax04a[0].transAxes)
# # Save
# fp.save_fig(fig04a, plots, 'fig04a_posterior_mean_summary')

# # Polishing averaging kernel sensitivities
# # Colorbar
# cax = fp.add_cax(fig04b, ax04b)
# cbar = fig04b.colorbar(cb, cax=cax, **dofs_cbar_kwargs)
# cbar = fp.format_cbar(cbar, cbar_title=r'$\partial\hat{x}/\partial x$')

# # Label
# ax04b[0].text(-0.3, 0.5, 'Averaging\nKernel',
#               fontsize=config.TITLE_FONTSIZE*config.SCALE,
#               rotation=90, ha='center', va='center',
#               transform=ax04b[0].transAxes)

# # Save
# fp.save_fig(fig04b, plots, 'fig04b_averaging_kernel_summary')

############################################################
### FIGURE 05: EST2_F POSTERIOR COMPARSION SCATTER PLOTS ###
############################################################
# fig05, _, _ = est2_f.full_analysis(true_f, clusters_plot)
# fp.save_fig(fig05, plots, 'fig05b_posterior_scattter_comparison')

#################################################
### FIGURE 06: REDUCED RANK SENSITIVITY TESTS ###
#################################################

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
# fig06, ax = fp.get_figax()
# cax = fp.add_cax(fig06, ax)
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
#         (n/2, n/2),
#         (3*n/4, 3*n/4)]
# # nm_summ[0, :] -= 1
# # nm_summ[1:, 0] -= 1
# cl = ax.contour(nm_summ, levels=3, colors=fp.color(3),
#                 linestyles='dotted')
# ax.clabel(cl, cl.levels,
#           inline=True,
#           manual=locs,
#           fmt='%d', fontsize=20)

# ax.scatter(mr2n(est1.model_runs),
#            mr2n(est2.model_runs-est1.model_runs),
#            zorder=10,
#            c=fp.color(3),
#            s=200, marker='*')

# # cbar = fig06.colorbar(cf, cax=cax, ticks=np.linspace(0, 1, 6))
# cbar = fig06.colorbar(cf, cax=cax,
#                     ticks=np.arange(0, true.dofs.sum(), 50))
# # cbar = fig06.colorbar(cf, cax=cax, ticks=np.arange(0, 2000, 500))

# # cbar = fp.format_cbar(cbar, r'r$^2$')
# cbar = fp.format_cbar(cbar, 'DOFS')
# # cbar = fp.format_cbar(cbar, 'Number of Constrained Cells')

# # Axis and tick labels
# ax = fp.add_labels(ax, 'First Update Model Runs', 'Second Update Model Runs')

# # ....This is still hard coded
# ax.set_xticks(np.arange(4, n, (n-4)/9))
# ax.set_xticklabels(np.arange(100, 1100, 100), fontsize=15)
# ax.set_xlim(0, n-1)

# ax.set_yticks(np.arange(4, n, (n-4)/9))
# ax.set_yticklabels(np.arange(100, 1100, 100), fontsize=15)
# ax.set_ylim(0, n-1)

# # fp.save_fig(fig, plots, 'fig28c_r2_comparison')
# fp.save_fig(fig06, plots, 'fig06_dofs_comparison')

####################################################
### FIGURE 37 : MULTISCALE GRID CONVERGENCE TEST ###
####################################################

