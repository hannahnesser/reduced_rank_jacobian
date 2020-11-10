import copy
import os
from os.path import join
import sys
import math
import itertools

sys.path.append('./python/')
import inversion as inv
import format_plots as fp
import config

import xarray as xr
import numpy as np

import pandas as pd
from scipy.sparse import diags, identity
from scipy import linalg, stats

from matplotlib import cm, colorbar, colors, rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import matplotlib.table as tbl
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.patch import geos_to_path

#########################
### PLOTTING DEFAULTS ###
#########################

# rcParams default
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'AppleGothic'
rcParams['font.size'] = config.LABEL_FONTSIZE*config.SCALE
rcParams['text.usetex'] = True
# rcParams['mathtext.fontset'] = 'stixsans'
rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
rcParams['axes.titlepad'] = 10

# Colormaps
c = plt.cm.get_cmap('inferno', lut=10)
plasma_trans = fp.cmap_trans('plasma')
plasma_trans_r = fp.cmap_trans('plasma_r')
rdbu_trans = fp.cmap_trans_center('RdBu_r', nalpha=70)

# Small (i.e. non-default) figure settings
small_fig_kwargs = {'max_width' : 3.25,
                    'max_height' : 3}
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

# Clusters
clusters = xr.open_dataarray(join(inputs, 'clusters_1x125.nc'))
clusters_plot = xr.open_dataarray(join(inputs, 'clusters_1x125_plot.nc'))

# Estimated, true, and reduced-dimension Jacobian
k_est = xr.open_dataarray(join(inputs, 'k_est.nc'))
k_est_sparse = xr.open_dataarray(join(inputs, 'k_est_sparse.nc'))
k_true = xr.open_dataarray(join(inputs, 'k_true.nc'))
k_rd = pd.read_csv(join(inputs, 'k_rd.csv'), header=None).to_numpy()

# Native-resolution prior and prior error
xa = xr.open_dataarray(join(inputs, 'xa.nc'))
xa_abs = xr.open_dataarray(join(inputs, 'xa_abs.nc'))
sa_vec = xr.open_dataarray(join(inputs, 'sa_vec.nc'))

# Reduced-dimension quantitites
state_vector_rd = pd.read_csv(join(inputs, 'state_vector_rd.csv'),
                              header=None).to_numpy().reshape(-1,)
xa_rd = pd.read_csv(join(inputs, 'xa_rd.csv'),
                    header=None).to_numpy().reshape(-1,)
xa_abs_rd = pd.read_csv(join(inputs, 'xa_abs_rd.csv'),
                        header=None).to_numpy().reshape(-1,)
sa_vec_rd = pd.read_csv(join(inputs, 'sa_vec_rd.csv'),
                        header=None).to_numpy().reshape(-1,)
xhat_long_rd = pd.read_csv(join(inputs, 'xhat_long_rd.csv'),
                           header=None).to_numpy().reshape(-1,)
dofs_long_rd = pd.read_csv(join(inputs, 'dofs_long_rd.csv'),
                           header=None).to_numpy().reshape(-1,)

# Vectorized observations and error
y = xr.open_dataarray(join(inputs, 'y.nc'))
y_base = xr.open_dataarray(join(inputs, 'y_base.nc'))
so_vec = xr.open_dataarray(join(inputs, 'so_vec.nc'))

# Gridded observations
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

#####################
### TRUE JACOBIAN ###
#####################

# Create a true Reduced Rank Jacobian object
true = inv.ReducedRankJacobian(k_true.values, xa.values, sa_vec.values,
                               y.values, y_base.values, so_vec.values)
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
### INITIAL JACOBIAN ###
########################

est0 = inv.ReducedRankJacobian(k_est.values, xa.values, sa_vec.values,
                               y.values, y_base.values, so_vec.values)
est0.xa_abs = xa_abs*1e3
est0.rf = RF
est0.edecomp()
est0.solve_inversion()

##################################
### REDUCED DIMENSION JACOBIAN ###
##################################
est_rd = inv.ReducedRankJacobian(k_rd, xa_rd, sa_vec_rd,
                                 y.values, y_base.values, so_vec.values)
est_rd.rf = RF*est_rd.nstate/true.nstate
est_rd.xa_abs = xa_abs_rd
est_rd.xhat_long = xhat_long_rd
est_rd.dofs_long = dofs_long_rd
est_rd.state_vector = state_vector_rd
est_rd.model_runs = 534
est_rd.solve_inversion()

print(est_rd.dofs.sum())

#############################
### REDUCED RANK JACOBIAN ###
#############################

# First set of updates
est1 = est0.update_jacobian(true.k, snr=2.5)

# Second set
est2 = est1.update_jacobian(true.k, rank=est0.get_rank(pct_of_info=0.985))
est2.model_runs += 1

# second set
est1a = est0.update_jacobian(true.k,
                              snr=1)
est2a = est1a.update_jacobian(true.k, rank=est2.model_runs-est1a.model_runs)

# third set
est1b = est0.update_jacobian(true.k,
                              snr=4)
est2b = est1b.update_jacobian(true.k, rank=est2.model_runs-est1b.model_runs)

# Filter
mask = np.diag(est2.a) > 0.01
est2_f, true_f = est2.filter(true, mask)

# Print diagnostics
print('-----------------------')
print('MODEL RUNS: ', est2.model_runs)
print('CONSTRAINED CELLS: ', len(est2_f.xhat))
print('DOFS: ', np.trace(est2_f.a))
print('-----------------------\n')

###############################################
### REDUCED-RANK JACOBIAN SENSITIVITY TESTS ###
###############################################

# Open summary files
r2_summ = np.load(join(inputs, 'r2_summary.npy'))
nc_summ = np.load(join(inputs, 'nc_summary.npy'))
nm_summ = np.load(join(inputs, 'nm_summary.npy'))
dofs_summ = np.load(join(inputs, 'dofs_summary.npy'))

###############################################
### FIGURE 1: RANK AND DIMENSION FLOW CHART ###
###############################################

# # General settings
# def flow_chart_settings(ax):
#     ax.add_feature(cartopy.feature.OCEAN, facecolor='white', zorder=2)
#     ax.coastlines(color='grey', zorder=5, linewidth=0.5)
#     ax.outline_patch.set_visible(False)
#     return ax

# title_kwargs = {'y' : 1,
#                 'pad' : (1.5*config.TITLE_PAD +
#                          1*config.SUBTITLE_FONTSIZE*config.SCALE)}

# ## Original dimension
# fig1a, ax = est0.plot_multiscale_grid(clusters_plot, colors='0.5', zorder=3,
#                                       linewidths=0.5,
#                                       fig_kwargs=small_fig_kwargs,
#                                       map_kwargs=small_map_kwargs)
# ax = flow_chart_settings(ax)
# ax = fp.add_title(ax, 'Native resolution', **title_kwargs)
# ax = fp.add_subtitle(ax, r'dimension $n$, rank $> k$')
# fp.save_fig(fig1a, loc=plots, name='fig1a_dimn_rankn')

# ## Reduced rank
# true.evec_sum = true.evecs[:, :3].sum(axis=1)
# fig1b, ax, c = true.plot_state('evec_sum', clusters_plot,
#                                 title='', cbar=False,  cmap='RdBu_r',
#                                 vmin=-0.1, vmax=0.1, default_value=0,
#                                 fig_kwargs=small_fig_kwargs,
#                                 map_kwargs=small_map_kwargs)
# ax = flow_chart_settings(ax)
# ax = fp.add_subtitle(ax, 'Reduced rank',
#                      fontsize=config.TITLE_FONTSIZE*config.SCALE,
#                      y=0, va='top', xytext=(0, -config.TITLE_PAD))
# ax = fp.add_subtitle(ax, r'dimension $n$, rank $k$',
#                      y=0, va='top',
#                      xytext=(0,
#                              -(1.5*config.TITLE_PAD +
#                                config.TITLE_FONTSIZE*config.SCALE)))
# fp.save_fig(fig1b, loc=plots, name='fig1b_dimn_rankk')

# ## Reduced rank and dimension (not aggregate)
# # Base images
# for i in range(3):
#     fig1c, ax, c = true.plot_state(('evecs', i), clusters_plot,
#                                     title='', cbar=False, cmap='RdBu_r',
#                                     vmin=-0.1, vmax=0.1,default_value=0,
#                                     fig_kwargs=small_fig_kwargs,
#                                     map_kwargs=small_map_kwargs)
#     ax = flow_chart_settings(ax)
#     fp.save_fig(fig1c, loc=plots, name='fig1c_evec' + str(i))

# # Reduced dimension (aggregate)
# fig1d, ax = est_rd.plot_multiscale_grid(clusters_plot,
#                                          colors='0.5', zorder=3,
#                                          linewidths=0.5,
#                                          fig_kwargs=small_fig_kwargs,
#                                          map_kwargs=small_map_kwargs)
# ax = flow_chart_settings(ax)
# ax = fp.add_title(ax, 'Reduced dimension', **title_kwargs)
# ax = fp.add_subtitle(ax, r'dimension $k$, rank $k$')
# fp.save_fig(fig1d, loc=plots, name='fig1d_dimk_rankk_ms')

# # # Compile plots
# # fig1, ax1 = fp.get_figax(rows=2, cols=2, maps=True,
# #                          lats=clusterS_plot.lat, lons=clusters_plot.lon)

# # Some labels
# fig1e, ax = fp.get_figax()
# ax.annotate(r'$\mathbf{\Gamma} (k \times n)$',
#             xy=(0.5, 0.75), xycoords='axes fraction',
#             fontsize=config.SUBTITLE_FONTSIZE*config.SCALE)
# ax.annotate(r'$\mathbf{\Gamma}^* (n \times k)$',
#             xy=(0.5, 0.5), xycoords='axes fraction',
#             fontsize=config.SUBTITLE_FONTSIZE*config.SCALE)
# ax.annotate(r'$\mathbf{\Pi} (n \times n)$',
#             xy=(0.5, 0.25), xycoords='axes fraction',
#             fontsize=config.SUBTITLE_FONTSIZE*config.SCALE)
# fp.save_fig(fig1e, loc=plots, name='fig1e_labels')


########################################################################
### FIGURE 2: AVERAGING KERNEL SENSITIVITY TO PRIOR AND OBSERVATIONS ###
########################################################################

# True averaging kernel
title = 'Native-resolution averaging\nkernel sensitivities'
avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
                'cbar_kwargs' : avker_cbar_kwargs,
                'fig_kwargs' : small_fig_kwargs,
                'map_kwargs' : small_map_kwargs}
fig2a, ax, c = true.plot_state('dofs', clusters_plot, title=title,
                                **avker_kwargs)
ax.text(0.025, 0.05, 'DOFS = %d' % np.trace(true.a),
        fontsize=config.LABEL_FONTSIZE*config.SCALE,
        transform=ax.transAxes)
fp.save_fig(fig2a, plots, 'fig2a_true_averaging_kernel')

# Initial estimate averaging kernel
title = 'Initial estimate averaging\nkernel sensitivities'
avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
avker_kwargs['cbar_kwargs'] = avker_cbar_kwargs
fig2b, ax, c = est0.plot_state('dofs', clusters_plot, title=title,
                                **avker_kwargs)
ax.text(0.025, 0.05, 'DOFS = %d' % np.trace(est0.a),
        fontsize=config.LABEL_FONTSIZE*config.SCALE,
        transform=ax.transAxes)
fp.save_fig(fig2b, plots, 'fig2b_est0_averaging_kernel')

# Prior error
true.sd_vec = true.sa_vec**0.5
true.sd_vec_abs = true.sd_vec*true.xa_abs
cbar_kwargs = {'title' : 'Tg/month'}
fig2c, ax, c = true.plot_state('sd_vec_abs', clusters_plot,
                                title='Prior error standard deviation',
                                cmap=fp.cmap_trans('viridis'),
                                vmin=0, vmax=15,
                                fig_kwargs=small_fig_kwargs,
                                cbar_kwargs=cbar_kwargs,
                                map_kwargs=small_map_kwargs)
fp.save_fig(fig2c, plots, 'fig2c_prior_error')

# Observational density
lat_res = np.diff(clusters_plot.lat)[0]
lat_edges = np.append(clusters_plot.lat - lat_res/2,
                      clusters_plot.lat[-1] + lat_res/2)
# lat_edges = lat_edges[::2]
obs['lat_edges'] = pd.cut(obs['lat'], lat_edges, precision=4)

lon_res = np.diff(clusters_plot.lon)[0]
lon_edges = np.append(clusters_plot.lon - lon_res/2,
                      clusters_plot.lon[-1] + lon_res/2)
# lon_edges = lon_edges[::2]
obs['lon_edges'] = pd.cut(obs['lon'], lon_edges, precision=4)

obs_density = obs.groupby(['lat_edges', 'lon_edges']).count()
obs_density = obs_density['Nobs'].reset_index()
obs_density['lat'] = obs_density['lat_edges'].apply(lambda x: x.mid)
obs_density['lon'] = obs_density['lon_edges'].apply(lambda x: x.mid)
obs_density = obs_density.set_index(['lat', 'lon'])['Nobs']
obs_density = obs_density.to_xarray()

title = 'GOSAT observation density\n(July 2009)'
viridis_trans_long = fp.cmap_trans('viridis', nalpha=90, ncolors=300)
cbar_kwargs = {'ticks' : np.arange(0, 25, 5),
               'title' : 'Count'}
fig2d, ax, c = true.plot_state_format(obs_density, title=title,
                                       vmin=0, vmax=10, default_value=0,
                                       cmap=viridis_trans_long,
                                       fig_kwargs=small_fig_kwargs,
                                       cbar_kwargs=cbar_kwargs,
                                       map_kwargs=small_map_kwargs)
fp.save_fig(fig2d, plots, 'fig2d_gosat_obs_density')

##############################################
### FIGURE 3 : CONSOLIDATED POSTERIOR PLOT ###
##############################################

fig3a, ax3a = fp.get_figax(rows=1, cols=3, maps=True,
                             lats=clusters_plot.lat, lons=clusters_plot.lon)
fig3b, ax3b = fp.get_figax(rows=1, cols=3, maps=True,
                             lats=clusters_plot.lat, lons=clusters_plot.lon)

def add_dofs_subtitle(inversion_object, ax,
                      state_vector_element_string='cell'):
    subtitle = ('%d DOFS (%.2f/%s)\n%d model simulations'
                % (round(np.trace(inversion_object.a)),
                  (np.trace(inversion_object.a)/inversion_object.nstate),
                  state_vector_element_string,
                  inversion_object.model_runs + 1))
    ax = fp.add_subtitle(ax, subtitle)
    return ax

title_kwargs = {'y' : 1,
                'pad' : (1.5*config.TITLE_PAD +
                         2*config.SUBTITLE_FONTSIZE*config.SCALE)}
state_cbar_kwargs = {'ticks' : np.arange(-1, 4, 1)}
dofs_cbar_kwargs = {'ticks' : np.arange(0, 1.1, 0.2)}
state_kwargs = {'default_value' : 1, 'cmap' : 'RdBu_r',
                'vmin' : -1, 'vmax' : 3,
                'cbar' : False, 'cbar_kwargs' : state_cbar_kwargs,
                'title_kwargs' : title_kwargs, 'map_kwargs' : small_map_kwargs}
dofs_kwargs =  {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
                'cbar' : False, 'cbar_kwargs' : dofs_cbar_kwargs,
                'title_kwargs' : title_kwargs, 'map_kwargs' : small_map_kwargs}
titles = ['Native resolution', 'Reduced dimension', 'Reduced rank']
sve_string = ['cell', 'cluster', 'cell']
quantities = ['', '_long', '']

for i, inv in enumerate([true, est_rd, est2]):
    state_kwargs['title'] = titles[i]
    state_kwargs['fig_kwargs'] = {'figax' : [fig3a, ax3a[i]]}
    dofs_kwargs['title'] = '' #'titles[i]'
    dofs_kwargs['fig_kwargs'] = {'figax' : [fig3b, ax3b[i]]}

    # Posterior emissions
    fig3a, ax3a[i], ca = inv.plot_state('xhat' + quantities[i],
                                         clusters_plot, **state_kwargs)
    ax3a[i] = add_dofs_subtitle(inv, ax3a[i], sve_string[i])

    # Averaging kernel sensitivities
    fig3b, ax3b[i], cb = inv.plot_state('dofs' + quantities[i],
                                         clusters_plot, **dofs_kwargs)
    # ax3b[i] = add_dofs_subtitle(inv, ax3b[i], sve_string[i])

# Polishing posterior emissions
# Colorbar
cax = fp.add_cax(fig3a, ax3a)
cbar = fig3a.colorbar(ca, cax=cax, **state_cbar_kwargs)
cbar = fp.format_cbar(cbar, cbar_title='Scaling factors')

# Label
ax3a[0].text(-0.3, 0.5, 'Posterior\nscaling\nfactors',
              fontsize=config.TITLE_FONTSIZE*config.SCALE,
              rotation=90, ha='center', va='center',
              transform=ax3a[0].transAxes)
# Save
fp.save_fig(fig3a, plots, 'fig3a_posterior_mean_summary')

# Polishing averaging kernel sensitivities
# Colorbar
cax = fp.add_cax(fig3b, ax3b)
cbar = fig3b.colorbar(cb, cax=cax, **dofs_cbar_kwargs)
cbar = fp.format_cbar(cbar, cbar_title=r'$\partial\hat{x}_i/\partial x_i$')

# Label
ax3b[0].text(-0.3, 0.5, 'Averaging\nkernel\nsensitivities',
              fontsize=config.TITLE_FONTSIZE*config.SCALE,
              rotation=90, ha='center', va='center',
              transform=ax3b[0].transAxes)

# Save
fp.save_fig(fig3b, plots, 'fig3b_averaging_kernel_summary')

#################################################
### FIGURE 4: REDUCED RANK SENSITIVITY TESTS ###
#################################################

min_mr = 0
max_mr = 1000
increment = 25
n = int((max_mr - min_mr)/increment) + 1

def mr2n(model_runs,
         min_mr=min_mr, max_mr=max_mr, max_n=n):
    '''This function converts a number of model runs
    to an index along an axis from 0 to n'''
    x0 = min_mr
    y0 = 0
    slope = (max_n - 1)/(max_mr - min_mr)
    func = lambda mr : slope*(mr - x0) + y0
    return func(model_runs)

# R2 plot
# fig4, ax = fp.get_figax()
figsize = fp.get_figsize(aspect=0.665, rows=1, cols=1)
fig4, ax = plt.subplots(2, 1,
                         figsize=figsize,
                         gridspec_kw={'height_ratios': [1, 4]})
plt.subplots_adjust(hspace=0.4)
cax = fp.add_cax(fig4, ax[1])
# ax = fp.add_title(ax, r'Posterior Emissions r$^2$'))
ax[0] = fp.add_title(ax[0], 'Reduced Rank DOFS', pad=config.TITLE_PAD*2)
# ax = fp.add_title(ax, 'Number of Constrained Grid Cells')

# Plot DOFS contours (shaded)
# cf = ax.contourf(r2_summ, levels=np.linspace(0, 1, 25), vmin=0, vmax=1)
cf = ax[1].contourf(dofs_summ,
                    levels=np.linspace(0, true.dofs.sum(), 25),
                    vmin=0, vmax=200, cmap='plasma')
# cf = ax.contourf(nc_summ, levels=np.arange(0, 2000, 100),
#                  vmin=0, vmax=2000)

# Get the ridge values
dofs_summ_short = dofs_summ[1:, 1:]
dofs_summ_flat = dofs_summ_short.flatten().reshape((-1, 1))
nm_summ_short = nm_summ[1:, 1:].flatten().reshape((-1, 1))
summ_short = np.concatenate([nm_summ_short, dofs_summ_flat], axis=1)
summ_short = pd.DataFrame(data=summ_short, columns=['model_runs', 'DOFS'])
summ_short = summ_short.groupby('model_runs').max().reset_index(drop=True)

x, y = np.where(np.isin(dofs_summ_short, summ_short['DOFS']))

dofs_summ_short = dofs_summ[1:, 1:]
nm_summ_short = nm_summ[1:, 1:]
x = []
y = []
z = []
np.set_printoptions(precision=0)
for i in np.unique(nm_summ_short):
    mask_nm = np.where(nm_summ_short == i)
    mask_dofs_max = np.argmax(dofs_summ_short[mask_nm])
    dofs_max = dofs_summ_short[mask_nm][mask_dofs_max]
    yi, xi = np.where((dofs_summ_short == dofs_max) & (nm_summ_short == i))
    x.append(xi[0] + 1)
    y.append(yi[0] + 1)
    z.append(dofs_max)

# ax[1].plot(x, y, c='0.25')
x = np.unique(nm_summ_short)
z = np.array(z)
ax[0].plot(x[x <= 1000], z[x <= 1000], c='0.25', ls='--')
ax[0].scatter(est2.model_runs, est2.dofs.sum(),
              zorder=10, c='0.25', s=75, marker='*')
ax[0].set_xticks(np.arange(200, 1010, 200))
ax[0].set_xlim(0, 1000)
ax[0].set_ylim(50, 216)
ax[0].set_facecolor('0.98')
ax[0] = fp.add_labels(ax[0],
                       xlabel='Total Model Runs',
                       ylabel='Optimal\nDOFS',
                       labelpad=config.LABEL_PAD/2)

# Plot number of model run contours (lines)
levels = [250, 750, 1250]
# locs = [(n/8, n/8),
#         (n/2, n/2),
#         (5*n/8, 5*n/8)]
locs = [(mr2n(l)/2, mr2n(l)/2) for l in levels]

# nm_summ[0, :] -= 1
# nm_summ[1:, 0] -= 1
cl = ax[1].contour(nm_summ, levels=levels, colors='white',
                   linestyles='dotted')
ax[1].clabel(cl, cl.levels, inline=True, manual=locs, fmt='%d',
             fontsize=config.TICK_FONTSIZE*config.SCALE)
cl = ax[1].contour(nm_summ, levels=[est2.model_runs], colors='0.25',
                    linestyles='dashed')
ax[1].clabel(cl, cl.levels, inline=True,
             manual=[(mr2n(est2.model_runs)/2, mr2n(est2.model_runs)/2)],
             fmt='%d',
             fontsize=config.TICK_FONTSIZE*config.SCALE)

ax[1].scatter(mr2n(est1.model_runs),
              mr2n(est2.model_runs-est1.model_runs),
              zorder=10,
              c='0.25',
              s=75, marker='*')
ax[1].scatter(mr2n(est1a.model_runs),
              mr2n(est2a.model_runs-est1a.model_runs),
              zorder=10, c='0.25', s=75, marker='.')
ax[1].scatter(mr2n(est1b.model_runs),
              mr2n(est2b.model_runs-est1b.model_runs),
              zorder=10, c='0.25', s=75, marker='.')

# cbar = fig4.colorbar(cf, cax=cax, ticks=np.linspace(0, 1, 6))
cbar = fig4.colorbar(cf, cax=cax,
                    ticks=np.arange(0, true.dofs.sum(), 50))
# cbar = fig4.colorbar(cf, cax=cax, ticks=np.arange(0, 2000, 500))

# cbar = fp.format_cbar(cbar, r'r$^2$')
cbar = fp.format_cbar(cbar, 'DOFS')
# cbar = fp.format_cbar(cbar, 'Number of Constrained Cells')

# Axis and tick labels
ax[1] = fp.add_labels(ax[1],
                      'First Update Model Runs',
                      'Second Update Model Runs',
                      labelpad=config.LABEL_PAD/2)

# ....This is still hard coded
ax[1].set_xticks(np.arange(4, n, (n-4)/9))
ax[1].set_xticklabels(np.arange(100, 1100, 100),
                      fontsize=config.TICK_FONTSIZE*config.SCALE)
ax[1].set_xlim(0, (n-1)*750/975)

ax[1].set_yticks(np.arange(4, n, (n-4)/9))
ax[1].set_yticklabels(np.arange(100, 1100, 100),
                      fontsize=config.TICK_FONTSIZE*config.SCALE)
ax[1].set_ylim(0, (n-1)*750/975)
ax[1].set_aspect('equal')

fp.save_fig(fig4, plots, 'fig4_dofs_comparison')

############################################################
### FIGURE 5: EST2_F POSTERIOR COMPARSION SCATTER PLOTS ###
############################################################
fig5, _, _ = est2_f.full_analysis(true_f, clusters_plot)

ax = fig5.axes
ax[0].set_xticks([0, 5])
ax[0].set_yticks([0, 5])

ax[1].set_xticks([-10, 0, 10])
ax[1].set_yticks([-10, 0, 10])

ax[2].set_xticks([0.2, 0.4])
ax[2].set_yticks([0.2, 0.4])

ax[3].set_xticks([0, 0.5, 1])
ax[3].set_yticks([0, 0.5, 1])

fp.save_fig(fig5, plots, 'fig5_posterior_scattter_comparison')

