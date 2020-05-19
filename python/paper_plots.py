import copy
import os
from os.path import join
import sys
import math
import itertools

sys.path.append('./python/')
import inversion as inv
import jacobian as j
# import inv_plot

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
import format_plots as fp

c = plt.cm.get_cmap('inferno', lut=10)

plasma_trans = fp.cmap_trans('plasma')
plasma_trans_r = fp.cmap_trans('plasma_r')

SCALE = fp.SCALE
# BASE_FIG_SIZE = 6
BASE_WIDTH = fp.BASE_WIDTH
BASE_HEIGHT = fp.BASE_HEIGHT

# Fontsizes
TITLE_FONTSIZE = fp.TITLE_FONTSIZE
SUBTITLE_FONTSIZE = fp.SUBTITLE_FONTSIZE
LABEL_FONTSIZE = fp.LABEL_FONTSIZE
TICK_FONTSIZE = fp.TICK_FONTSIZE

# Position
TITLE_LOC = fp.TITLE_LOC
CBAR_PAD = fp.CBAR_PAD
LABEL_PAD = fp.LABEL_PAD
CBAR_LABEL_PAD = fp.CBAR_LABEL_PAD

# Other font details
rcParams['font.family'] = 'serif'
rcParams['font.size'] = LABEL_FONTSIZE*SCALE
rcParams['text.usetex'] = True

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
sa_vec = xr.open_dataarray(join(inputs, 'sa_vec.nc'))

# Load observations and error
y = xr.open_dataarray(join(inputs, 'y.nc'))
y_base = xr.open_dataarray(join(inputs, 'y_base.nc'))
so_vec = xr.open_dataarray(join(inputs, 'so_vec.nc'))

# Load the gridded emissions
emis = j.get_delta_emis(clusters_long=clusters,
                        emis_loc=join(inputs, 'base_emis.nc'))

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
true.xa_abs = emis*1e3
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
est0.xa_abs = emis *1e3
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
mask = np.diag(est2.a) > 0.05
est2_f, true_f = est2.filter(true, mask)

print('-----------------------')
print('MODEL RUNS: ', est2.model_runs)
print('CONSTRAINED CELLS: ', len(est2_f.xhat))
print('DOFS: ', np.trace(est2_f.a))
print('-----------------------')

# And multiscale grid results
est1_ms = est0.update_jacobian_ms(true.k, clusters_plot)

# fig, ax, c = est1_ms.plot_state('state_vector', clusters_plot,
#                                 cmap=plasma_trans,
#                                 title='State Vector')

#############################
### FIGURE 00: FLOW CHART ###
#############################

# Original dimension
# fig00a, ax = est0.plot_multiscale_grid(clusters_plot,
#                                        colors='0.5', zorder=3,
#                                        map_kwargs={'draw_labels' : False})
# ax.add_feature(cartopy.feature.OCEAN, facecolor='white', zorder=2)
# ax.coastlines(color='grey', zorder=5)
# ax.outline_patch.set_visible(False)
# fp.save_fig(fig00a, loc=plots, name='fig00a_dimn_rankn')

# # Reduced rank
# true.evec_sum = true.evecs[:, :3].sum(axis=1)
# fig00b, ax, c = true.plot_state('evec_sum', clusters_plot,
#                                 title='', cbar=False, cmap='RdBu_r',
#                                 vmin=-0.1, vmax=0.1,
#                                 default_value=0,
#                                 map_kwargs={'draw_labels' : False})
# ax.add_feature(cartopy.feature.OCEAN, facecolor='white', zorder=2)
# ax.coastlines(color='grey', zorder=4)
# ax.outline_patch.set_visible(False)
# fp.save_fig(fig00b, loc=plots, name='fig00b_dimn_rankk')

fig00c = plt.figure()
ax = fig00c.add_axes([0, 0, 1, 1], projection='3d')
# proj_ax = plt.figure().add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
# ax.projection = proj_ax.projection

# concat = lambda iterable: list(itertools.chain.from_iterable(iterable))
# target_projection = ccrs.PlateCarree()
# feature = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m')
# geoms = feature.geometries()
# geoms = [target_projection.project_geometry(geom, feature.crs)
#          for geom in geoms]
# paths = concat(geos_to_path(geom) for geom in geoms)
# polys = concat(path.to_polygons() for path in paths)

cmap = fp.cmap_trans_center('RdBu_r')
norm = colors.Normalize(vmin=-0.1, vmax=0.1)
for i in range(1):
    data = true.match_data_to_clusters(true.evecs[:, i], clusters_plot,
                                       default_value=0)
    x, y = np.meshgrid(data.lat, data.lon)
    colors = cmap(norm(data.values))
    z = np.zeros_like(x) + i
    # cs = proj_ax.pcolormesh(x, y, z,
    #                           transform=ccrs.PlateCarree(),
    #                           facecolors=colors, snap=True, zorder=0)
    fig, proj_ax, c = true.plot_state(('evecs', i), clusters_plot,
                                      title='', cbar=False, cmap='RdBu_r',
                                      vmin=-0.1, vmax=0.1,
                                      default_value=0,
                                      map_kwargs={'draw_labels' : False})
    ax.projection = proj_ax.projection
    cs = ax.pcolormesh(y, x, z,
                       # transform=ccrs.PlateCarree(),
                       # facecolors=colors,
                       cmap='RdBu_r', vmin=-0.1, vmax=0.1,
                       snap=True, zorder=0)

    # add map
    # clip_geom = proj_ax._get_extent_geom().buffer(0)
    fp.add_feature3d(ax, cartopy.feature.OCEAN, zs=i)
    fp.add_feature3d(ax, cartopy.feature.LAND, zs=i)
    fp.add_feature3d(ax, cartopy.feature.COASTLINE, zs=i)

ax.set_xlim(clusters_plot.lon.min(), clusters_plot.lon.max())
ax.set_ylim(clusters_plot.lat.min(), clusters_plot.lat.max())
ax.set_zlim(bottom=0, top=3.5)
fp.save_fig(fig00c, loc=plots, name='fig00c_dimk_rankk')


# # Reduced dimension (aggregate)
# fig00d, ax = est1_ms.plot_multiscale_grid(clusters_plot,
#                                           colors='0.5', zorder=3,
#                                           map_kwargs={'draw_labels' : False})
# ax.add_feature(cartopy.feature.OCEAN, facecolor='white', zorder=2)
# ax.coastlines(color='grey', zorder=5)
# ax.outline_patch.set_visible(False)
# fp.save_fig(fig00d, loc=plots, name='fig00d_dimk_rankk_ms')


#####################################
### FIGURE 01: GOSAT OBSERVATIONS ###
#####################################
# fig01, ax = fp.make_axes(maps=True, lats=obs['lat'], lons=obs['lon'])

# col = ax.scatter(obs['lon'], obs['lat'], c=obs['GOSAT'],
#                cmap='inferno', vmin=1700, vmax=1800,
#                s=100)

# ax = fp.add_title(ax, r'GOSAT XCH$_4$ (July 2009)')
# ax = fp.format_map(ax, obs['lat'], obs['lon'])
# ax.set_xlim(clusters_plot.lon.min(), clusters_plot.lon.max())
# ax.set_ylim(clusters_plot.lat.min(), clusters_plot.lat.max())

# cax = fp.add_cax(fig01, ax)
# cbar = plt.colorbar(col, cax=cax)
# cbar = fp.format_cbar(cbar, 'XCH4 (ppb)')

# # Save plot
# fig01.savefig(join(plots, 'fig01_gosat_obs.png'),
#              bbox_inches='tight', dpi=300)
# print('Saved fig01_gosat_obs.png')

############################################
### FIGURE 02: GOSAT OBSERVATION DENSITY ###
############################################
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

# viridis_trans_long = fp.cmap_trans('viridis', nalpha=90, ncolors=300)
# cbar_kwargs = {'ticks' : np.arange(0, 25, 5),
#                'title' : 'Observation Count'}
# fig02, ax, c = true.plot_state_format(obs_density, default_value=0,
#                                       **{'vmin' : 0,
#                                          'vmax' : 10,
#                                          'cmap' : viridis_trans_long,
#                                          'title' : 'GOSAT Observation Density',
#                                          'cbar_kwargs' : cbar_kwargs})

# fig02.savefig(join(plots, 'fig02_gosat_obs_density.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig02_gosat_obs_density.png')

#################################
## FIGURE 03: PRIOR EMISSIONS ###
#################################
# cbar_kwargs = {'title' : 'Emissions (Gg/month)'}
# fig03, ax, c = true.plot_state('xa_abs', clusters_plot,
#                                **{'title' : 'Prior Emissions',
#                                   'cmap' : fp.cmap_trans('viridis'),
#                                   'cbar_kwargs' : cbar_kwargs})

# fig03.savefig(join(plots, 'fig03_prior_emissions.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig03_prior_emissions.png')

###############################
### FIGURE 03b: PRIOR ERROR ###
###############################
# true.sd_vec = true.sa_vec**0.5
# true.sd_vec_abs = true.sd_vec*true.xa_abs
# cbar_kwargs = {'title' : 'Emissions (Gg/month)'}
# fig03b, ax, c = true.plot_state('sd_vec_abs', clusters_plot,
#                                  **{'title' : 'Prior Error Standard Deviation',
#                                     'cmap' : fp.cmap_trans('viridis'),
#                                     'cbar_kwargs' : cbar_kwargs})

# fig03b.savefig(join(plots, 'fig03b_prior_error.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig03_prior_emissions.png')

######################################
### FIGURE 04: TRUE POSTERIOR MEAN ###
######################################
# cbar_kwargs = {'ticks' : np.arange(-1, 4, 1),
#                'title' : 'Scaling Factors'}
# fig04, ax, c = true.plot_state('xhat',
#                                clusters_plot,
#                                default_value=1,
#                                **{'title' : 'True Posterior Mean',
#                                   'cmap' : 'RdBu_r',
#                                   'vmin' : -1,
#                                   'vmax' : 3,
#                                   'cbar_kwargs' : cbar_kwargs})

# fig04.savefig(join(plots, 'fig04_true_posterior_mean.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig04_true_posterior_mean.png')

########################################
### FIGURE 05: TRUE AVERAGING KERNEL ###
########################################
# cbar_kwargs = {'title' : r'$\partial\hat{x}/\partial x$'}
# fig05, ax, c = true.plot_state('dofs', clusters_plot,
#                                **{'title' : 'True Averaging Kernel',
#                                   'cmap' : plasma_trans,
#                                   'vmin' : 0,
#                                   'vmax' : 1,
#                                   'cbar_kwargs' : cbar_kwargs})
# ax.text(0.025, 0.05, 'DOFS = %.2f' % np.trace(true.a),
#         fontsize=LABEL_FONTSIZE*SCALE,
#         transform=ax.transAxes)

# fig05.savefig(join(plots, 'fig05_true_averaging_kernel.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig05_true_averaging_kernel.png')

###########################################
### FIGURE 06: INITIAL AVERAGING KERNEL ###
###########################################
# cbar_kwargs = {'title' : r'$\partial\hat{x}/\partial x$'}
# fig06, ax, c = est0.plot_state('dofs', clusters_plot,
#                                **{'title' : 'Initial Averaging Kernel',
#                                   'cmap' : plasma_trans,
#                                   'vmin' : 0,
#                                   'vmax' : 1,
#                                   'cbar_kwargs' : cbar_kwargs})
# ax.text(0.025, 0.05, 'DOFS = %.2f' % np.trace(est0.a),
#         fontsize=LABEL_FONTSIZE*SCALE,
#         transform=ax.transAxes)

# fig06.savefig(join(plots, 'fig06_est0_averaging_kernel.png'),
#             bbox_inches='tight', dpi=300)
# print('Saved fig06_est0_averaging_kernel.png')

#####################################
### FIGURE 07 - 09: EST0 ANALYSIS ###
#####################################
# fig07, fig08, fig09 = est0.full_analysis(true, clusters_plot)
# fig07.suptitle('Initial Estimate', fontsize=TITLE_FONTSIZE*SCALE,
#                y=1.5)
# fig07.savefig(join(plots, 'fig07_est0_comparison.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig07_est0_comparison.png')
# fig08.savefig(join(plots, 'fig08_est0_spectrum.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig08_est0_spectrum.png')
# fig09.savefig(join(plots, 'fig09_est0_eigenvectors.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig09_est0_eigenvectors.png')

#####################################
### FIGURE 10 - 12: EST1 ANALYSIS ###
#####################################
# fig10, fig11, fig12 = est1.full_analysis(true, clusters_plot)
# # fig10.suptitle('First Update', fontsize=TITLE_FONTSIZE*SCALE,
# #                y=1.5)
# fig10.axes[0].text(-0.65, 0.5, 'First Update\n(Unfiltered)',
#                    fontsize=LABEL_FONTSIZE*SCALE,
#                    rotation=90, ha='center', va='center',
#                    transform=fig10.axes[0].transAxes)
# fig10.savefig(join(plots, 'fig10_est1_comparison.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig10_est1_comparison.png')
# fig11.savefig(join(plots, 'fig11_est1_spectrum.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig11_est1_spectrum.png')
# fig12.savefig(join(plots, 'fig12_est1_eigenvectors.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig12_est1_eigenvectors.png')

#####################################
### FIGURE 13 - 15: EST2 ANALYSIS ###
#####################################
# fig13, fig14, fig15 = est2.full_analysis(true, clusters_plot)
# # fig14.suptitle('Final Update', fontsize=TITLE_FONTSIZE*SCALE,
# #                y=1.5)
# # fig13.axes[0].text(-0.65, 0.5, 'Second Update\n(Unfiltered)',
# #                    fontsize=LABEL_FONTSIZE*SCALE,
# #                    rotation=90, ha='center', va='center',
# #                    transform=fig13.axes[0].transAxes)
# fig13.savefig(join(plots, 'fig13_est2_comparison.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig13_est2_comparison.png')
# fig14.savefig(join(plots, 'fig14_est2_spectrum.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig14_est2_spectrum.png')
# fig15.savefig(join(plots, 'fig15_est2_eigenvectors.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig15_est2_eigenvectors.png')

#######################################
### FIGURE 16 - 18: EST2_F ANALYSIS ###
#######################################
# fig16, fig17, fig18 = est2_f.full_analysis(true_f, clusters_plot)
# # fig17.suptitle('Final Update', fontsize=TITLE_FONTSIZE*SCALE,
# #                y=1.5)
# # fig16.axes[0].text(-0.65, 0.5, 'Second Update\n(Filtered)',
# #                    fontsize=LABEL_FONTSIZE*SCALE,
# #                    rotation=90, ha='center', va='center',
# #                    transform=fig16.axes[0].transAxes)
# fig16.savefig(join(plots, 'fig16_est2_f_comparison.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig16_est2_f_comparison.png')
# fig17.savefig(join(plots, 'fig17_est2_f_spectrum.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig17_est2_f_spectrum.png')
# fig18.savefig(join(plots, 'fig18_est2_f_eigenvectors.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig18_est2_f_eigenvectors.png')

############################################
### FIGURE 19 -20: K0 AND KTRUE ANALYSIS ###
############################################
# n = 1100
# fig19, ax = fp.make_axes(maps=True, lats=obs['lat'], lons=obs['lon'])

# col = ax.scatter(obs['lon'], obs['lat'], c=est0.k[:, n],
#                  cmap=fp.cmap_trans('Reds'), vmin=0, vmax=0.2,
#                  s=100)

# ax = fp.add_title(ax, r'Initial Estimate: $\frac{\mathrm{d}\mathbf{y}}{\mathrm{d}x_{%d}}$' % n)
# ax = fp.format_map(ax, obs['lat'], obs['lon'])

# cax = fp.add_cax(fig19, ax)
# cbar = plt.colorbar(col, cax=cax)
# cbar = fp.format_cbar(cbar, r'$\frac{\mathrm{d}y}{\mathrm{d}x_{%d}}$ (ppb)' % n)
# cbar.set_ticks(np.arange(0, 0.21, 0.1))

# fig19.savefig(join(plots, 'fig19_kinit_col.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig18_est2_f_eigenvectors.png')


# fig20, ax = fp.make_axes(maps=True, lats=obs['lat'], lons=obs['lon'])

# col = ax.scatter(obs['lon'], obs['lat'], c=true.k[:, n],
#                  cmap=fp.cmap_trans('Reds'), vmin=0, vmax=0.2,
#                  s=100)

# ax = fp.add_title(ax, r'True: $\frac{\mathrm{d}\mathbf{y}}{\mathrm{d}x_{%d}}$' % n)
# ax = fp.format_map(ax, obs['lat'], obs['lon'])

# cax = fp.add_cax(fig20, ax)
# cbar = plt.colorbar(col, cax=cax)
# cbar = fp.format_cbar(cbar, r'$\frac{\mathrm{d}y}{\mathrm{d}x_{%d}}$ (ppb)' % n)
# cbar.set_ticks(np.arange(0, 0.21, 0.1))

# fig20.savefig(join(plots, 'fig20_kinit_col.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig18_est2_f_eigenvectors.png')

#####################################
## FIGURE 21: EST2 POSTERIOR MEAN ###
#####################################
# cbar_kwargs = {'ticks' : np.arange(-1, 4, 1),
#                'title' : 'Scaling Factors'}
# fig21, ax, c = est2.plot_state('xhat',
#                                clusters_plot,
#                                default_value=1,
#                                **{'title' : 'Estimated Posterior Mean',
#                                   'cmap' : 'RdBu_r',
#                                   'vmin' : -1,
#                                   'vmax' : 3,
#                                   'cbar_kwargs' : cbar_kwargs})

# fig21.savefig(join(plots, 'fig21_est2_posterior_mean.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig21_true_posterior_mean.png')

#######################################
## FIGURE 22: EST2 AVERAGING KERNEL ###
#######################################
# cbar_kwargs = {'title' : r'$\partial\hat{x}/\partial x$'}
# fig22, ax, c = est2.plot_state('dofs', clusters_plot,
#                                **{'title' : 'Estimated Averaging Kernel',
#                                   'cmap' : plasma_trans,
#                                   'vmin' : 0,
#                                   'vmax' : 1,
#                                   'cbar_kwargs' : cbar_kwargs})
# ax.text(0.025, 0.05, 'DOFS = %.2f' % np.trace(true.a),
#         fontsize=LABEL_FONTSIZE*SCALE,
#         transform=ax.transAxes)

# fig22.savefig(join(plots, 'fig22_est2_averaging_kernel.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig22_est2_averaging_kernel.png')

##################################
## FIGURE 22: DETAILED SPECTRA ###
##################################
# fig23, ax = true.plot_info_frac(label='True',
#                                 color=fp.color(0),
#                                 text=False)
# fig23, ax = est0.plot_info_frac(figax=[fig23, ax],
#                                 label='Initial Estimate',
#                                 ls=':',
#                                 color=fp.color(6),
#                                 text=False)

# frac = np.cumsum(est0.evals_q/est0.evals_q.sum())
# snr_idx = np.argwhere(est0.evals_q >= 0.555555)[-1][0]
# ax.scatter(snr_idx, frac[snr_idx], s=30*SCALE, c=fp.color(6))
# ax.text(snr_idx + est0.nstate*0.05, frac[snr_idx] - 0.05,
#         r'SNR = 1.25',
#         ha='left', va='top', fontsize=LABEL_FONTSIZE*SCALE,
#         color=fp.color(6))
# ax.text(snr_idx + est0.nstate*0.05, frac[snr_idx] - 0.125,
#         'n = %d' % snr_idx,
#         ha='left', va='top', fontsize=LABEL_FONTSIZE*SCALE,
#         color=fp.color(6))
# ax.text(snr_idx + est0.nstate*0.05, frac[snr_idx] - 0.2,
#         r'$f_{DOFS}$ = %.2f' % frac[snr_idx],
#         ha='left', va='top', fontsize=LABEL_FONTSIZE*SCALE,
#         color=fp.color(6))

# fig23.savefig(join(plots, 'fig23_est0_spectra.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig23_est0_spectra.png')

# fig23, ax = est1.plot_info_frac(figax=[fig23, ax],
#                                 label='Updated Estimate',
#                                 ls=':',
#                                 color=fp.color(4),
#                                 text = False)
# fig23.savefig(join(plots, 'fig24_est0_spectra.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig24_est0_spectra.png')

# fig24, ax = true.plot_info_frac(label='True',
#                                 color=fp.color(0),
#                                 text=False)
# fig24, ax = est0.plot_info_frac(figax=[fig24, ax],
#                                 label='Initial Estimate',
#                                 ls=':',
#                                 color=fp.color(6),
#                                 text = False)
# fig24, ax = est1.plot_info_frac(figax=[fig24, ax],
#                                 label='Updated Estimate',
#                                 ls=':',
#                                 color=fp.color(4),
#                                 text = False)

# frac = np.cumsum(est0.evals_q/est0.evals_q.sum())
# info_idx = np.argwhere(frac >= 0.975)[0][0]
# ax.scatter(info_idx, frac[info_idx], s=30*SCALE, c=fp.color(6))
# ax.text(info_idx + est0.nstate*0.05, frac[info_idx] - 0.05,
#         r'\% DOFS = 97.5\%',
#         ha='left', va='top', fontsize=LABEL_FONTSIZE*SCALE,
#         color=fp.color(6))
# ax.text(info_idx + est0.nstate*0.05, frac[info_idx] - 0.125,
#         'n = %d' % info_idx,
#         ha='left', va='top', fontsize=LABEL_FONTSIZE*SCALE,
#         color=fp.color(6))
# # ax.text(info_idx + est0.nstate*0.05, frac[info_idx] - 0.15,
# #         r'$f_{DOFS}$ = %.2f' % frac[info_idx],
# #         ha='left', va='top', fontsize=LABEL_FONTSIZE*SCALE,
# #         color=fp.color(6))

# fig24.savefig(join(plots, 'fig25_est1_spectra.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig24_est1_spectra.png')

# fig24, ax = est2.plot_info_frac(figax=[fig24, ax],
#                                 label='Final Estimate',
#                                 ls=':',
#                                 color=fp.color(2),
#                                 text = False)

# fig24.savefig(join(plots, 'fig26_est1_spectra.png'),
#               bbox_inches='tight', dpi=300)
# print('Saved fig24_est1_spectra.png')


# # fig25, ax = true.plot_info_frac(label='True',
# #                                 color=fp.color(0),
# #                                 text=False)
# # fig25, ax = est0.plot_info_frac(figax=[fig25, ax],
# #                                 label='Initial Estimate',
# #                                 ls=':',
# #                                 color=fp.color(6),
# #                                 text = False)
# # fig25, ax = est1.plot_info_frac(figax=[fig25, ax],
# #                                 label='Updated Estimate',
# #                                 ls=':',
# #                                 color=fp.color(4),
# #                                 text = False)
# # fig25, ax = est2.plot_info_frac(figax=[fig25, ax],
# #                                 label='Final Estimate',
# #                                 ls=':',
# #                                 color=fp.color(2),
# #                                 text = False)

# # fig25.savefig(join(plots, 'fig25_est1_spectra.png'),
# #               bbox_inches='tight', dpi=300)
# # print('Saved fig25_est1_spectra.png')
