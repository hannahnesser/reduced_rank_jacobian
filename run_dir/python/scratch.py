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
                            snr=2.5)

# Second set
# est2 = est1.update_jacobian(true.k,
#                             rank=est0.get_rank(pct_of_info=0.98))
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
# mask = ~np.isclose(est2.xhat, np.ones(est2.nstate), atol=1e-2)
est2_f, true_f = est2.filter(true, mask)

print('-----------------------')
print('MODEL RUNS: ', est2.model_runs)
print('CONSTRAINED CELLS: ', len(est2_f.xhat))
print('DOFS (filtered): ', np.trace(est2_f.a))
print('DOFS (total): ', np.trace(est2.a))
print('Native Resolution DOFS (filtered):', true.dofs[mask].sum())
print('DOFS (SNR = 1): ', np.trace(est2a.a))
print('DOFS (SNR = 4): ', np.trace(est2b.a))
frac = np.cumsum(est0.evals_q/est0.evals_q.sum())
rank = int(est2.model_runs - est1.model_runs)
ranka = int(est2a.model_runs - est1a.model_runs)
rankb = int(est2b.model_runs - est1b.model_runs)
print('% DOFS update 2 (standard): ', 100*frac[rank])
print('% DOFS update 2 (SNR = 1): ', 100*frac[ranka])
print('% DOFS update 2 (SNR = 4): ', 100*frac[rankb])
print('-----------------------\n')


# open summary files
r2_summ = np.load(join(inputs, 'r2_summary_R3.npy'))
nc_summ = np.load(join(inputs, 'nc_summary_R3.npy'))
nm_summ = np.load(join(inputs, 'nm_summary_R3.npy'))
dofs_summ = np.load(join(inputs, 'dofs_summary_R3.npy'))

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
# fig07, ax = fp.get_figax()
figsize = fp.get_figsize(aspect=0.675, rows=1, cols=1)
fig07, ax = plt.subplots(2, 1,
                         figsize=figsize,
                         gridspec_kw={'height_ratios': [1, 4]})
plt.subplots_adjust(hspace=0.4)
cax = fp.add_cax(fig07, ax[1])
# ax = fp.add_title(ax, r'Posterior Emissions r$^2$'))
ax[0] = fp.add_title(ax[0], 'Reduced Rank DOFS', y=1.6)
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

print(np.unique(nm_summ_short))
print(z)

# ax[1].plot(x, y, c='0.25')
x = np.unique(nm_summ_short)
z = np.array(z)
ax[0].plot(x[x <= 1000], z[x <= 1000], c='0.25', ls='--')
ax[0].scatter(est2.model_runs, est2.dofs.sum(),
              zorder=10, c='0.25', s=200, marker='*')
# ax[0].scatter(est2a.model_runs, est2a.dofs.sum(),
#               zorder=10, c='0.25', s=100, marker='_')
# ax[0].scatter(est2b.model_runs, est2b.dofs.sum(),
#               zorder=10, c='0.25', s=100, marker='_')
ax[0].set_xticks(np.arange(200, 1010, 200))
ax[0].set_xlim(0, 1000)
ax[0].set_ylim(50, 216)
ax[0].set_facecolor('0.98')
ax[0] = fp.add_labels(ax[0],
                       xlabel='Total Model Runs',
                       ylabel='Optimal\nDOFS',
                       labelpad=config.LABEL_PAD/2)

# ax.plot(y, x, c='green', lw=5)


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
              s=200, marker='*')
ax[1].scatter(mr2n(est1a.model_runs),
              mr2n(est2a.model_runs-est1a.model_runs),
              zorder=10, c='0.25', s=100, marker='.')
ax[1].scatter(mr2n(est1b.model_runs),
              mr2n(est2b.model_runs-est1b.model_runs),
              zorder=10, c='0.25', s=100, marker='.')

# np.set_printoptions(precision=0)
# print(dofs_summ[:20, :20])

# cbar = fig07.colorbar(cf, cax=cax, ticks=np.linspace(0, 1, 6))
cbar = fig07.colorbar(cf, cax=cax,
                    ticks=np.arange(0, true.dofs.sum(), 50))
# cbar = fig07.colorbar(cf, cax=cax, ticks=np.arange(0, 2000, 500))

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

fp.save_fig(fig07, plots, 'fig07_dofs_comparison')

############################################################
### FIGURE 06: EST2_F POSTERIOR COMPARSION SCATTER PLOTS ###
############################################################
# fig06, _, _ = est2_f.full_analysis(true_f, clusters_plot)

# ax = fig06.axes
# ax[0].set_xticks([0, 5])
# ax[0].set_yticks([0, 5])

# ax[1].set_xticks([-10, 0, 10])
# ax[1].set_yticks([-10, 0, 10])

# ax[2].set_xticks([0.2, 0.4])
# ax[2].set_yticks([0.2, 0.4])

# ax[3].set_xticks([0, 0.5, 1])
# ax[3].set_yticks([0, 0.5, 1])

# fp.save_fig(fig06, plots, 'fig06_posterior_scattter_comparison')


# Stacking attempts
# import matplotlib.transforms as mt
# def plot_skew(ax, im, transform):
#     im = ax.imshow(im)
#     trans_data = ax.transData + transform
#     im.set_transform(trans_data)

# # fig, ax = fp.get_figax(maps=True, lats=clusters_plot.lat,
# #                        lons=clusters_plot.lon)
# fig, ax = plt.subplots()
# for i in range(2, -1, -1):
#     im = mpimg.imread(join(plots, 'fig1c_evec%d.png' % i))
#     y, x = im.shape[0]*0.5, im.shape[1]*0.5
#     transform = mt.Affine2D().skew_deg(-60, 0).rotate_deg(15).translate(0, -250*(i+1))
#     plot_skew(ax, im, transform)

# ax.set_xlim(-1000, 0)
# # ax.set_ylim(1000, -1000)
# fp.save_fig(fig, loc=plots, name='test')

# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# # Load initial data
# data = true.match_data_to_clusters(true.evecs[:, 0], clusters_plot)

# # Get map information
# proj_ax = plt.figure().add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
# pcm = data.plot(ax=proj_ax, cmap=rdbu_trans, vmin=-0.1, vmax=0.1)
# concat = lambda iterable: list(itertools.chain.from_iterable(iterable))
# target_projection = proj_ax.projection
# feature = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m')
# geoms = feature.geometries()
# boundary = proj_ax._get_extent_geom()
# geoms = [target_projection.project_geometry(geom, feature.crs)
#          for geom in geoms]
# # geoms = [boundary.intersection(geom) for geom in geoms]
# paths = concat(geos_to_path(geom) for geom in geoms)
# polys = concat(path.to_polygons() for path in paths)
# lc = PolyCollection(polys, edgecolor='grey', linewidth=0.5,
#                     facecolor='0.98', closed=True)

# # plot figure
# fig = plt.figure()
# ax3d = fig.add_axes([0, 0, 1, 1], projection='3d')
# # data = true.match_data_to_clusters(true.evecs[:, 0], clusters_plot)
# xx, yy = np.meshgrid(data.lon, data.lat)
# norm = colors.Normalize(vmin=-0.1, vmax=0.1)
# ax3d.add_collection3d(lc, zs=-0.1)
# ax3d.plot_surface(xx, yy, np.atleast_2d(0),
#                   cstride=1, rstride=1, zorder=10,
#                   facecolors=rdbu_trans(norm(data.values)))


# ax3d.set_zlim(-5, 5)


# # for collection in pcm.collections:
# #     paths = pcm.get_paths()
# #     # Figure out the matplotlib transform to take us from the X, Y coordinates
# #     # to the projection coordinates.
# #     trans_to_proj = collection.get_transform() - proj_ax.transData
# #     paths = [trans_to_proj.transform_path(path) for path in paths]
# #     codes = [path.codes for path in paths]
# #     pc = Poly3DCollection([])
# #     pc.set_codes(codes)

# #     # Copy all of the parameters from the contour (like colors) manually.
# #     # Ideally we would use update_from, but that also copies things like
# #     # the transform, and messes up the 3d plot.
# #     pc.set_facecolor(collection.get_facecolor())
# #     pc.set_edgecolor(collection.get_edgecolor())
# #     pc.set_alpha(collection.get_alpha())

# #     ax3d.add_collection3d(pc)

# # proj_ax.autoscale_view()
# # ax3d.set_xlim(*proj_ax.get_xlim())
# # ax3d.set_ylim(*proj_ax.get_ylim())

# # fig.set_facecolor('white')
# # ax_3d.set_facecolor('white')
# # ax_3d.grid(False)
# # ax_3d.set_axis_off()
# # for i in range(3):

# #     ax_3d.pcolormesh(data.lon, data.lat, data.values,
# #                      cmap='RdBu_r', vmin=-0.1, vmax=0.1)
# #     # img = mpimg.imread(join(plots, 'fig1c_evec%d.png' % i))
# #     # x = np.arange(0, img.shape[0])
# #     # y = np.arange(0, img.shape[1])
# #     # xx, yy = np.meshgrid(x, y)
# #     # ax_3d.pcolormesh(xx, yy, np.atleast_2d(2*i),
# #     #                    rstride=5, cstride=5,
# #     #                    facecolors=np.einsum('ijk->jik', img),
# #     #                    vmin=-0.1, vmax=0.1)

# fp.save_fig(fig, loc=plots, name='test')
