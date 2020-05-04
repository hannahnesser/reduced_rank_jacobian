import copy
import os
from os.path import join
import sys
import math

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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

c = plt.cm.get_cmap('inferno', lut=10)

def color(k, cmap='inferno', lut=10):
    c = plt.cm.get_cmap(cmap, lut=lut)
    return colors.to_hex(c(k))

def cmap_trans(cmap, ncolors=300, nalpha=20):
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.append(np.linspace(0.0, 1.0, nalpha),
                                  np.ones(ncolors-nalpha))

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='plasma_trans',
                                                   colors=color_array)

    return map_object

plasma_trans = cmap_trans('plasma')
plasma_trans_r = cmap_trans('plasma_r')

rcParams['text.usetex'] = True

SCALE = 4
BASEFONT = 10
TITLE_LOC = 1.11
CBAR_PAD = 0.05
LABEL_PAD = 5
CBAR_LABEL_PAD = 75

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

###################################
### FIGURE : GOSAT OBSERVATIONS ###
###################################
# fig01, ax = plt.subplots(figsize=(8*SCALE/1.25,6*SCALE/1.25),
#                          subplot_kw={'projection' :
#                                      ccrs.PlateCarree()})

# col = ax.scatter(obs['lon'], obs['lat'], c=obs['GOSAT'],
#                cmap='inferno', vmin=1700, vmax=1800,
#                s=100)

# ax.set_title(r'GOSAT XCH$_4$ (July 2009)', y=TITLE_LOC,
#              fontsize=(BASEFONT+10)*SCALE)
# ax.add_feature(cartopy.feature.OCEAN, facecolor='0.98')
# ax.add_feature(cartopy.feature.LAND, facecolor='0.98')
# ax.coastlines(color='grey')

# gl = ax.gridlines(linestyle=':', draw_labels=True, color='grey')
# gl.xlabel_style = {'fontsize' : (BASEFONT-5)*SCALE}
# gl.ylabel_style = {'fontsize' : (BASEFONT-5)*SCALE}

# cax = fig01.add_axes([ax.get_position().x1 + 0.05,
#                       ax.get_position().y0,
#                       0.005*SCALE,
#                       ax.get_position().height])
# cbar = plt.colorbar(col, cax=cax)
# cbar.set_label(label='XCH4', fontsize=(BASEFONT+5)*SCALE,
#                labelpad=CBAR_LABEL_PAD)
# cbar.ax.tick_params(labelsize=BASEFONT*SCALE)

# # Save plot
# fig01.savefig(join(plots, 'fig01_gosat_obs.png'),
#              bbox_inches='tight')
# print('Saved fig01_gosat_obs.png')

##########################################
### FIGURE : GOSAT OBSERVATION DENSITY ###
##########################################
# lat_res = np.diff(clusters_plot.lat)[0]
# lat_edges = np.append(clusters_plot.lat - lat_res/2,
#                       clusters_plot.lat[-1] + lat_res/2)
# lat_edges = lat_edges[::2]
# obs['lat_edges'] = pd.cut(obs['lat'], lat_edges, precision=4)

# lon_res = np.diff(clusters_plot.lon)[0]
# lon_edges = np.append(clusters_plot.lon - lon_res/2,
#                       clusters_plot.lon[-1] + lon_res/2)
# lon_edges = lon_edges[::2]
# obs['lon_edges'] = pd.cut(obs['lon'], lon_edges, precision=4)

# obs_density = obs.groupby(['lat_edges', 'lon_edges']).count()
# obs_density = obs_density['Nobs'].reset_index()
# obs_density['lat'] = obs_density['lat_edges'].apply(lambda x: x.mid)
# obs_density['lon'] = obs_density['lon_edges'].apply(lambda x: x.mid)
# obs_density = obs_density.set_index(['lat', 'lon'])['Nobs']
# obs_density = obs_density.to_xarray()

# viridis_trans_long = cmap_trans('viridis', nalpha=50, ncolors=300)
# fig02, ax, c = true.plot_state_format(obs_density, default_value=0,
#                                       **{'vmin' : 0,
#                                          'vmax' : 50,
#                                          'cmap' : viridis_trans_long,
#                                          'title' : 'GOSAT Observation Density'})
# c.set_label(label='Number of Observations', fontsize=(BASEFONT+5)*SCALE,
#             labelpad=CBAR_LABEL_PAD)

# fig02.savefig(join(plots, 'fig02_gosat_obs_density.png'),
#               bbox_inches='tight')
# print('Saved fig02_gosat_obs_density.png')

################################
### FIGURE : PRIOR EMISSIONS ###
################################
# fig03, ax, c = true.plot_state('xa_abs', clusters_plot,
#                                **{'title' : 'Prior Emissions',
#                                   'cmap' : cmap_trans('viridis')})
# c.set_label(label='CH4 Emissions (Gg/month)', fontsize=(BASEFONT+5)*SCALE,
#             labelpad=CBAR_LABEL_PAD)

# fig03.savefig(join(plots, 'fig03_prior_emissions.png'),
#               bbox_inches='tight')
# print('Saved fig03_prior_emissions.png')
