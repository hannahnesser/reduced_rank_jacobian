import xarray as xr
import numpy as np
# from numpy import diag as diags
# from numpy import identity
from numpy.linalg import inv, norm, eigh
from scipy.sparse import diags, identity
from scipy.stats import linregress
# from scipy.linalg import eigh
import pandas as pd
from tqdm import tqdm
import copy

# clustering
from sklearn.cluster import KMeans

import math

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams, colorbar, colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy

SCALE = 4
BASEFONT = 10
TITLE_LOC = 1.11
CBAR_PAD = 0.05
LABEL_PAD = 5
CBAR_LABEL_PAD = 75
rcParams['font.family'] = 'serif'
rcParams['font.size'] = BASEFONT*SCALE
rcParams['text.usetex'] = True

def color(k, cmap='inferno', lut=10):
    c = plt.cm.get_cmap(cmap, lut=lut)
    return colors.to_hex(c(k))

def cmap_trans(cmap, ncolors=300, nalpha=20):
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.append(np.linspace(0.0, 1.0, nalpha),
                                  np.ones(ncolors-nalpha))

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='plasma_trans',colors=color_array)

    return map_object

def get_figsize(aspect, rows, cols):
    if aspect > 1:
        return (6*SCALE, 6*SCALE/aspect)
    else:
        return (6*SCALE*aspect, 6*SCALE)

def get_aspect(rows, cols, aspect=None, maps=False, lats=None, lons=None):
    if maps:
        aspect = np.cos(np.mean([lats.min(), lats.max()])*np.pi/180)
        xsize = np.ptp([lons.max(), lons.min()])*aspect
        ysize = np.ptp([lats.max(), lats.min()])
        aspect = xsize/ysize
    aspect *= cols/rows
    return aspect

def make_axes(rows=1, cols=1, aspect=None, maps=False, lats=None, lons=None):
    aspect = get_aspect(rows, cols, aspect, maps, lats, lons)
    if maps:
        figsize = get_figsize(aspect, rows, cols)
        fig, ax = plt.subplots(rows, cols,
                               figsize=figsize,
                               subplot_kw={'projection' : ccrs.PlateCarree()})
    else:
        figsize = get_figsize(aspect, rows, cols)
        fig, ax = plt.subplots(rows, cols,
                               figsize=figsize)
    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    return fig, ax

def add_cax(fig, ax):
    try:
        cax = fig.add_axes([ax.get_position().x1 + 0.05,
                            ax.get_position().y0,
                            0.005*SCALE,
                            ax.get_position().height])
    except AttributeError:
        cax = fig.add_axes([ax[-1, -1].get_position().x1 + 0.05,
                            ax[-1, -1].get_position().y0,
                            0.005*SCALE,
                            ax[0, -1].get_position().y1 \
                            - ax[-1, -1].get_position().y0])

    return cax

def get_figax(rows=1, cols=1, aspect=1,
              maps=False, lats=None, lons=None, kw={}):
    n = len(kw)
    if 'figax' in kw.keys():
        fig, ax = kw.pop('figax')
    else:
        fig, ax = make_axes(rows, cols, aspect, maps, lats, lons)

    if (rows > 1) or (cols > 1):
        for axis in ax.flatten():
            axis.set_facecolor('0.98')
    else:
        ax.set_facecolor('0.98')

    if n > 0:
        return fig, ax, kw
    else:
        return fig, ax

def add_labels(ax, title, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=(BASEFONT+6)*SCALE,
                  labelpad=LABEL_PAD)
    ax.set_ylabel(ylabel, fontsize=(BASEFONT+6)*SCALE,
                  labelpad=LABEL_PAD)
    ax.set_title(title, fontsize=(BASEFONT+10)*SCALE, y=TITLE_LOC-0.06)
    ax.tick_params(axis='both', which='both', labelsize=BASEFONT*SCALE)
    return ax

def add_legend(ax):
    ax.legend(frameon=False, fontsize=(BASEFONT+5)*SCALE)
    return ax

def get_square_limits(xdata, ydata):
    # Get data limits
    dmin = min(np.min(xdata), np.min(ydata))
    dmax = max(np.max(xdata), np.max(ydata))
    pad = (dmax - dmin)*0.05
    dmin -= pad
    dmax += pad

    try:
        # get lims
        ylim = kw.pop('ylim')
        xlim = kw.pop('xlim')
        xy = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    except:
        # set lims
        xlim = ylim = xy = (dmin, dmax)

    return xlim, ylim, xy, dmin, dmax

def format_map(ax):
    ax.add_feature(cartopy.feature.OCEAN, facecolor='0.98')
    ax.add_feature(cartopy.feature.LAND, facecolor='0.98')
    ax.coastlines(color='grey')
    gl = ax.gridlines(linestyle=':', draw_labels=True, color='grey')
    gl.xlabel_style = {'fontsize' : (BASEFONT-5)*SCALE}
    gl.ylabel_style = {'fontsize' : (BASEFONT-5)*SCALE}
    return ax

def format_cbar(cbar, **kw):
    cbar_title = kw.pop('cbar_title', '')
    cbar.set_label(cbar_title, fontsize=(BASEFONT+5)*SCALE)
    cbar.ax.tick_params(axis='both', which='both',
                        labelsize=(BASEFONT+5)*SCALE)
    return cbar

def plot_one_to_one(ax):
    xlim, ylim, _, _, _ = get_square_limits(ax.get_xlim(),
                                            ax.get_ylim())
    ax.plot(xlim, xlim, c='0.1', lw=2, ls=':',
            alpha=0.5, zorder=0)
    return ax
