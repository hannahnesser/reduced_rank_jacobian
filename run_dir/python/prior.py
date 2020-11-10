import xarray as xr
import numpy as np


from os.path import join
from os import listdir
import sys

input_dir = str(sys.argv[1])
emissions = str(sys.argv[2])
prior_error = str(sys.argv[3])
resolution = str(sys.argv[4])

def get_prior(clusters_long, emis_loc=None, relative=None):
    if (emis_loc is not None) & (~relative):
        xa = get_emis(emis_loc)

        # Join in the long clusters file
        xa = xa.to_dataset(name='emis')
        xa['NSV'] = clusters_long

        # Add together emissions in state vector elements
        xa = xa.groupby('NSV').sum(xr.ALL_DIMS)
        xa = xa.where(xa.NSV > 0, drop=True)

        # Change to dataarray
        xa = xa['emis']

    elif (emis_loc is None) & relative:
        SV_elems = np.unique(clusters_long)[1:]
        xa = np.ones(len(SV_elems))
        xa = xr.DataArray(xa, dims=('NSV'), coords={'NSV' : SV_elems})

    else:
        print('Improper inputs.')
        sys.exit()

    return xa

def get_error_emis(error_emis_loc, emis_loc, clusters_long):
    # Load absolute emissions
    xa_abs = get_emis(emis_loc)
    xa_abs = xa_abs.to_dataset(name='emis')

    sa = xr.open_dataset(error_emis_loc)['data']

    # Convert to absolute
    sa_abs = sa*xa_abs
    sa_abs['NSV'] = clusters_long

    # Group by NSV and add in quadrature
    sa_vec = sa_abs.groupby('NSV').apply(add_quadrature)
    sa_vec = sa_vec.where(sa_vec.coords['NSV'] != 0, drop=True)
    sa_vec = sa_vec['emis']

    # Make relative: import absolute emisssions
    xa_abs['NSV'] = clusters_long
    xa_abs = xa_abs.groupby('NSV').sum(xr.ALL_DIMS)
    xa_abs = xa_abs.where(xa_abs.NSV > 0, drop=True)
    xa_abs = xa_abs['emis']
    # sa_vec /= xa_abs

    # Make relative
    sa_vec /= xa_abs

    # And set a threshold
    sa_vec.values[sa_vec >= 0.5] = 0.5

    # Sigh just set them all to 50% because we can't actually
    # add relative errors in quadrature, silly.
    # sa_vec.values = 0.5*np.ones(len(sa_vec))

    sa_vec = sa_vec**2 # get variances from errors

    return sa_vec

def add_quadrature(x):
    return np.sqrt((x**2).sum())

if not set(['sa_vec.nc', 'xa.nc']).issubset(listdir(input_dir)):
    print('Retrieving prior error and prior.')
    clusters = xr.open_dataarray(join(input_dir, 'clusters_' + resolution + '.nc'))

    # prior - absolute
    xa_abs = get_prior(clusters, emis_loc=join(input_dir, emissions),
                       relative=False)
    xa_abs.to_netcdf(join(input_dir, 'xa_abs.nc'))

    # prior - relative
    xa = get_prior(clusters, relative=True)
    xa.to_netcdf(join(input_dir, 'xa.nc'))

    # prior error
    sa_vec = get_error_emis(join(input_dir, prior_error),
                            join(input_dir, emissions),
                            clusters)
    sa_vec.to_netcdf(join(input_dir, 'sa_vec.nc'))

else:
    print('Prior error and prior already exist.')
