import xarray as xr
import numpy as np


from os.path import join
from os import listdir
import sys
sys.path.append('.')
import jacobian as j

input_dir = str(sys.argv[1])
emissions = str(sys.argv[2])
prior_error = str(sys.argv[3])
resolution = str(sys.argv[4])

if not set(['sa_vec.nc', 'xa.nc']).issubset(listdir(input_dir)):
    print('Retrieving prior error and prior.')
    clusters = xr.open_dataarray(join(input_dir, 'clusters_' + resolution + '.nc'))

    # prior - absolute
    xa_abs = j.get_prior(clusters, emis_loc=join(input_dir, emissions),
                             relative=False)
    xa_abs.to_netcdf(join(input_dir, 'xa_abs.nc'))

    # prior - relative
    xa = j.get_prior(clusters, relative=True)
    xa.to_netcdf(join(input_dir, 'xa.nc'))

    # prior error
    sa_vec = j.get_error_emis(join(input_dir, prior_error),
                                  join(input_dir, emissions),
                                  clusters)
    sa_vec.to_netcdf(join(input_dir, 'sa_vec.nc'))

else:
    print('Prior error and prior already exist.')
