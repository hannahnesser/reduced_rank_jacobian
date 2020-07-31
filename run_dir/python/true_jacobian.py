import pandas as pd 
import numpy as np
import xarray as xr 
import sys
from os.path import join
from os import listdir

input_dir = str(sys.argv[1])
jac_str = str(sys.argv[2])
resolution = str(sys.argv[3])
model_response_data = str(sys.argv[4])
glint_data = str(sys.argv[5])

if jac_str + '_true.nc' not in listdir(input_dir):
    print('True Jacobian does not yet exist.')

    # Read in the clusters
    clusters = xr.open_dataarray(join(input_dir, 'clusters_' + resolution + '.nc'))

    # Read in the model response and do some formatting. This is specific
    # to the format used and will need to be modified
    delta_obs_true = pd.read_csv(join(input_dir, model_response_data))
    delta_obs_true = delta_obs_true.set_index('NNN')
    delta_obs_true = delta_obs_true.drop(columns='model_base')
    for col in delta_obs_true:
        col_name = col.split('_')[-1]
        delta_obs_true = delta_obs_true.rename(columns={col : col_name})
        
    delta_obs_true = delta_obs_true.unstack().reset_index()
    delta_obs_true = delta_obs_true.rename(columns={'level_0' : 'NSV',
                                                    'NNN' : 'Nobs',
                                                    0 : 'delta_obs'})
    delta_obs_true['NSV'] = delta_obs_true['NSV'].astype(int)
    delta_obs_true = delta_obs_true.set_index(['Nobs', 'NSV'])

    delta_obs_true = delta_obs_true.to_xarray()

    # Define K
    k_true = delta_obs_true/0.5

    # Eliminate superfluous NSV elements
    NSV_all = np.unique(clusters)
    k_true = k_true.where(k_true.coords['NSV'].isin(NSV_all[1:]),
                          drop=True)
    k_true = k_true['delta_obs']

    # Eliminate glint
    if len(glint_data) > 1:
        base_full = pd.read_csv(join(input_dir, glint_data), 
                               delim_whitespace=True,
                               header=0)
        y_glint = base_full['NNN'][base_full['GLINT']==True]

        k_true = k_true.where(~k_true.coords['Nobs'].isin(y_glint), drop=True)
        delta_obs_true = delta_obs_true.where(~delta_obs_true.coords['Nobs'].isin(y_glint), drop=True)

    # save out k_true
    k_true.to_netcdf(join(input_dir, jac_str + '_true.nc'))

else:
    print('True Jacobian already exists.')