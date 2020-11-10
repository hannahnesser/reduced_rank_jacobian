import yaml
import xarray as xr
import pandas as pd
import numpy as np
from os import remove, listdir
from os.path import isfile, isdir, join
import tqdm

import sys
sys.path.append('/Users/hannahnesser/Documents/Harvard/Research/Python')
import clusters
import inv_rr

def build_jacobian(obs, emis=None, clusters_long=None, clusters_mapping=None):
    # get obs
    if clusters_mapping is not None:
        delta_obs = get_delta_obs(obs, clusters_mapping=clusters_mapping)
    else:
        delta_obs = get_delta_obs(obs)

    # Get emis
    if emis is not None:
        if type(emis) == str:
            print('Calculating absolute emissions from filepath.')
            delta_emis = get_delta_emis(clusters_long, emis, relative=False)
        else:
            print('Using given emissions.')
            delta_emis = emis
    else:
        print('Calculating relative emissions (0.5).')
        delta_emis = get_delta_emis(clusters_long, relative=True)

    # Calculate the Jacobian
    k = delta_obs/delta_emis

    return k

#### CHECK DELTA_EMIS AND DELTA_OBS FOR GROUPING ISSUES


def get_delta_obs(obs, clusters_mapping=None):
    if type(obs) == str:
        delta_obs = load_delta_obs(obs)
    else:
        delta_obs = obs

    # Once we have delta_obs, we want to eliminate those observations that
    # aren't effected by perturbations
    # (We won't worry about altering the state vector dimension--this
    # occurs in the reduce_SV_elements routine.)
    delta_obs = delta_obs.where(delta_obs.sum(dim='NSV') > 0, drop=True)

    # Keep in mind that this is currently delta_obs at the highest resolution,
    # corresponding with '1'

    if clusters_mapping is not None:
        # n_clust = clusters_map.shape[0] #Account for 0s
        # grouping_factor = len(delta_obs.coords['NSV'])/n_clust
        # if grouping_factor == int(grouping_factor):
        #     if grouping_factor > 1: # if

        delta_obs_df = delta_obs.to_dataframe(name='delta_obs').reset_index()
        # since this is at the highest resolution
        delta_obs_df = delta_obs_df.rename(columns={'NSV' : '1'})

        # Join in the larger clusters
        delta_obs_df = delta_obs_df.merge(clusters_mapping)

        # Sum
        delta_obs_df = delta_obs_df.groupby(['Nobs', '2']).sum()['delta_obs'].reset_index()

        # Create an xarray
        delta_obs_df = delta_obs_df.rename(columns={'2' : 'NSV'})
        delta_obs = delta_obs_df.set_index(['Nobs', 'NSV']).to_xarray()
        delta_obs = delta_obs['delta_obs']

                # SV_groups = np.repeat(np.unique(clusters_long)[1:], grouping_factor)
                # delta_obs['NSV'] = SV_groups
                # delta_obs = delta_obs.groupby('NSV').sum('NSV')
        # else:
        #     print('Desired state vector size does not divide evenly into true state vector size.')
        #     print('delta_obs NSV dimension is %d' % (len(delta_obs.coords['NSV'])))
        #     print('Number of clusters is %d' % n_clust)
        #     print('Grouping factor is %.2f' % grouping_factor)
        #     sys.exit()

    return delta_obs

def load_delta_obs(obs_loc):
    if isfile(obs_loc):
        delta_obs = pd.read_csv(obs_loc, delim_whitespace=True, header=None)
        delta_obs = xr.DataArray(delta_obs, dims=('Nobs', 'NSV'))

    elif isdir(obs_loc):
        # This function will calculate an m x n matrix. The ith column represents
        # the difference between the ith perturbation run and the base run.
        runs = [d for d in listdir(obs_loc)
                if isdir(obs_loc + '/' + d) & (len(d)==11)]
        runs.sort()

        base_run = pd.read_csv(obs_loc + '/' + runs[0] + '/sat_obs.gosat.00.m',
                               delim_whitespace=True,
                               header=0)['model']*1e9

        delta_obs = np.zeros((len(base_run), len(runs[1:])))
        for i, run in enumerate(runs[1:]):
            #if i % 1 == 0:
            print('%d perturbations processed.' % (i-1))
            fname = obs_loc + '/' + run + '/sat_obs.gosat.00.m'
            delta_obs[:,i] = pd.read_csv(fname,
                                         delim_whitespace=True,
                                         header=0)['model']*1e9 - base_run

    return delta_obs

def get_emis_frac(emis_loc, clusters1, clusters2):
    # We want c2 to be the coarser resolution
    # c1: finer resolution
    # c2: coarser resolution
    n1 = clusters.find_grouping_factor(clusters1)
    n2 = clusters.find_grouping_factor(clusters2)
    if n2 < n1: # If c1 is coarser than c2
        temp = clusters1
        clusters1 = clusters2
        clusters2 = temp

    # Find the mapping from clusters 1 to clusters 2
    cs = clusters.find_cluster_mapping(clusters1, clusters2)

    # Load emissions
    delta_emis = get_delta_emis(clusters1, emis_loc)
    delta_emis = delta_emis.to_dataframe().reset_index()
    delta_emis = delta_emis.rename(columns={'NSV' : '1'})

    # Join in the larger clusters
    delta_emis = delta_emis.merge(cs)[['1', '2', 'emis']]

    # Calculate the total emissions of the larger clusters
    sums2 = delta_emis.groupby('2').sum()['emis'].reset_index()
    sums2 = sums2.rename(columns={'emis' : 'emis2'})

    # Join it back
    delta_emis = delta_emis.merge(sums2)

    # Find the fraction
    delta_emis['frac'] = delta_emis['emis']/delta_emis['emis2']

    # Convert to xarray
    delta_emis = delta_emis.rename(columns={'1' : 'NSV'})
    delta_emis = delta_emis[['NSV', 'frac']].set_index('NSV').to_xarray()

    return delta_emis


def get_emis(emis_loc):
    emis = xr.open_dataset(emis_loc)

    # Take only the first level
    emis = emis.where(emis.lev == emis.lev.min(), drop=True)
    emis = emis.drop(['time', 'lev', 'hyam', 'hybm', 'P0'])
    emis = emis.squeeze(['time', 'lev'])

    # Find total emissions, not considering soil absorption
    emis['EmisCH4_Total'] = emis['EmisCH4_Total'] - emis['EmisCH4_SoilAbsorb']

    # Units are originally in kg/m2/s. Change it to Tg/month.
    emis['EmisCH4_Total'] = emis['EmisCH4_Total']*emis['AREA']*(60*60*24*31)/1e9
    emis = emis['EmisCH4_Total']

    return emis

def add_quadrature(x):
    return np.sqrt((x**2).sum())

def get_error_obs(error_obs_loc, delta_obs):
    so_vec = xr.open_dataset(error_obs_loc)
    so_vec.coords['Nobs'] = np.arange(1, len(so_vec['SO'])+1)
    so_vec = so_vec.where(so_vec.coords['Nobs'].isin(delta_obs.coords['Nobs']), drop=True)
    so_vec = so_vec**2 # get variances from errors

    return so_vec['SO']


def get_obs(obs_loc, delta_obs=None):
    obs = pd.read_csv(obs_loc, delim_whitespace=True,
                      header=0)
    obs['GOSAT'] = obs['GOSAT']*1e9
    obs['model'] = obs['model']*1e9

    if delta_obs is not None:
        obs = obs[obs['NNN'].isin(delta_obs.coords['Nobs'])]

    return obs['GOSAT'], obs['model']


