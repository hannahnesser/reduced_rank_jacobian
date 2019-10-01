from os import listdir
from os.path import join
import sys
sys.path.append('.')
import jacobian as j
import inv_plot

import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
colors = plt.cm.get_cmap('inferno', lut=9)

input_dir = sys.argv[1]
jac_str = str(sys.argv[2])
resolution = str(sys.argv[3])
model_emissions = sys.argv[4]
sat_obs = sys.argv[5]

if jac_str + '_est.nc' not in listdir(input_dir):
    print('Building estimated Jacobian based on mass balance approach.')

    clusters = xr.open_dataarray(join(input_dir, 'clusters_' + resolution + '.nc'))
    clusters_reduced = xr.open_dataarray(join(input_dir, 'clusters_' + resolution + '_plot.nc'))

    emis = j.get_delta_emis(clusters_long=clusters, 
                            emis_loc=join(input_dir, model_emissions))

    obs = pd.read_csv(join(input_dir, sat_obs), delim_whitespace=True, header=0)

    obs['GOSAT'] *= 1e9
    obs['model'] *= 1e9
    obs = obs[obs['GLINT'] == False]

    obs = obs[['NNN', 'LON', 'LAT', 'GOSAT', 'model']]
    obs = obs.rename(columns={'LON' : 'lon', 
                              'LAT' : 'lat',
                              'NNN' : 'Nobs'})

    lat_width = np.diff(clusters_reduced.coords['lat'].values)[0]/2
    lat_edges = clusters_reduced.coords['lat'].values - lat_width
    lat_edges = np.append(lat_edges, clusters_reduced.coords['lat'].values[-1]+lat_width)
    obs['lat_bin'] = pd.cut(obs['lat'], 
                            bins=lat_edges, 
                            labels=clusters_reduced.coords['lat'].values)
    obs['lat_bin'] = obs['lat_bin'].astype(float)

    lon_width = np.diff(clusters_reduced.coords['lon'].values)[0]/2
    lon_edges = clusters_reduced.coords['lon'].values - lon_width
    lon_edges = np.append(lon_edges, clusters_reduced.coords['lon'].values[-1]+lon_width)
    obs['lon_bin'] = pd.cut(obs['lon'], 
                            bins=lon_edges, 
                            labels=clusters_reduced.coords['lon'].values)
    obs['lon_bin'] = obs['lon_bin'].astype(float)

    obs = obs.rename(columns={'lon' : 'lon_true',
                              'lat' : 'lat_true',
                              'lon_bin' : 'lon',
                              'lat_bin' : 'lat'})

    clusters_reduced_df = clusters_reduced.to_dataframe('Clusters').reset_index()
    obs = obs.merge(clusters_reduced_df, on=['lat', 'lon'], how='left')
    obs = obs.rename(columns={'Clusters' : 'NSV'})
    obs = obs.sort_values(by='Nobs')

    # # Calculate the change above each grid cell
    # get moles air
    emis_loc = join(input_dir, 'HEMCO_diagnostics.200907010000.nc')
    emis_pert = xr.open_dataset(emis_loc)
    emis_pert = emis_pert.where(emis_pert.lev == emis_pert.lev.min(), drop=True)
    emis_pert = emis_pert.drop(['time', 'lev', 'hyam', 'hybm', 'P0'])
    emis_pert = emis_pert.squeeze(['time', 'lev'])
    emis_pert['EmisCH4_Total'] = emis_pert['EmisCH4_Total'] - emis_pert['EmisCH4_SoilAbsorb']

    # Units are originally in kg/m2/s. Change it to mole/s. (Normally we do per month
    # but we are converting to observations...)
    emis_pert['EmisCH4_Total'] = 0.5*emis_pert['EmisCH4_Total']*emis_pert['AREA']/0.016

    # Group by NSV
    emis_pert['NSV'] = clusters
    emis_pert = emis_pert.groupby('NSV').sum(xr.ALL_DIMS)[['EmisCH4_Total', 'AREA']]
    emis_pert = emis_pert.where(emis_pert['NSV'] > 0, drop=True)

    # Calculate ppb using Daniel's equation...
    Mair = 0.02897 #kg/mol
    P = 100000 # Pa
    g = 9.8 # m/s^2
    U = 5*(1000/3600) # 5 km/hr in m/s
    W = emis_pert['AREA']**0.5 # m
    emis_pert['ppbv_CH4'] = 1e9*Mair*emis_pert['EmisCH4_Total']*g/(U*W*P)


    c = clusters_reduced.stack(NSV=['lat', 'lon'])
    c = c.where(c > 0, drop=True)
    c = c.reset_index(['NSV', 'lat', 'lon'])
    c['NSV'] = c.values
    c = c.coords.to_dataset()
    emis_pert = c.merge(emis_pert)

    def set_res_dec(res_str):
        if len(res_str) > 1:
            return int(res_str)/(10**(len(res_str)-1))
        else:
            return int(res_str)

    def get_res(res_str):
        res_lat = set_res_dec(res_str.split('x')[0])
        res_lon = set_res_dec(res_str.split('x')[-1])
        return(res_lat, res_lon)

    def condition(pert, pert_loc, resolution, factor):
        res_lat, res_lon = get_res(resolution)
        condition = (pert['lat'] <= pert_loc['lat'].values + factor*res_lat) &\
                    (pert['lat'] >= pert_loc['lat'].values - factor*res_lat) &\
                    (pert['lon'] <= pert_loc['lon'].values + factor*res_lon) &\
                    (pert['lon'] >= pert_loc['lon'].values - factor*res_lon)
        return condition

    # for i in a.coords['NSV'].values:
    y_short = obs[['Nobs', 'NSV']]
    for i in emis_pert.coords['NSV'].values:
        pert = emis_pert.where(emis_pert.coords['NSV'] == i)
        pert_loc = emis_pert.where(emis_pert.coords['NSV'] == i, drop=True)

        cond2 = condition(pert, pert_loc, resolution, 2)
        cond1 = condition(pert, pert_loc, resolution, 1)
        cond0 = condition(pert, pert_loc, resolution, 0)

        # We apply cond1 second, so we need it to be the same shape
        # as our perturbation realm.
        cond1 = cond1.where(cond2, drop=True).astype(bool)
        cond0 = cond0.where(cond2, drop=True).astype(bool)

        # print(cond1)
        
        pert_nb = emis_pert.where(cond2, drop=True)
        pert_nb['EmisCH4_Total'] *= 0.2/(cond2.sum()- cond1.sum())*pert_loc['EmisCH4_Total'].values[0]/pert_nb['EmisCH4_Total']
        pert_nb['EmisCH4_Total'] = pert_nb['EmisCH4_Total'].where(~cond1,
                                                                  0.3/(cond1.sum()-cond0.sum())*pert_loc['EmisCH4_Total'].values[0])
        pert_nb['EmisCH4_Total'] = pert_nb['EmisCH4_Total'].where(~cond0,
                                                                  0.5*pert_loc['EmisCH4_Total'].values[0])
        
        W = pert_nb['AREA']**0.5 # m
        pert_nb[i] = 1e9*Mair*pert_nb['EmisCH4_Total']*g/(U*W*P)
        pert_nb = pert_nb[i].to_dataframe().reset_index()
        pert_nb = pert_nb.drop(columns=['lat', 'lon'])
        y_short = y_short.merge(pert_nb, on=['NSV'], how='left')
        y_short[i] = y_short[i].fillna(0)

    k_est = y_short.drop(columns=['NSV'])/0.5
    k_est = k_est.set_index('Nobs')
    k_est = k_est.unstack().reset_index()
    k_est = k_est.rename(columns={'level_0' : 'NSV',
                                   0 : 'k'})
    k_est = k_est.set_index(['Nobs', 'NSV']).to_xarray()
    k_est = k_est['k']
    k_est.to_netcdf(join(input_dir, jac_str + '_est.nc'))

    # We can check our results by plotting the regridded Jacobian against
    # the true jacobian
    print(k_est.shape)
    k_true = xr.open_dataarray(join(input_dir, 'k_true.nc'))
    print(k_true.shape)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_facecolor('0.98')
    ax.scatter(k_true.values, k_est.values, alpha=0.5, s=5, c=colors(3))
    ax.plot((-2,20), (-2,20), c='0.5', lw=2, ls=':', zorder=0)
    ax.set_ylim(-2,20)
    ax.set_xlim(-2,20)
    ax.set_xlabel('True Jacobian', fontsize=18)
    ax.set_ylabel('Estimated Jacobian', fontsize=18)
    ax.set_title('Estimated vs. True Jacobian', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(join(input_dir, 'k_est.png'))

else:
    print('Estimated Jacobian already exists.')
