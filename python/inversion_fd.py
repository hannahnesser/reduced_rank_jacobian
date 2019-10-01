import yaml
import xarray as xr

import numpy as np
from numpy.linalg import inv

import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import inv as spinv

from os import remove
import time
import sys

## JACOBIAN
def import_jacobian(obs_criteria, loc='/n/home04/hnesser/RR_inversion/data/'):
    init_time = time.time()
    
    with xr.open_dataset('/n/home00/maasakkers/Adj_Code/Make_J_Dec/J_Emis_Dec_SF_CF.nc')['J'] as k:
        k = k.values
        k = k[:, obs_criteria]
        k[k < 0] = 0
        k = k.T

    final_time = time.time()
    print('Jacobian imported (%.2f s)' % (final_time-init_time))
    return k

## PRIORS
def import_priors(loc='/n/home04/hnesser/RR_inversion/data/'):
    init_time = time.time()

    xa_abs = xr.open_dataset('/n/home00/maasakkers/Adj_Code/Dec_Arrays/Dec_Update_Ems_Array_CF.nc')['J']
    xa_abs_mod = xr.open_dataset('/n/seasasfs02/hnesser/pca/modified_prior/Dec_Update_Ems_Array_CF.nc')['J']
    xa = xa_abs_mod/xa_abs # Calculate relative prior
    xa = xa.values
    xa_abs.close()
    xa_abs_mod.close()

    xtruth = np.ones(len(xa))

    final_time = time.time()
    print('Priors imported (%.2f s)' % (final_time-init_time))
    return xa.reshape(-1,1), xtruth.reshape(-1,1)

## OBSERVATIONS
def import_obs(obs_type, obs_criteria, loc='data/'):
    '''
    Takes obs_type in ['gosat', 'pseudo_noise', or 'pseudo_nn']
    '''
    init_time = time.time()

    y_base = xr.open_dataset('/n/home00/maasakkers/Adj_Code/Dec_Linear/April/Dec_MODEL_CF_Monthafter.nc')['J']
    y_base.close()
    y_base = y_base.values
    y_base = y_base[obs_criteria]

    if obs_type == 'gosat':
        y =  xr.open_dataset('/n/home00/maasakkers/Adj_Code/Dec_Linear/April/Dec_GOSAT_CF_Monthafter.nc')['J']
        y.close()
        y = y.values
        y = y[obs_criteria]
    elif obs_type == 'pseudo_noise':
        y = xr.open_dataset('/n/seasasfs02/hnesser/pca/yy_pseudo_noise.nc')['yy_pseudo']
        y.close()
        y = y.values
    elif obs_type == 'pseudo_nn':
        y = xr.open_dataset('n/seasasfs02/hnesser/pca/yy_pseudo_nn.nc')['yy_pseudo']
        y.close()
        y = y.values
    
    final_time = time.time()
    print('Observations imported (%.2f s)' % (final_time-init_time))
    return y.reshape(-1,1), y_base.reshape(-1,1)

## ERROR
def import_error(obs_criteria, lambda_reg=10, loc='data/'):
    init_time = time.time()

    # observational error
    so_v = xr.open_dataset('/n/home00/maasakkers/Adj_Code/Dec_Linear/April/Dec_OBSERR_CF_Monthafter.nc')['J']
    so_v.close()
    so_v = so_v.values
    so_v = so_v[obs_criteria]
    so_v = (1/lambda_reg)*so_v**2
    
    # prior error
    sa_v = xr.open_dataset('/n/seasasfs02/hnesser/pca/modified_prior/Dec_Update_Ems_Error_CF.nc', lock=True)['J']
    sa_v.close()
    sa_v = sa_v.values
    sa_v = sa_v**2

    final_time = time.time()
    print('Error imported (%.2f s)' % (final_time-init_time))
    return so_v, sa_v

## CRITERIA
def import_criteria():
    init_time = time.time()

    y_yrs = xr.open_dataset('/n/home00/maasakkers/Adj_Code/Dec_Arrays/GOSAT_Years.nc')['J']
    y_glint = xr.open_dataset('/n/home00/maasakkers/Adj_Code/Dec_Arrays/GOSAT_Glint.nc')['J']
    y_lat = xr.open_dataset('/n/home00/maasakkers/Adj_Code/Dec_Arrays/GOSAT_Lat.nc')['J']
    criteria = (y_yrs>2009) & (y_glint<0.5) & (y_lat<60)
    criteria = criteria.values
    
    y_yrs.close()
    y_glint.close()
    y_lat.close()

    final_time = time.time()
    print('Criteria imported (%.2f s)' % (final_time-init_time))
    return criteria

def obs_mod_diff(x, y, k, c):
    return (y - (k @ x + c))

def cost(x, xa, so_inv, sa_inv, diff_func=obs_mod_diff):
    init_time = time.time()

    obso = diff_func(x)
    cost_obs = obso.T @ so_inv @ obso
    cost_emi = (x - xa).T @ sa_inv @ (x - xa)

    final_time = time.time()
    print('Cost function calculated (%.2f s)' % (final_time-init_time))
    return cost_obs + cost_emi

if __name__ == '__main__':
    init_time = time.time()
    
    y_criteria = import_criteria()
    k          = import_jacobian(obs_criteria=y_criteria)
    xa, xtruth = import_priors()
    y, y_base  = import_obs('pseudo_noise', obs_criteria=y_criteria)
    so_v, sa_v = import_error(obs_criteria=y_criteria)
    
    so_inv = sparse.diags(1/so_v)
    sa_inv = sparse.diags(1/sa_v)

    task_time = time.time()
    c = y_base - k @ xtruth
    t = time.time()
    print('C calculated (%.2f s)' % (t-task_time))

    # define local obs_mod_diff and cost functions.
    obs_mod_diff_func = lambda x: obs_mod_diff(x, y=y, k=k, c=c)
    cost_func = lambda x: cost(x, 
                               xa=xa, 
                               so_inv=so_inv, 
                               sa_inv=sa_inv,
                               diff_func=obs_mod_diff_func)

    cost_prior = cost_func(xa)
    print('    Iteration: %d\n    Cost Function: %.2f\n    Negative Cells: %d' % (-1, cost_prior, xa[xa < 0].shape[0]))

    task_time = time.time()
    shat = inv(k.T @ so_inv @ k + sa_inv)
    t = time.time()
    print('Posterior error covariance calculated (%.2f s)' % (t-task_time))

    task_time = time.time()
    g = shat @ k.T @ so_inv
    t = time.time()
    print('Gain matrix calculated (%.2f s)' % (t-task_time))

    task_time = time.time()
    xhat = np.asarray(xa + (g @ obs_mod_diff_func(xa)))
    t = time.time()
    print('Posterior estimate calculated (%.2f s)' % (t-task_time))

    cost_post = cost_func(xhat)
    print('    Iteration: %d\n    Cost Function: %.2f\n    Negative Cells: %d' % (1, cost_post, xhat[xhat < 0].shape[0]))

    # Calculate the averaging kernel
    task_time = time.time()
    a = sparse.identity(len(xa)) - shat @ sa_inv
    t = time.time()
    print('Averaging kernel calculated (%.2f s)' % (t-task_time))

    # Calculate y out
    task_time = time.time()
    y_out = k @ xhat + c
    t = time.time()
    print('Updated model observations calculated (%.2f s)' % (t-task_time))

    # Inversion complete.
    tot_time = t-init_time
    print('INVERSION COMPLETE (%.0f m %.0f s)' % (divmod(tot_time, 60)))
    print('Output saving to netcdf.')
    
    ## Save output.
    loc = str(sys.argv[1])
    
    # c
    temp = xr.Dataset({'J' : (['obs'], c.reshape(-1))}, attrs={'Description' : 'c'})
    #remove(loc + '/c.nc')
    temp.to_netcdf(loc + '/c.nc')
    
    # a
    temp = xr.Dataset({'J' : (['posterior', 'truth'], a)}, attrs={'Description' : 'a', 'Rows' : 'Posterior', 'Columns' : 'Truth'})
    #remove(loc + '/a.nc')
    temp.to_netcdf(loc + '/a.nc')

    # shat
    temp = xr.Dataset({'J' : (['a', 'b'], shat)}, attrs={'Description' : 'shat'})
    #remove(loc + '/shat.nc')
    temp.to_netcdf(loc + '/shat.nc')
    
    # xhat
    temp = xr.Dataset({'J' : (['state'], xhat.reshape(-1))}, attrs={'Description' : 'xhat'})
    #remove(loc + '/xhat.nc')
    temp.to_netcdf(loc + '/xhat.nc')

    # yout
    temp = xr.Dataset({'J' : (['obs'], y_out.reshape(-1))}, attrs={'Description' : 'yout'})
    #remove(loc + '/yout.nc')
    temp.to_netcdf(loc + '/yout.nc')

    print('All files saved.')
