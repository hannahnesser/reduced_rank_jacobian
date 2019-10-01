import yaml
import xarray as xr

import numpy as np
from numpy.linalg import inv

import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import inv as spinv

from os import remove, listdir
from os.path import isfile, join
import time
import sys
sys.path.append('/n/home04/hnesser/inversion/python')
#import plot as inv_plot
import imports as imp

def shat_p_sum(p, evals, evecs):
    sum_mat = np.zeros((evecs.shape[0], evecs.shape[0]))
    for i in range(p):
        l_i = evals[i]
        v_i = evecs[:,i].reshape(-1,1)
        sum_mat += (v_i @ v_i.T)/(1 + l_i)
    return sum_mat

def a_p_sum(p, evals, evecs):
    sum_mat = np.zeros((evecs.shape[0], evecs.shape[0]))
    for i in range(p):
        l_i = evals[i]
        v_i = evecs[:,i].reshape(-1,1)
        sum_mat += l_i*(v_i @ v_i.T)/(1 + l_i)
    return sum_mat

def xhat_p_sum(p, evals, evecs, so_v, sa_v, k):
    sum_mat = np.zeros((k.shape[1], k.shape[0]))
    so_sqrt_inv = sparse.diags(1/so_v**0.5)
    sa_sqrt = sparse.diags(sa_v**0.5)
    for i in range(p):
        l_i = evals[i]
        v_i = evecs[:,i].reshape(-1,1)
        w_i = so_sqrt_inv @ k @ sa_sqrt @ v_i
        sum_mat += l_i**0.5/(1+l_i)*v_i @ w_i.T
    return sum_mat

def calc_pph(k, sa_v, so_v):
    # Calculate the prior-preconditioned Hessian
    task_time = time.time()
    so_inv = sparse.diags(1/so_v)
    sa_sqrt = sparse.diags(sa_v**0.5)
    brackets = sa_sqrt @ k.T
    pph = brackets @ so_inv @ brackets.T
    t = time.time()
    print('Prior-preconditioned Hessian calculated (%.2f s)' % (t-task_time))

    return pph

def e_decomp(matrix):
    # Perform the eigenvalue decomposition
    task_time = time.time()
    evals, evecs = np.linalg.eigh(matrix)

    # sort by eigenvalue
    evals = evals[::-1]
    evecs = evecs[:,::-1]
    t = time.time()
    print('Eigendecomposition complete (%.2f s)' % (t-task_time))

    return evals, evecs

def max_dofs_proj(p, evals, evecs, k, xa, sa_v, y, so_v, c):
    task_time = time.time()
    
    sa_sqrt = sparse.diags(sa_v**0.5)
    sa_sqrt_inv = sparse.diags(1/sa_v**0.5)
    so_inv = sparse.diags(1/so_v)
    so_sqrt_inv = sparse.diags(1/so_v**0.5)

    # Calculate posterior covariance for a given p
    subtask_time = time.time()
    shat_p = sa_sqrt @ shat_p_sum(p, evals, evecs) @ sa_sqrt
    print('Reduced rank posterior error covariance calculated (%.2f s)' % (time.time() - subtask_time))
    
    # Posterior state
    #xhat_p = sa_sqrt @ xhat_p_sum(p, evals, evecs, so_v, sa_v, k)
    #xhat_p = xhat_p @ so_sqrt_inv @ (y - (k @ xa + c))
    subtask_time = time.time()
    xhat_p = k.T @ so_inv @ (y - (k @ xa + c))
    xhat_p = shat_p @ xhat_p + xa
    print('Reduced rank posterior estimate calculated (%.2f s)' % (time.time() - subtask_time))
    
    # Averaging kernel
    subtask_time = time.time()
    a_p = sa_sqrt @ a_p_sum(p, evals, evecs) @ sa_sqrt_inv
    print('Reduced rank averaging kernel calculated (%.2f s)' % (time.time() - subtask_time))

    # DOFS
    dofs_p = np.sum(evals[:p]/(1+evals[:p]))
    
    t = time.time()
    print('Maximum DOFS projection solution complete (%.2f s)' % (t-task_time))
    print('    DOFS (algebraic):  %.2f' % dofs_p)
    print('    DOFS (trace of A): %.2f' % np.trace(a_p))

    return xhat_p, shat_p, a_p, dofs_p

def proj_fwd_mod(p, evals, evecs, xhat_p, shat_p, a_p, sa_v):
    task_time = time.time()
    shat_k_p = np.asarray(sparse.identity(a_p.shape[0]) - a_p) @ sparse.diags(sa_v)
    xhat_k_p = xhat_p
    a_k_p = a_p
    t = time.time()
    print('Projected forward model solution complete (%.2f s)' % (t-task_time))
    return xhat_k_p, shat_k_p, a_k_p

if __name__ == '__main__':
    start_time = time.time()

    # Import stuff

    #Inputs
    #Regularization factor
    if len(sys.argv) > 3:
        rf = float(sys.argv[3])
        print(rf)
    else:
        rf = 1

    # Location for storing output
    loc = str(sys.argv[1])
    # delete any files in there currently.
    existing_files = [join(loc, f) 
                      for f in listdir(loc) 
                      if isfile(join(loc, f))
                      & (f[-3:] == '.nc')]
    if len(existing_files) > 0:
        for file in existing_files:
            remove(file)

    # read in p that will be used to determine rank
    p = int(sys.argv[2])

    print('p = %d, rf = %.4f' % (p, rf))

    y_criteria = imp.import_criteria()
    k = imp.import_jacobian(obs_criteria=y_criteria)
    xa, xtruth, xtruth_abs = imp.import_priors()
    y, y_base = imp.import_obs('pseudo_noise', obs_criteria=y_criteria)
    so_v, sa_v = imp.import_error(obs_criteria=y_criteria, lambda_reg=rf)
    print('    Regularizing Factor = %d' % rf)

    task_time = time.time()
    c = y_base - k @ xtruth
    t = time.time()
    print('C calculated (%.2f s)' % (t-task_time))

    # Calculate the prior-preconditioned Hessian:
    pph = calc_pph(k, sa_v, so_v)

    # Calculate eigendecomposition
    evals, evecs = e_decomp(pph)

    # Check for imaginary values in eigenvalues
    if np.any(np.iscomplex(evals)):
        last_real_idx = np.where(np.iscomplex(evals))[0][0] - 1
        if (p - 1) > last_real_idx:
            print('Imaginary eigenvalues > 0 exist within rank p at index %d. Continuing against better judgment.' % last_real_idx)
            #exit()
        else:
            print('Imaginary eigenvalues > 0 exist, but not within rank p. Continuing inversion.')

    evals = np.real(evals)

    # check for imaginary values in eigenvectors
    if np.any(np.iscomplex(evecs)):
        last_real_idx = np.where(np.iscomplex(evecs))[1][0] - 1
        if (p - 1) > last_real_idx:
            print('Imaginary eigenvalues > 0 exist within rank p at index %d. Continuing against better judgment.' % last_real_idx)
            #exit()
        else:
            print('Imaginary eigenvalues > 0 exist, but not within rank p. Continuing inversion.')

    evecs = np.real(evecs)

    # evecs
    temp = xr.Dataset({'evecs' : (['elements', 'idx'], evecs), 
                       'evals' : (['idx'], evals)},
                      attrs={'Description' : 'eigenvectors and eigenvalues'})
    temp.to_netcdf(loc + '/evs.nc')

    evals = evals[:p]
    evecs = evecs[:,:p]
    
    # Complete the max DOFS projection
    xhat_p, shat_p, a_p, dofs_p = max_dofs_proj(p=p, 
                                                evals=evals, evecs=evecs, 
                                                k=k, 
                                                xa=xa, sa_v=sa_v, 
                                                y=y, so_v=so_v,
                                                c=c)

    # Complete the projected forward model solution
    xhat_k_p, shat_k_p, a_k_p = proj_fwd_mod(p=p,
                                               evals=evals, evecs=evecs,
                                               xhat_p = xhat_p, shat_p=shat_p, a_p=a_p,
                                               sa_v=sa_v)

    # Complete the full rank approximation
    task_time = time.time()
    xhat_fr_p = k.T @ sparse.diags(1/so_v) @ (y - (k @ xa + c))
    xhat_fr_p = shat_k_p @ xhat_fr_p
    t = time.time()
    print('FR approximation solution complete (%.2f s)' % (t-task_time))

    # Save output
    final_time = time.time()
    print('ALL INVERSIONS COMPLETE (%.0f m %.0f s)' % (divmod(final_time-start_time, 60)))
    print('Saving output.\n')

    # a_p
    temp = xr.Dataset({'J' : (['posterior', 'truth'], a_p)}, attrs={'Description' : 'a', 'Rows' : 'Posterior', 'Columns' : 'Truth'})
    temp.to_netcdf(loc + '/a_p.nc')

    # shat_p
    temp = xr.Dataset({'J' : (['a', 'b'], shat_p)}, attrs={'Description' : 'shat'})
    temp.to_netcdf(loc + '/shat_p.nc')

    # shat_k_p
    temp = xr.Dataset({'J' : (['a', 'b'], shat_k_p)}, attrs={'Description' : 'shat'})
    temp.to_netcdf(loc + '/shat_k_p.nc')    

    # xhat
    temp = xr.Dataset({'J' : (['state'], xhat_p.reshape(-1))}, attrs={'Description' : 'xhat'})
    temp.to_netcdf(loc + '/xhat_p.nc')

    # xhat_fr
    temp = xr.Dataset({'J' : (['state'], xhat_fr_p.reshape(-1))}, attrs={'Description' : 'xhat FR'})
    temp.to_netcdf(loc + '/xhat_fr_p.nc')

# Check on if c needs to be incorporated in the posterior state vector calculation.

# Check: that the solution converges to the full rank solution as p --> n

# plot of eigenvectors (we can call these the principal components)--so need to return the eigenvectors and eigenvalues
