import copy
import os
import sys

sys.path.append('./python/')
import inversion as inv
import jacobian as j
# import inv_plot

import xarray as xr
import numpy as np
import pandas as pd
from scipy.sparse import diags, identity
from scipy import linalg

from matplotlib import colorbar, colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
%matplotlib inline
c = plt.cm.get_cmap('inferno', lut=10)

# Import clusters
clusters = xr.open_dataarray('../input/clusters_1x125.nc')
clusters_plot = xr.open_dataarray('../input/clusters_1x125_plot.nc')

# Load estimated and true Jacobian
k_est = xr.open_dataarray('../input/k_est.nc')
k_est_sparse = xr.open_dataarray('../input/k_est_sparse.nc')
k_true = xr.open_dataarray('../input/k_true.nc')

# Load prior and error
xa = xr.open_dataarray('../input/xa.nc')
sa_vec = xr.open_dataarray('../input/sa_vec.nc')

# Load observations and error
y = xr.open_dataarray('../input/y.nc')
y_base = xr.open_dataarray('../input/y_base.nc')
so_vec = xr.open_dataarray('../input/so_vec.nc')

###############################################################
##################           TRUTH          ###################
###############################################################

# Create a true Reduced Rank Jacobian object
true = inv.ReducedRankJacobian(k_true.values, 
                               xa.values, 
                               sa_vec.values, 
                               y.values,
                               y_base.values, 
                               so_vec.values)

# We will use a regularization factor of 20.
true.rf = 1 #10

# Complete an eigendecomposition of the prior pre-
# conditioned Hessian, filling in the eigenvalue 
# and eigenvector attributes of true.
true.edecomp()

# Solve the inversion, too.
true.solve_inversion()

###############################################################
#############          INITIAL ESTIMATE          ##############
###############################################################

est0 = inv.ReducedRankJacobian(k_est.values, 
                               xa.values, 
                               sa_vec.values, 
                               y.values,
                               y_base.values, 
                               so_vec.values)
est0.rf = 1 #10
est0.edecomp()
est0.solve_inversion()

###############################################################
#############          CONVERGENCE TESTS          #############
###############################################################

# Broyden
ranks = np.append(np.arange(50, 600, 50), np.arange(600, 2100, 100))
for r in ranks:
    est1 = est0.update_jacobian_br(true.k, rank=r,
                                   k_base=np.zeros(est0.k.shape))
    np.savetxt('k_%d.csv' % r, est1.k)

# est2 = est0.update_jacobian_ag()
