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

def update_krr(fwd_model, rank, k_prev, sa_vec, so_vec, threshold=0.1):
    pph = inv_rr.calc_pph(k_prev, sa_vec, so_vec)
    evals, evecs = inv_rr.e_decomp(pph)
    wk = evecs[:,:rank]
    prolong = (wk.T * (sa_vec**0.5)).T

    kw = fwd_model @ prolong
    k_update = np.zeros(k_prev.shape)
    pbar = tqdm(total=len(k_update.flatten()))
    for i in range(k_update.shape[0]):
        for j in range(k_update.shape[1]):
            pbar.update(1)
            numer = (kw[i,:]*prolong[j,:]).sum()
            denom = (prolong[j,:]**2).sum()
            if denom > threshold:
                k_update[i,j] = numer/denom

    return k_update