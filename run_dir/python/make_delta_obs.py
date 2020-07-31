import yaml
from os import remove, listdir
from os.path import isfile, isdir, join

import numpy as np
import pandas as pd

runs = [d for d in listdir() if isdir(d) & (len(d)==11)]
runs.sort()

base_run = pd.read_csv(runs[0] + '/sat_obs.gosat.00.m', delim_whitespace=True, header=0)[['NNN', 'model']]
base_run['model'] *= 1e9
base_run = base_run.rename(columns={'model' : 'model_base'})
base_run = base_run.set_index('NNN')

for i, run in enumerate(runs[1:]):
     print('%d perturbations processed.' % (i+1))
     run_i = pd.read_csv(run + '/sat_obs.gosat.00.m', delim_whitespace=True, header=0)[['NNN', 'model']]
     run_i['model'] *= 1e9
     run_i = run_i.set_index('NNN')
     run_i = run_i.rename(columns={'model' : 'model_' + run.split('_')[-1]})
     base_run = base_run.merge(run_i, how='outer', on='NNN')


for i, col in enumerate(base_run.columns):
	if col != 'model_base':
		base_run[col] -= base_run['model_base']


base_run.to_csv('../delta_obs_r2.csv')