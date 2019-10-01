#!/bin/bash

INPUT_DIR="../input"

## INPUT FILE NAMES
CLUSTERS_IN="clusters_raw" # clusters_raw_[size].nc
RESOLUTION="1x125"
JACOBIAN_STR="k"
MODEL_RESPONSE="delta_obs.csv"
MODEL_EMISSIONS="base_emis.nc"
SAT_OBS="sat_obs.gosat.00.m"
PRIOR_ERROR="relative_errors.nc"

# Copy all python code into the run directory
mkdir -p python
cp -r ../python/* ./python/

# Build and save the clusters in the run dir.
python -W ignore python/clusters.py $INPUT_DIR $CLUSTERS_IN $RESOLUTION

# Make the true Jacobian
python -W ignore python/true_jacobian.py $INPUT_DIR $JACOBIAN_STR $RESOLUTION $MODEL_RESPONSE $SAT_OBS

# Make the observational fields and estimated Jacobian
python -W ignore python/est_jacobian_and_obs.py $INPUT_DIR $JACOBIAN_STR $RESOLUTION $MODEL_EMISSIONS $SAT_OBS

# Make the prior fields
python -W ignore python/prior.py $INPUT_DIR $PRIOR_ERROR $RESOLUTION