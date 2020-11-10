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
FORMAT_LOC="/Users/hannahnesser/Documents/Harvard/Research/Python"

## INPUT INSTRUCTIONS
BUILD_RD_JACOBIAN=false
CONDUCT_RR_SENSITIVITY_TESTS=false
COPY_FORMAT_CODE=true

# Copy all python code into the run directory
mkdir -p python
cp -r ../python/*.py ./python/
if [ $COPY_FORMAT_CODE = true ]
then
  cp ${FORMAT_LOC}/format_plots.py ./python/
  cp ${FORMAT_LOC}/config.py ./python/
fi

# Activate the appropriate python environment
source /Users/hannahnesser/opt/anaconda3/etc/profile.d/conda.sh
conda activate reduced_rank_jacobian

# Build and save the clusters in the run dir.
python python/clusters.py $INPUT_DIR $CLUSTERS_IN $RESOLUTION

# Make the true Jacobian
python python/true_jacobian.py $INPUT_DIR $JACOBIAN_STR $RESOLUTION $MODEL_RESPONSE $SAT_OBS

# Make the observational fields and estimated Jacobian
python python/est_jacobian_and_obs.py $INPUT_DIR $JACOBIAN_STR $RESOLUTION $MODEL_EMISSIONS $SAT_OBS

# Make the prior fields
python python/prior.py $INPUT_DIR $MODEL_EMISSIONS $PRIOR_ERROR $RESOLUTION

# If specified, build the reduced-dimension Jacobian
# (This is largely hard coded.)
if [ $BUILD_RD_JACOBIAN = true ]
then
  python python/jacobian_rd.py
fi

# If specified, conduuct the reduced-rank sensitivity tests
# (Again, this is largely hard coded.)
if [ $CONDUCT_RR_SENSITIVITY_TESTS = true ]
then
  python python/jacobian_rr_sensitivity.py
fi

# In all cases, construct the reduced rank Jacobian and
# save out figures 2-5 from Nesser et al. (in prep).
python python/paper_plots.py
