#!/bin/bash

#PBS -l ncpus=1
#PBS -l mem=16gb
# PBS -l ngpus=1
#PBS -l walltime=00:20:00 
#PBS -o logs

#PBS -m e
#PBS -M priyam.dalmia@student.uts.edu.au
#PBS -j oe 

set -euo pipefail                   # fail fast on errors

# 1. Stage scratch
SCRATCH="/scratch/${USER}"
JOB_DIR="${SCRATCH}/${PBS_JOBID%.*}"
mkdir -p "${JOB_DIR}"
trap 'rm -rf "${JOB_DIR}"' EXIT     # always clean up, even on failure

# 2. Move to the directory the job was submitted from
# By default, at $HOME 
cd "${PBS_O_WORKDIR}"

# 3. Copy code + data into scratch (or symlink read-only datasets)
# rsync -a --exclude='.git' ./code/ "${SCRATCH}/code/"
cp -r ./data "${JOB_DIR}/"
mkdir -p "${JOB_DIR}/results"

# 4. Activate environment
source "${HOME}/venvs/ml/bin/activate"

# 5. Run 
python hello_hpc.py

# 5. Run from scratch
# cd "${SCRATCH}/code"
# python train.py \
#    --data-dir "${SCRATCH}/data" \
#    --output-dir "${SCRATCH}/results" \
#    --seed 42

# 6. Copy results back; scratch is wiped by the trap
mkdir -p "${PBS_O_WORKDIR}/results/${PBS_JOBID%.*}"
cp -r "${JOB_DIR}/results/." "${PBS_O_WORKDIR}/results/${PBS_JOBID%.*}/"
