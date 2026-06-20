#!/bin/bash

#PBS -l ncpus=1
#PBS -l mem=4gb
#PBS -l walltime=00:10:00

#PBS -m a 
#PBS -M priyam.dalmai@uts.edu.au 
#PBS -j oe

#PBS -o logs/ 

# ----- bash safety flags ---------------------------------------------------
# -e : exit on any command failure
# -u : error on unset variables
# -o pipefail : a pipeline fails if any stage fails (not just the last)
set -euo pipefail


# ----- helpers -------------------------------------------------------------
hr() { printf '\n========== %s ==========\n' "$1"; }

hr "JOB START"
echo "date          : $(date -Is)"
echo "hostname      : $(hostname)"
echo "running as    : $(whoami)"
echo "shell PID     : $$"

# ----- shell-level location and identity -----------------------------------
hr "SHELL / FILESYSTEM"
echo "USER          : ${USER}"
echo "HOME          : ${HOME}"
echo "PWD (now)     : $(pwd)"
# Switch to the directory you submitted from (your code lives there).
cd "${PBS_O_WORKDIR}"
echo "PWD (after cd): $(pwd)"

# ----- IF REQUIRED: run_smmary -------------------------------------------------------------
./job_summary.sh
# ./job_summary.py # TODO improve 

hr "SETUP"
echo "date          : $(date -Is)"

# ----- python ---------------------------------------------------------------
# We're using the system python here so the script is self-contained.
# In real jobs you'd activate your venv/conda env before this.

# PYTHON_BIN="$(command -v python3)"
PYTHON_BIN="$(command -v python3)"
echo "python binary : ${PYTHON_BIN}"
echo "python version: $(${PYTHON_BIN} --version)"

# ----- set up /scratch like the example shows ------------------------------
# Use the local fast SSD; remove on exit so we don't leave debris behind.

# # hr "SCRATCH SETUP"
# SCRATCH="/scratch/${USER}_${PBS_JOBID%.*}"
# JOB_DIR="${SCRATCH}/${PBS_JOBID%.*}"
# mkdir -p "${JOB_DIR}"
# mkdir -p "${SCRATCH}"
# trap 'echo "[trap] cleaning ${SCRATCH}"; rm -rf "${SCRATCH}"' EXIT
# echo "SCRATCH       : ${SCRATCH}"
# ls -ld "${SCRATCH}"


#  Copy code + data into scratch (or symlink read-only datasets)
# rsync -a --exclude='.git' ./code/ "${SCRATCH}/code/"
# cp -r ./data "${JOB_DIR}/"
# mkdir -p "${JOB_DIR}/results"

# activate ml
echo "Activating new Python environment"

VENV_DIR="${HOME}/venvs/ml"
source "${VENV_DIR}/bin/activate"

PYTHON_BIN="$(command -v python3)"
PIP_BIN="$(command -v pip)"

echo "python binary  : ${PYTHON_BIN}"
echo "python version : $(${PYTHON_BIN} --version)"
echo "pip binary     : ${PIP_BIN}"
echo "venv path      : ${VIRTUAL_ENV}"
echo "working dir    : $(pwd)"


"${PYTHON_BIN}" - <<'PY'
import sys
import site
import os

print(f"sys.executable : {sys.executable}")
print(f"sys.prefix     : {sys.prefix}")
print(f"base_prefix    : {sys.base_prefix}")
print(f"in venv        : {sys.prefix != sys.base_prefix}")
print(f"site-packages  : {site.getsitepackages()}")
print(f"VIRTUAL_ENV    : {os.environ.get('VIRTUAL_ENV')}")
PY

# Run the introspection script. We pass the scratch dir as an argument so
# the python side can demonstrate writing there.
# ${PYTHON_BIN} "${PBS_O_WORKDIR}/hello_hpc.py" --scratch "${SCRATCH}"

hr "MAIN"

${PYTHON_BIN} 10_numbers.py

# ----- demonstrate copy-back from scratch ----------------------------------
# The python script writes a small report into SCRATCH; copy it back so it
# survives after the trap cleans /scratch.
# hr "COPY RESULTS BACK"
# if [[ -f "${SCRATCH}/python_report.txt" ]]; then
#    cp "${SCRATCH}/python_report.txt" "${PBS_O_WORKDIR}/python_report_${PBS_JOBID%.*}.txt"
#   echo "saved: ${PBS_O_WORKDIR}/python_report_${PBS_JOBID%.*}.txt"
# fi

hr "JOB END"
echo "date          : $(date -Is)"
