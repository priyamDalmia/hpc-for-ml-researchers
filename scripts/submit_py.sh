#!/bin/bash 
# submit_py.sh - submits a single script on a single CPU 
# Submit this and analyze the outputs with: qsub submit_py.sh


#PBS -N hello_hpc
#PBS -l ncpus=2
#PBS -l mem=4gb
#PBS -l walltime=00:10:00
#PBS -m ae
#PBS -M priyam.dalmia@student.uts.edu.au 

# merge stderr and stdout 
#PBS -j oe 

# ----- bash safety flags, BEST PRACTICES -----
# no experiment is better than uncontrolled experiment 
# use safe - grep "loss" train.log > loss_lines.txt || true
# use safe defaults - echo "${MY_VAR:-default_value}"
# -e: exit on any command failure 
# -u: error on unset variables 
# -o pipefail: a pipeline fails if any stage fails (not just the last)
set -euo pipefail

# ----- helpers -----
hr() { printf '\n========== %s ==========\n' "$1"; }

# Switch to the directory you submitted from (code lives there).
cd "${PBS_O_WORKDIR}"
echo "PWD (after cd): $(pwd)"


# ----- python ---------------------------------------------------------------
# activate a virtualenv of your choice ; default python version is 3.6, and latest is 3.12+; so ideally a successful change should reflect that
source ~/venvs/research/bin/activate
hr "PYTHON LAUNCH"
PYTHON_BIN="$(command -v python3)"
echo "python binary : ${PYTHON_BIN}"
echo "python version: $(${PYTHON_BIN} --version)"

${PYTHON_BIN} evaluation/run_r1.py