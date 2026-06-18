#!/bin/bash 
# hello_hpc.sh - small introspection job script for the UTS HPCC. 
# Submit this and analyze the outputs with: qsub hello_hpc.sh


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

hr "JOB START"
echo "date          : $(date -Is)"
echo "hostname      : $(hostname)"
echo "running as    : $(whoami)"
echo "shell PID     : $$"


# ----- PBS-supplied environment variables -----
# PBS_O_*       -> values from the SUBMISSION environment (your login shell)
# PBS_JOBID     -> e.g. 184327.hpcnode0
# PBS_JOBNAME   -> the -N name above
# PBS_QUEUE     -> queue this job was routed to
# PBS_NODEFILE  -> path to a file listing one line per allocated CPU
# NCPUS         -> number of CPUs PBS allocated to this job
hr "PBS ENVIRONMENT"
echo "PBS_JOBID     : ${PBS_JOBID:-<not set>}"
echo "PBS_JOBNAME   : ${PBS_JOBNAME:-<not set>}"
echo "PBS_QUEUE     : ${PBS_QUEUE:-<not set>}"
echo "PBS_O_HOST    : ${PBS_O_HOST:-<not set>}     # host you submitted from"
echo "PBS_O_WORKDIR : ${PBS_O_WORKDIR:-<not set>}  # dir you submitted from"
echo "PBS_O_LOGNAME : ${PBS_O_LOGNAME:-<not set>}"
echo "PBS_NODEFILE  : ${PBS_NODEFILE:-<not set>}"
echo "NCPUS         : ${NCPUS:-<not set>}"
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
    echo "Nodes allocated to this job:"
    sort -u "${PBS_NODEFILE}" | sed 's/^/  /'
fi



# ----- shell-level location and identity -----------------------------------
hr "SHELL / FILESYSTEM"
echo "USER          : ${USER}"
echo "HOME          : ${HOME}"
echo "PWD (now)     : $(pwd)            # note: starts in \$HOME, not workdir"

# Switch to the directory you submitted from (your code lives there).
cd "${PBS_O_WORKDIR}"
echo "PWD (after cd): $(pwd)"


# ----- hardware visible to this job ----------------------------------------
hr "HARDWARE VISIBLE TO THIS JOB"
echo "nproc (allocated CPUs) : $(nproc)"
echo "nproc --all (node total): $(nproc --all)"
echo
echo "CPU model:"
lscpu | grep -E 'Model name|Socket|Core|Thread' | sed 's/^/  /'
echo
echo "Memory (free -h):"
free -h | sed 's/^/  /'
echo
echo "Disk (df -h on key paths):"
df -h "${HOME}" /scratch 2>/dev/null | sed 's/^/  /'


# ----- set up /scratch like the example shows ------------------------------
# Use the local fast SSD; remove on exit so we don't leave debris behind.
hr "SCRATCH SETUP"
SCRATCH="/scratch/${USER}_${PBS_JOBID%.*}"
mkdir -p "${SCRATCH}"
trap 'echo "[trap] cleaning ${SCRATCH}"; rm -rf "${SCRATCH}"' EXIT
echo "SCRATCH       : ${SCRATCH}"
ls -ld "${SCRATCH}"


# ----- python ---------------------------------------------------------------
# activate a virtualenv of your choice ; default python version is 3.6, and latest is 3.12+; so ideally a successful change should reflect that
source ~/venvs/research/bin/activate
hr "PYTHON LAUNCH"
PYTHON_BIN="$(command -v python3)"
echo "python binary : ${PYTHON_BIN}"
echo "python version: $(${PYTHON_BIN} --version)"


# Run the introspection script. We pass the scratch dir as an argument so
# the python side can demonstrate writing there.
${PYTHON_BIN} hello_hpc.py --scratch "${SCRATCH}"