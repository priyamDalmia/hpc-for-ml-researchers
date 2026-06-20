#!/bin/bash
# ----- helpers -------------------------------------------------------------
hr() { printf '\n========== %s ==========\n' "$1"; }



# ----- PBS-supplied environment variables ----------------------------------
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

