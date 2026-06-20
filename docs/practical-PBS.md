# Using PBS Pro  

Basic PBS commands
Task	Command
Submit job	qsub job.pbs
Show all **jobs**	qstat -a
Show my jobs	qstat -u "$USER"
Watch my jobs	watch -n 10 'qstat -u "$USER"'
Detailed job info	qstat -f <job_id>
Show node allocation	qstat -n1 <job_id>
Show finished jobs	qstat -x -u "$USER"
Cancel job	qdel <job_id>
Show queues	qstat -Q
Detailed queue info	qstat -Qf
Show node info	pbsnodes -a
Example:
qsub train_model.pbs
qstat -u "$USER"
qstat -f 12345
qdel 12345


3. Recommended aliases

## References  

These are some nice references that talk about PBS commands.

- TODO external add references here. 

- TODO link to personal notes on useful PBS commands here.

## Job Scripts

**A canonical job script**: 

```bash
#PBS -N myexperiment
#PBS -l cpus=4
#PBS -l mem=16gb 
#PBS -l walltime=04:00:00
#PBS -m abe 
#PBS -M your.email@uts.edu.au
#PBS -j oe        # merge stderr and stdout into one log 

set -euo pipefail     # fail fast on errors




```


# Local Home layout 

Use a clean seperation between **code, environments, datasets, scratch outputs, and final results.**

```bash
$HOME/
  projects/
    
  venvs/
    rag-agents-py312/

  logs/
    # outputs from scripts  

  hf_hub/              # only if allowed and enough quota
  datasets/            # small metadata only; avoid huge corpora here if possible
```