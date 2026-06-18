# Model 

**UTS HPCC has:**

1. **Login Node** - SSH, edit files, install lightweight environments, submit jobs
2. **Execution/Job nodes** - Actual CPU/GPU computation
3. **Head node** - Scheduling management; not directly used
4. **Scratch Storage** - Large temporary experiment data, checkpoints, logs
5. **Home/shared Storage** - Code, config, small important files

> The current system has 14 execution nodes, most nodes have 64 cores, most have 754 GB RAM, two GPU nodes have Tesla V100 GPUs with 32 GB GPU memory each, and most nodes have fast local scratch storage.


**Filesystem layout:**

- `/shared/homes/u_student_id` - home directory, on the networked lsilon storage. Mounted on every node, but reads/writes go over the network.
- `/scratch/` - **local SSD on each compute node.**

Ideal workflow - stage data to /scratch at job start, do your I/O there, copy results back at job end, then clean up.
 - For ML jobs  , this matters, because if your DataLoader is hitting `/shared/homes/..` every batch, you are hammering the network ans slowing both yourself and every other job on the cluster.


TODO - How to setup ssh for easy login. 

TODO - How to setup local folder, bashrc, aliases, and scripts. 

TODO - How to setup Github ssh for projects.

TODO - How to setup VSCode for remote development.
  - How to connect Jupyter notebook to an interactive session for live debugging. 
  - https://researchcomputing.princeton.edu/support/knowledge-base/vs-code

# Using PBS Pro  

## Commands 

These are some nice references that talk about PBS commands.

- TODO external add references here. 

- TODO link to personal notes on useful PBS commands here.

```bash 
# TODO add more commands and their aliases if available
qstat -u $USER  # just your jobs.
qstat -fx <jobid> # full info on a finished ("expired") job, including actual mem/cpu/walltime used. Essential for right-sizing future jobs.

qdel <jobid> # cancel running job 
```

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
    rag-agents/
      src/
      scripts/
      configs/
      jobs/
      requirements.txt
      pyproject.toml
      README.md

  envs/
    rag-agents-py312/

  logs/
    rag-agents/

  results/
    rag-agents/
      final/

  models/              # only if allowed and enough quota
  datasets/            # small metadata only; avoid huge corpora here if possible

/scratch/$USER/
  rag-agents/
    datasets/
    models/
    indexes/
    runs/
    checkpoints/
    tmp/
```

# Setups local paths 

Setup local paths for libraries that are used very frequently like 