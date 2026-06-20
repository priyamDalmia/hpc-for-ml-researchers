# Setting up your Account 

## Mental Model of the HPC.

**UTS HPCC has Nodes:**

1. **Login Node**, `/shared/homes/uXXXXXX` — your home directory -- SSH - here you can edit files, install lightweight environments, submit jobs
2. **Execution/Job nodes** - Actual CPU/GPU computation
3. **Head node** - Scheduling management; not directly used

**Filesystem layout:**

4. **Scratch Storage** - local SSD on each compute node. Each node has its own; they are not shared. Reads and writes here are dramatically faster
- `/scratch/` - **local SSD on each compute node.**
6. **Home/shared Storage** - Code, config, small important files
   - `/shared/homes/u_student_id` - home directory, on the networked lsilon storage. Mounted on every node, but reads/writes go over the network.

## SSH

Set up SSH key-based login from your laptop to avoid retyping your password.

1. TODO Create secure keys and add On your laptop: ssh-keygen -t ed25519, then ssh-copy-id uXXXXXXX@<login-ip>.

```
chmod 700 ~/.ssh # TODO add comments 

ssh-keygen -t ed25519 -C "priyam.dalmia@student.uts.edu.au" # TODO Add comments 

chmod 600 ~/.ssh/hpc_ed25519 # protect private key 

ssh-copy-id -i ~/.ssh/hpc_ed25519.pub your_username@hpc.hostname.edu # TODO comments copies into this files ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

3. Add a `ssh/config` entry, protect the cofnig. 

```bash
```

**Using**

```
ssh hpc
scp file.txt hpc:~
rsync -avz project/ hpc:~/project/
```

## `bashrc`

See the TODO add link to the bashrc. 

```
# ~/.bashrc
# Minimal, safe, HPC-friendly Bash configuration

# ------------------------------------------------------------
# Helper: safely prepend directories to PATH without duplicates
# ------------------------------------------------------------

path_prepend() {
    for dir in "$@"; do
        [ -d "$dir" ] || continue
        case ":$PATH:" in
            *":$dir:"*) ;;
            *) PATH="$dir${PATH:+:$PATH}" ;;
        esac
    done
}

# User scripts and local binaries
path_prepend "$HOME/scripts" "$HOME/.local/bin" "$HOME/bin"
export PATH

# ------------------------------------------------------------
# Project-level environment variables
# Safe for interactive shells and batch jobs
# ------------------------------------------------------------

# Hugging Face caches: keep large model/data caches off $HOME
# export HF_HOME="$PROJECT_HOME/hf_cache"

export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Optional Python cache location
# export PYTHONPYCACHEPREFIX="$PROJECT_HOME/python_cache"

# ------------------------------------------------------------
# Stop here for non-interactive shells
# Important for HPC jobs, scp, rsync, and remote commands
# ------------------------------------------------------------

case $- in
    *i*) ;;
    *) return ;;
esac

# ------------------------------------------------------------
# Global Bash definitions
# ------------------------------------------------------------

# RHEL / CentOS / Rocky / AlmaLinux
[ -f /etc/bashrc ] && source /etc/bashrc

# Debian / Ubuntu
[ -f /etc/bash.bashrc ] && source /etc/bash.bashrc

# ------------------------------------------------------------
# History
# ------------------------------------------------------------

HISTCONTROL=ignoredups:ignorespace
HISTSIZE=5000
HISTFILESIZE=10000

shopt -s histappend
shopt -s checkwinsize

# Save history more reliably across terminals
PROMPT_COMMAND="history -a; history -n${PROMPT_COMMAND:+; $PROMPT_COMMAND}"

# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

# Cyan prompt:
# UTS-HPCC-user@node:path $
PS1='\[\033[1;34m\]CETUS-\u@\h:\w\[\033[0m\]$ '

# Alternative purple prompt:
# PS1='\[\033[35m\]UTS-HPCC-\u@\h:\w\[\033[0m\]$ '

# ------------------------------------------------------------
# Aliases
# ------------------------------------------------------------

alias ls='ls --color=auto'
alias ll='ls -lh --color=auto'
alias la='ls -lah --color=auto'
alias grep='grep --color=auto'
alias du1='du -h --max-depth=1'
alias q='squeue -u "$USER"'

# Load personal aliases if present
[ -f "$HOME/.bash_aliases" ] && source "$HOME/.bash_aliases"

# ------------------------------------------------------------
# Bash completion
# ------------------------------------------------------------

if [ -f /etc/bash_completion ] && ! shopt -oq posix; then
    source /etc/bash_completion
fi

# ------------------------------------------------------------
# Shell behaviour
# ------------------------------------------------------------

# Use vi-style command line editing
set -o vi

# Load readline settings if present
[ -f "$HOME/.inputrc" ] && bind -f "$HOME/.inputrc"
```

## More 


TODO - How to setup Github ssh for projects.

TODO - How to setup VSCode for remote development.
  - How to connect Jupyter notebook to an interactive session for live debugging. 
  - https://researchcomputing.princeton.edu/support/knowledge-base/vs-code

TODO - Hugging Face 

```
# Root Hugging Face cache/config directory
export HF_HOME="/path/to/cache_root"

# Hub cache: models, raw datasets, Spaces repos
export HF_HUB_CACHE="/path/to/hub_cache"

# Datasets cache: Arrow files, indices, processed datasets
export HF_DATASETS_CACHE="/path/to/datasets_cache"
```


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


# Local Home layout 

Use a clean seperation between **code, environments, datasets, scratch outputs, and final results.**

```bash
$HOME/
  projects/
    rag-agents/
      src/
      scripts/
```

# Python Virtual envs, shortcuts

1. Easily create python virtual envs
2. Activate (and print small message, python version, packages.)

#  