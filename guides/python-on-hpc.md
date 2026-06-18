# Python 

Unlike other HPC, UTS HPC is quite old and dosen't have a comprehesenve list of modules (see unimelb module list for example, thousands of modules avaiable, and they dont even have a technology in their name)

Python execuatblse live under `/urs/bin/`. Versions aviable are 
- `python3.8`
- `python3.12`

## Create a virtualenv 

1. NOTE - Create a common place for all your venvs; I can be a little tedious to manage than simply having all your .venv in the local project folder, but for best reporducble  ML experimnets, you wil be cloning the repo on every run; hence install all packages will not make sense. 

`python3.12 -m venv --prompt py3.13` 

then activate 

`activate` 

```
# One-time: pick a place for your environments
mkdir -p ~/venvs

# Create a project environment
/usr/bin/python3.12 -m venv ~/venvs/research --prompt="research"

# Activate
source ~/venvs/research/bin/activate

# Always upgrade pip first — system pip is usually ancient
python -m pip install --upgrade pip setuptools wheel

# Install what you need
pip install numpy pandas scikit-learn matplotlib
pip install torch torchvision           # ships its own CUDA libs
pip install transformers datasets accelerate
pip install gymnasium "stable-baselines3[extra]"

# Freeze the exact versions for reproducibility
pip freeze > ~/projects/myresearch/requirements.lock.txt
```

Inside a PBS job:
bashsource "${HOME}/venvs/research/bin/activate"
python train.py
