# Python 

Python execuatblse live under `/urs/bin/`. Versions aviable are 
- `python3.8`
- `python3.12`

- TODO create python and pip shortcuts  
```
mkdir -p ~/.local/bin

ln -sf /usr/bin/python3.12 ~/.local/bin/python
ln -sf /usr/bin/python3.12 ~/.local/bin/python3
```

## Create a virtualenv 

1. NOTE - Create a common place for all your venvs;for best reporducble  ML experimnets, clone the repo on every run;

2. use `uv` to manage your environments
   1. install `uv` with - curl -LsSf https://astral.sh/uv/install.sh | sh

3. create new envs with `uv`
   1. uv venv ~/venvs/ml-312 --python ~/.local/bin/python 
   2. uv python find # to check the current version

TODO write alias to autmatically acitvate env with `activate env_name` 

```
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
