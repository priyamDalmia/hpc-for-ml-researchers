# hpc-for-ml-researchers
Practical guide and reproducible scripts for running large-scale ML, RL, and LLM experiments on PBS Pro HPC clusters. Worked examples target the UTS eResearch HPCC but generalise to most academic HPC.

# hpc-for-ml-researchers

A practical guide and reproducible scripts for running machine learning,
reinforcement learning, and LLM experiments on PBS Pro HPC clusters.

Worked examples target the [UTS eResearch HPCC](https://hpc.research.uts.edu.au),
but most of what's here generalises to any PBS Pro cluster (including NCI Gadi)
and much of it adapts straightforwardly to Slurm.

## Who this is for

ML/RL/LLM researchers — typically PhD students — who:

- have access to a university HPC cluster and want to use it well, not just *use* it
- come from a software/data-science background rather than a traditional HPC one
- want their experiments to be reproducible by default, not as an afterthought

This is the document I wish I'd had on day one.

## What's here

This repo is a work in progress and will grow as I (and hopefully others) hit
new problems and write up the solutions. Current contents:

- `guide/` — written explainers, organised by topic
- `templates/` — PBS job script templates you can copy and adapt
- `scripts/` — utility scripts (resource auditing, environment setup, etc.)
- `examples/` — end-to-end worked examples of common workflows

See [`guide/00-start-here.md`](guide/00-start-here.md) for a recommended reading order.

## Status

Early days. Expect things to move around. If you find something useful,
star the repo so others can find it. If you find something wrong or missing,
open an issue or PR — contributions from other UTS students and researchers
elsewhere are very welcome.

## Acknowledgements

Built on top of the excellent official documentation at
<https://hpc.research.uts.edu.au>. This repo is a complement, not a replacement —
when in doubt, the official docs are authoritative.

## Licence

MIT for code, CC-BY-4.0 for written guides.
