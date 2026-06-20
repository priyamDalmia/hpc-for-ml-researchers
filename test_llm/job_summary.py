#!/usr/bin/env python3
"""job_summary.py — introspect what Python sees inside a PBS job."""

from __future__ import annotations

import time
import argparse
import importlib
import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path
from textwrap import indent


def section(title: str) -> None:
    print(f"\n---------- {title} ----------")


def kv(label: str, value) -> None:
    print(f"  {label:<22}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    # -- Python / interpreter -------------------------------------------------
    section("PYTHON INTERPRETER")
    kv("executable", sys.executable)
    kv("version", sys.version.replace("\n", " "))
    kv("implementation", platform.python_implementation())
    kv("prefix", sys.prefix)
    kv("base_prefix", sys.base_prefix)
    kv("in_venv", sys.prefix != sys.base_prefix)
    kv("sys.path entries", len(sys.path))
    for p in sys.path:
        print(f"      - {p}")

    # -- Platform / OS --------------------------------------------------------
    section("PLATFORM")
    kv("hostname", socket.gethostname())
    kv("FQDN", socket.getfqdn())
    kv("platform", platform.platform())
    kv("machine", platform.machine())
    kv("processor", platform.processor() or "<not reported>")
    kv("python compiler", platform.python_compiler())

    # -- CPU visibility -------------------------------------------------------
    # IMPORTANT distinction on HPC:
    #   os.cpu_count()                    -> total CPUs on the node
    #   len(os.sched_getaffinity(0))      -> CPUs THIS process is allowed to use
    # When PBS gives you ncpus=2 on a 64-core node, the first is 64, the
    # second is 2. Use the second to size thread pools (numpy, torch, joblib).
    section("CPU VISIBILITY")
    total = os.cpu_count()
    try:
        allocated = len(os.sched_getaffinity(0))
    except AttributeError:
        allocated = "<sched_getaffinity not available on this OS>"
    kv("os.cpu_count() (node)", total)
    kv("sched_getaffinity (job)", allocated)
    kv("$NCPUS (PBS)", os.environ.get("NCPUS", "<not set>"))

    # -- Paths ----------------------------------------------------------------
    section("PATHS")
    kv("cwd", Path.cwd())
    kv("home", Path.home())

    # -- Environment variables (PBS + a few common ones) ----------------------
    section("KEY ENVIRONMENT VARIABLES")
    interesting_prefixes = ("PBS_", "CUDA", "OMP_", "MKL_", "NUMBA_", "HF_", "TORCH_")
    interesting_exact = {"USER", "HOME", "SHELL", "LANG", "VIRTUAL_ENV", "CONDA_PREFIX"}
    keys = sorted(
        k for k in os.environ
        if k in interesting_exact or k.startswith(interesting_prefixes)
    )
    for k in keys:
        # PATH-like values get truncated for readability
        v = os.environ[k]
        if len(v) > 120:
            v = v[:117] + "..."
        kv(k, v)

    # -- Library versions (optional, fail soft) -------------------------------
    # We try-import a handful so this script works on any environment.
    section("LIBRARY VERSIONS (best effort)")
    candidates = [
        "numpy", "scipy", "pandas", "sklearn", "matplotlib",
        "torch", "torchvision", "jax", "tensorflow", "transformers",
        "datasets", "accelerate", "gymnasium", "stable_baselines3",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            kv(name, getattr(mod, "__version__", "<unknown>"))
        except Exception:
            pass  # not installed — skip silently

    # -- GPU / CUDA via PyTorch if available ----------------------------------
    section("GPU / CUDA (via PyTorch if present)")
    try:
        import torch
        kv("torch.__version__", torch.__version__)
        kv("torch.cuda.is_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            kv("device count", torch.cuda.device_count())
            kv("cuda runtime", torch.version.cuda)
            kv("cudnn", torch.backends.cudnn.version())
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"      [{i}] {props.name}  "
                      f"{props.total_memory / 1024**3:.1f} GiB  "
                      f"compute {props.major}.{props.minor}")
        else:
            kv("note", "no CUDA visible — fine if this isn't a GPU job")
    except ImportError:
        kv("torch", "not installed")

    # -- Demonstrate I/O on scratch ------------------------------------------

    # parser.add_argument(
    #     "--scratch", type=Path, required=False,
    #    help="Scratch directory created by the shell job script.",
    # )
    # args = parser.parse_args()

    # section("SCRATCH I/O DEMO")
    # kv("scratch (arg)", args.scratch)
    # kv("scratch exists", args.scratch.is_dir())
    # kv("PATH entries", len(os.environ.get("PATH", "").split(":")))

    # report_path = args.scratch / "python_report.txt"
    # payload = {
    #     "hostname": socket.gethostname(),
    #     "pid": os.getpid(),
    #     "python": sys.version.split()[0],
    #     "cwd": str(Path.cwd()),
    #     "scratch": str(args.scratch),
    #     "pbs_jobid": os.environ.get("PBS_JOBID"),
    #     "allocated_cpus": allocated,
    # }
    # report_path.write_text(json.dumps(payload, indent=2) + "\n")
    # kv("wrote", report_path)
    # kv("size (bytes)", report_path.stat().st_size)

    # # Show disk usage of scratch for fun
    # usage = shutil.disk_usage(args.scratch)
    # kv("scratch free GiB", f"{usage.free / 1024**3:.1f}")



if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nSummary finished in {(t0 - time.perf_counter()):.3f}s.")
