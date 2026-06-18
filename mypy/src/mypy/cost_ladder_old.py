"""
cost_ladder.py - watch the memory/compute cost of NN building blocks in real time.

A "ladder" of models, smallest to largest:
    1. small MLP: 1 hidden layer, 128 units
    2. large LSTM: seqeuential; time scaling
    3. single-head attention: quadratic scaling with sequence length

"""

import sys
import time, statistics, platform, gc
import torch
import torch.nn as nn

import logging

_log = logging.getLogger(__name__)

try:
    import psutil

    _PROC = psutil.Process()
    _log.info(
        f"Running on {platform.system()} {platform.release()} with {psutil.virtual_memory().total / 1e9:.1f} GB RAM"
    )

except ImportError:
    _log.warning("psutil not found; system monitoring will be unavailable.")
    _PROC = None

# MacOS HArdware report
import platform
import subprocess
import psutil
import torch
import os


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"


def mac_hardware_report():
    _log.info("=" * 80)
    _log.info("Hardware / Runtime")
    _log.info("=" * 80)

    # Basic OS / Python
    _log.info(f"machine              = {platform.machine()}")
    _log.info(f"processor            = {platform.processor()}")
    _log.info(f"platform             = {platform.platform()}")
    _log.info(f"python               = {platform.python_version()}")
    _log.info(f"torch                = {torch.__version__}")

    # macOS version
    macos_version = run_cmd(["sw_vers", "-productVersion"])
    _log.info(f"macOS                = {macos_version}")

    # Chip name
    chip = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    _log.info(f"chip                 = {chip}")

    # CPU cores
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    _log.info(f"CPU cores            = {physical} physical / {logical} logical")

    # Memory
    vm = psutil.virtual_memory()
    _log.info(f"RAM total            = {vm.total / 1024**3:.2f} GiB")
    _log.info(f"RAM available        = {vm.available / 1024**3:.2f} GiB")

    # PyTorch devices
    _log.info(f"CUDA available       = {torch.cuda.is_available()}")
    _log.info(f"MPS available        = {torch.backends.mps.is_available()}")
    _log.info(f"MPS built            = {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        _log.info("accelerator          = Apple GPU via MPS")
        _log.info(
            f"MPS allocated        = {torch.mps.current_allocated_memory() / 1024**2:.2f} MB"
        )
        _log.info(
            f"MPS driver allocated = {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB"
        )

    # Optional: environment variables that affect threading/perf
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "PYTORCH_ENABLE_MPS_FALLBACK",
    ]:
        _log.info(f"{key:<21}= {os.environ.get(key, 'unset')}")

    _log.info("=" * 80)


# ----------------------------------------------------------------------------
# device selection: this is exactly the snippet you reuse in every project
# ----------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
DTYPE = torch.float32  # MPS is happiest in fp32/bf16; CUDA can do bf16/fp16


# ----------------------------------------------------------------------------
# memory + timing helpers
# ----------------------------------------------------------------------------
def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def rss_mb():
    return _PROC.memory_info().rss / 1e6 if _PROC else float("nan")


def accel_mem_mb():
    """GPU/unified-memory allocator usage, if the backend exposes it."""
    if DEVICE.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if DEVICE.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return float("nan")


def tensor_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 1024**2


def log_tensor_size(name: str, t: torch.Tensor):
    _log.debug(
        f"{name}: shape={tuple(t.shape)}, "
        f"dtype={t.dtype}, "
        f"numel={t.numel():,}, "
        f"element_size={t.element_size()} B, "
        f"raw_size={tensor_mb(t):.2f} MiB"
    )


def reset_accel_peak():
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def log_mem(label: str, base_rss: float | None = None):
    rss = rss_mb()
    accel = accel_mem_mb()

    # Column widths
    w = {
        "label": 56,
        "rss": 12,
        "delta": 12,
        "accel": 14,
    }

    def fmt_mb(x: float | None) -> str:
        if x is None:
            return "n/a"
        if x != x:  # NaN check
            return "n/a"
        return f"{x:,.2f}"

    def line(label, rss, delta, accel):
        return (
            f"{str(label).ljust(w['label'])}  "
            f"{str(rss).rjust(w['rss'])}  "
            f"{str(delta).rjust(w['delta'])}  "
            f"{str(accel).rjust(w['accel'])}"
        )

    # Print header once
    if not getattr(log_mem, "_header_printed", False):
        hdr = line("event", "rss_mb", "delta_rss", "accel_mb")
        sep = "-" * len(hdr)
        _log.debug(hdr)
        _log.debug(sep)
        log_mem._header_printed = True

    delta_rss = None if base_rss is None else rss - base_rss

    _log.debug(
        line(
            label,
            fmt_mb(rss),
            "n/a" if delta_rss is None else f"{delta_rss:+,.2f}",
            fmt_mb(accel),
        )
    )

    return rss, accel


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):  # warmup pays JIT/kernel-cache/allocator costs once
        fn()
        sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def measure(name, build, make_input, fwd_flops, train=True):
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    # elif DEVICE.type == "mps":
    #     torch.mps.empty_cache()

    base_rss, base_accel = log_mem(f"Before building {name}")
    reset_accel_peak()

    model = build().to(DTYPE)
    nparam = sum(p.numel() for p in model.parameters())

    cpu_model_rss, cpu_model_accel = log_mem(f"After building {name}", base_rss)

    # ---- inference (no autograd) ----
    x, y = make_input()
    log_tensor_size("input x", x)
    cpu_input_rss, cpu_input_accel = log_mem(
        f"After making input on {x.device}", base_rss
    )

    # --- move model to accelerator if possible ---
    model = model.to(DEVICE)
    model.eval()
    mps_model_rss, mps_model_accel = log_mem(
        f"After moving model to {DEVICE}",
        base_rss,
    )

    # ---- move inputs to accelerator if needed ----
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    mps_input_rss, mps_input_accel = log_mem(
        f"After moving inputs to {DEVICE}",
        base_rss,
    )

    # ---- inference ----
    model.eval()

    with torch.no_grad():
        fwd_t = timed(lambda: model(x))

    infer_rss, infer_accel = log_mem(
        "After one inference pass",
        base_rss,
    )

    _log.debug(
        f"Expected raw parameter memory: "
        f"{nparam * torch.empty([], dtype=DTYPE).element_size() / 1024**2:.2f} MB"
    )
    _log.debug(
        f"One inference pass changed RSS by "
        f"{infer_rss - mps_input_rss:+.2f} MB and accelerator memory by "
        f"{infer_accel - mps_input_accel:+.2f} MB"
    )

    row = {
        "name": name,
        "params": nparam,
        "fwd_gflop": fwd_flops / 1e9,
        "fwd_ms": fwd_t * 1e3,
        "fwd_gflops": (fwd_flops / fwd_t) / 1e9,
        "base_rss": base_rss,
        "rss_mb": infer_rss - base_rss,
        "accel_mb": infer_accel,
    }

    del model
    return row


# ----------------------------------------------------------------------------
# the four rungs  (params chosen so an Air won't choke; FLOPs computed by hand)
# ----------------------------------------------------------------------------
def mlp_rung():
    B, D_IN, H, D_OUT = 256, 1024, 2048, 1024

    def build():
        return nn.Sequential(
            nn.Linear(D_IN, H),
            nn.GELU(),
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, D_OUT),
        )

    def inp():
        return (
            torch.randn(B, D_IN, device=DEVICE, dtype=DTYPE),
            torch.randn(B, D_OUT, device=DEVICE, dtype=DTYPE),
        )

    # linear matmul = 2*B*in*out  (mul+add)
    flops = 2 * B * (D_IN * H + H * H + H * D_OUT)
    inp()
    return measure("small MLP (3 linear)", build, inp, flops)


def lstm_rung():
    B, T, D_IN, H, LYR = 64, 128, 512, 1024, 2

    def build():
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(D_IN, H, LYR, batch_first=True)
                self.head = nn.Linear(H, D_IN)

            def forward(self, x):
                o, _ = self.lstm(x)
                return self.head(o)

        return M()

    def inp():
        return (
            torch.randn(B, T, D_IN, device=DEVICE, dtype=DTYPE),
            torch.randn(B, T, D_IN, device=DEVICE, dtype=DTYPE),
        )

    # LSTM FLOPs are a bit more complex; see e.g. https://arxiv.org/pdf/1409.2329.pdf
    # For each of the 4 gates, we have:
    #   - input matmul: 2*(in+H)*H, per timespte, per layer, per batch
    per_layer = lambda i: 2 * 4 * (i + H) * H
    flops = B * T * (per_layer(D_IN) + per_layer(H) * (LYR - 1)) + 2 * B * T * H * D_IN
    return measure("large LSTM (2 layers)", build, inp, flops)


def main():
    _log.info(
        f"device = {DEVICE.type}   dtype = {DTYPE}   "
        f"machine = {platform.machine()}   torch = {torch.__version__}"
    )
    mac_hardware_report()
    rows = [mlp_rung(), lstm_rung()]

    hdr = (
        "model",
        "params",
        "fwd GFLOP",
        "fwd ms",
        "GFLOP/s",
        "train ms",
        "ΔRSS MB",
        "accel MB",
    )
    w = (24, 12, 11, 9, 9, 10, 9, 9)
    line = lambda c: "  ".join(str(x).ljust(wi) for x, wi in zip(c, w))
    _log.info(line(hdr))
    _log.info("-" * sum(w) + "-" * 2 * len(w))
    for r in rows:
        _log.info(
            line(
                (
                    r["name"],
                    f"{r['params'] / 1e6:.2f}M",
                    f"{r['fwd_gflop']:.3f}",
                    f"{r['fwd_ms']:.2f}",
                    f"{r['fwd_gflops']:.1f}",
                    f"{r.get('train_ms', float('nan')):.2f}",
                    f"{r['rss_mb']:.0f}",
                    f"{r['accel_mb']:.0f}" if r["accel_mb"] == r["accel_mb"] else "n/a",
                )
            )
        )
    _log.info("Notes:")
    _log.info(
        " * GFLOP/s vs your device peak = MFU. Far below ~30-50%? You're memory/IO bound."
    )
    _log.info(" * train GFLOP ~= 3x fwd (backward is ~2x forward).")
    _log.info(
        " * accel MB is MPS/CUDA allocator; on CPU it's n/a (memory shows in ΔRSS)."
    )
