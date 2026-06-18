"""
cost_ladder.py - watch the memory / compute cost of NN building blocks.

A "ladder" of models, smallest to largest:
    1. small MLP  : 3 linear layers, no sequence dimension
    2. large LSTM : sequential, scales with sequence length

For each rung we report, on the active device (CUDA > MPS > CPU):
    * parameter count, raw parameter memory, and theoretical forward FLOPs
    * forward-pass latency at three granularities -- a single sample, a full
      batch, and a sample-by-sample loop -- plus the throughput each implies
    * resident-set (RSS) and accelerator-allocator memory, logged event by
      event with the delta between *consecutive* events

The module only ever calls ``logging.getLogger(__name__)``; logging is assumed
to be configured by the caller. The ``__main__`` guard configures a basic
handler so the script is also runnable stand-alone.
"""

from __future__ import annotations

import gc
import logging
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)

try:
    import psutil

    _PROC = psutil.Process()
except ImportError:  # RSS reporting degrades gracefully if psutil is missing
    psutil = None
    _PROC = None
    _log.warning("psutil not found; RSS memory reporting will be unavailable.")


# ----------------------------------------------------------------------------
# units
# ----------------------------------------------------------------------------
_MIB = 1024**2


def _bytes_to_mib(n: float) -> float:
    return n / _MIB


# ----------------------------------------------------------------------------
# device selection: the snippet reused in every project
# ----------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
DTYPE = torch.float32  # MPS is happiest in fp32/bf16; CUDA can also do fp16/bf16
_ELEM_BYTES = torch.empty([], dtype=DTYPE).element_size()


# ----------------------------------------------------------------------------
# synchronisation, memory readers, cache management
# ----------------------------------------------------------------------------
def sync() -> None:
    """Block until queued accelerator work is finished (no-op on CPU)."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def rss_mib() -> float:
    return _bytes_to_mib(_PROC.memory_info().rss) if _PROC else float("nan")


def accel_current_mib() -> float:
    """Currently allocated GPU / unified memory, if the backend exposes it."""
    if DEVICE.type == "cuda":
        return _bytes_to_mib(torch.cuda.memory_allocated())
    if DEVICE.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return _bytes_to_mib(torch.mps.current_allocated_memory())
    return float("nan")


def accel_peak_mib() -> float:
    """Peak allocated accelerator memory. CUDA only; MPS has no stable counter."""
    if DEVICE.type == "cuda":
        return _bytes_to_mib(torch.cuda.max_memory_allocated())
    return float("nan")


def reset_accel_peak() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def free_caches() -> None:
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ----------------------------------------------------------------------------
# aligned-table formatting helpers
# ----------------------------------------------------------------------------
def _fmt(value, width: int, *, right: bool = True, nan: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and value != value):
        text = nan
    elif isinstance(value, float):
        text = f"{value:,.2f}"
    else:
        text = str(value)
    return text.rjust(width) if right else text.ljust(width)


def _fmt_delta(value, width: int, nan: str = "--") -> str:
    if value is None or value != value:  # None or NaN
        return nan.rjust(width)
    return f"{value:+,.2f}".rjust(width)


def _log_table(headers, rows, widths, aligns) -> None:
    """Log a header + separator + rows, every column right/left justified."""

    def cell(val, w, a):
        text = val if isinstance(val, str) else str(val)
        return text.rjust(w) if a == "r" else text.ljust(w)

    header = "  ".join(cell(h, w, a) for h, w, a in zip(headers, widths, aligns))
    _log.info(header)
    _log.info("-" * len(header))
    for row in rows:
        _log.info("  ".join(cell(c, w, a) for c, w, a in zip(row, widths, aligns)))


# ----------------------------------------------------------------------------
# memory tracker: logs RSS / accel memory with consecutive-event deltas
# ----------------------------------------------------------------------------
@dataclass
class MemTracker:
    """Mark named events; each row shows the step delta from the previous mark."""

    _COLS = (
        ("event", 42, False),
        ("rss_mib", 12, True),
        ("d_rss", 11, True),
        ("accel_mib", 12, True),
        ("d_accel", 11, True),
        ("peak_mib", 12, True),
    )

    _prev_rss: float | None = field(default=None, init=False)
    _prev_accel: float | None = field(default=None, init=False)
    _base_rss: float | None = field(default=None, init=False)
    _header_done: bool = field(default=False, init=False)

    def _header(self) -> None:
        cells = [_fmt(name, w, right=r) for name, w, r in self._COLS]
        line = "  ".join(cells)
        _log.info(line)
        _log.info("-" * len(line))
        self._header_done = True

    def mark(self, label: str) -> tuple[float, float]:
        if not self._header_done:
            self._header()

        rss = rss_mib()
        accel = accel_current_mib()
        peak = accel_peak_mib()

        d_rss = None if self._prev_rss is None else rss - self._prev_rss
        d_accel = None if self._prev_accel is None else accel - self._prev_accel

        widths = [c[1] for c in self._COLS]
        _log.info(
            "  ".join(
                [
                    _fmt(label, widths[0], right=False),
                    _fmt(rss, widths[1]),
                    _fmt_delta(d_rss, widths[2]),
                    _fmt(accel, widths[3]),
                    _fmt_delta(d_accel, widths[4]),
                    _fmt(peak, widths[5]),
                ]
            )
        )

        if self._base_rss is None:
            self._base_rss = rss
        self._prev_rss, self._prev_accel = rss, accel
        return rss, accel

    @property
    def base_rss(self) -> float:
        return self._base_rss if self._base_rss is not None else float("nan")


# ----------------------------------------------------------------------------
# timing helper
# ----------------------------------------------------------------------------
def timed(fn: Callable[[], object], iters: int = 20, warmup: int = 5) -> float:
    """Median wall-clock seconds for one call of ``fn`` (accelerator-synced)."""
    for _ in range(warmup):  # pay JIT / kernel-cache / allocator costs once
        fn()
    sync()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


# ----------------------------------------------------------------------------
# rung definition + benchmark
# ----------------------------------------------------------------------------
@dataclass
class Rung:
    name: str
    build: Callable[[], nn.Module]
    make_input: Callable[[int], torch.Tensor]  # batch_size -> input tensor
    flops_per_sample: int
    batch_size: int


def bench_rung(rung: Rung, loop_n: int = 32) -> dict:
    """Build, time, and profile memory for a single rung; return a result row."""
    free_caches()
    reset_accel_peak()
    mem = MemTracker()

    _log.info("")
    _log.info("=" * 80)
    _log.info("Rung: %s", rung.name)
    _log.info("=" * 80)

    base_rss, _ = mem.mark("before build")

    model = rung.build().to(DTYPE)
    n_params = sum(p.numel() for p in model.parameters())
    mem.mark("after build (cpu)")

    model = model.to(DEVICE).eval()
    mem.mark(f"after model -> {DEVICE.type}")

    x_batch = rung.make_input(rung.batch_size)
    _log.debug(
        "batch input: shape=%s dtype=%s raw=%.2f MiB",
        tuple(x_batch.shape),
        x_batch.dtype,
        _bytes_to_mib(x_batch.numel() * x_batch.element_size()),
    )
    mem.mark(f"after batch input (b={rung.batch_size}) -> {DEVICE.type}")

    # ---- latency at three granularities ----
    with torch.no_grad():
        x1 = rung.make_input(1)
        t_single = timed(lambda: model(x1))

        t_batch = timed(lambda: model(x_batch))

        loop_inputs = [rung.make_input(1) for _ in range(loop_n)]

        def run_loop():
            for xi in loop_inputs:
                model(xi)

        t_loop = timed(run_loop, iters=10, warmup=3)

    final_rss, _ = mem.mark("after inference")

    batch = rung.batch_size
    total_flops = rung.flops_per_sample * batch
    param_mib = _bytes_to_mib(n_params * _ELEM_BYTES)

    # ---- per-rung detail ----
    _log.info("")
    _log.info(f"parameters          : {n_params:,}  ({n_params / 1e6:.2f} M)")
    _log.info(f"param memory ({DTYPE}): {param_mib:,.2f} MiB")
    _log.info(
        f"forward FLOPs       : {total_flops / 1e9:.3f} GFLOP / batch "
        f"({rung.flops_per_sample / 1e9:.3f} GFLOP / sample, batch={batch})"
    )
    _log.info("")

    base_per_sample_ms = t_single * 1e3
    modes = [
        ("single (b=1)", t_single, 1),
        (f"batch (b={batch})", t_batch, batch),
        (f"loop x{loop_n}", t_loop, loop_n),
    ]
    lat_rows = []
    for label, total, count in modes:
        per_sample_ms = total / count * 1e3
        lat_rows.append(
            (
                label,
                f"{total * 1e3:.3f}",
                f"{per_sample_ms:.3f}",
                f"{count / total:.1f}",
                f"{base_per_sample_ms / per_sample_ms:.2f}x",
            )
        )
    _log_table(
        ("mode", "latency ms", "per-sample ms", "samples/s", "vs single"),
        lat_rows,
        (16, 12, 14, 12, 10),
        ("l", "r", "r", "r", "r"),
    )

    result = {
        "name": rung.name,
        "params_m": n_params / 1e6,
        "param_mib": param_mib,
        "fwd_gflop_batch": total_flops / 1e9,
        "single_ms": t_single * 1e3,
        "batch_ms": t_batch * 1e3,
        "batch_per_sample_ms": (t_batch / batch) * 1e3,
        "loop_per_sample_ms": (t_loop / loop_n) * 1e3,
        "gflops": (total_flops / t_batch) / 1e9,
        "rss_delta_mib": final_rss - base_rss,
        "accel_mib": accel_current_mib(),
        "accel_peak_mib": accel_peak_mib(),
    }

    del model
    return result


# ----------------------------------------------------------------------------
# the rungs  (sizes chosen so an Air won't choke; FLOPs computed by hand)
# ----------------------------------------------------------------------------
def mlp_rung() -> Rung:
    D_IN, H, D_OUT, B = 1024, 2048, 1024, 256

    def build() -> nn.Module:
        return nn.Sequential(
            nn.Linear(D_IN, H),
            nn.GELU(),
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, D_OUT),
        )

    def make_input(batch: int) -> torch.Tensor:
        return torch.randn(batch, D_IN, device=DEVICE, dtype=DTYPE)

    # Each linear is 2*in*out FLOPs / sample (one multiply + one add).
    # GELU and bias adds are negligible and ignored.
    flops_per_sample = 2 * (D_IN * H + H * H + H * D_OUT)
    return Rung("small MLP (3 linear)", build, make_input, flops_per_sample, B)


def lstm_rung() -> Rung:
    D_IN, H, LAYERS, T, B = 512, 1024, 2, 128, 64

    def build() -> nn.Module:
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(D_IN, H, LAYERS, batch_first=True)
                self.head = nn.Linear(H, D_IN)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out)

        return Net()

    def make_input(batch: int) -> torch.Tensor:
        return torch.randn(batch, T, D_IN, device=DEVICE, dtype=DTYPE)

    # Per layer, per timestep: 4 gates, each with W_ih @ x (in*H) and
    # W_hh @ h (H*H), times 2 for the multiply-add.  See arXiv:1409.2329.
    def layer_flops(in_size: int) -> int:
        return 2 * 4 * (in_size * H + H * H)

    per_step = layer_flops(D_IN) + layer_flops(H) * (LAYERS - 1)
    head_per_step = 2 * H * D_IN  # head runs on every timestep's output
    flops_per_sample = T * (per_step + head_per_step)
    return Rung("large LSTM (2 layers)", build, make_input, flops_per_sample, B)


# ----------------------------------------------------------------------------
# hardware / runtime report
# ----------------------------------------------------------------------------
def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def hardware_report() -> None:
    _log.info("=" * 80)
    _log.info("Hardware / Runtime")
    _log.info("=" * 80)

    fields = {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if platform.system() == "Darwin":
        fields["macOS"] = _run_cmd(["sw_vers", "-productVersion"])
        fields["chip"] = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    if psutil is not None:
        fields["cpu cores"] = (
            f"{psutil.cpu_count(logical=False)} physical "
            f"/ {psutil.cpu_count(logical=True)} logical"
        )
        vm = psutil.virtual_memory()
        fields["RAM total"] = f"{_bytes_to_mib(vm.total) / 1024:.2f} GiB"
        fields["RAM available"] = f"{_bytes_to_mib(vm.available) / 1024:.2f} GiB"

    fields["CUDA available"] = torch.cuda.is_available()
    fields["MPS available"] = torch.backends.mps.is_available()
    if torch.backends.mps.is_available():
        fields["accelerator"] = "Apple GPU via MPS"

    for key, value in fields.items():
        _log.info(f"{key:<20} = {value}")

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "PYTORCH_ENABLE_MPS_FALLBACK",
    ):
        _log.info(f"{key:<20} = {os.environ.get(key, 'unset')}")
    _log.info("=" * 80)


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def main(loop_n: int = 32) -> None:
    _log.info(
        f"device = {DEVICE.type}   dtype = {DTYPE}   "
        f"machine = {platform.machine()}   torch = {torch.__version__}"
    )
    hardware_report()

    rungs = [mlp_rung(), lstm_rung()]
    rows = [bench_rung(rung, loop_n=loop_n) for rung in rungs]

    _log.info("")
    _log.info("=" * 80)
    _log.info("Summary")
    _log.info("=" * 80)

    headers = (
        "model",
        "params(M)",
        "GFLOP",
        "batch ms",
        "b/samp ms",
        "1samp ms",
        "loop ms",
        "GFLOP/s",
        "dRSS MiB",
        "accel MiB",
        "peak MiB",
    )
    widths = (22, 10, 9, 10, 10, 10, 10, 9, 10, 10, 10)
    aligns = ("l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r")

    table_rows = []
    for r in rows:
        table_rows.append(
            (
                r["name"],
                f"{r['params_m']:.2f}",
                f"{r['fwd_gflop_batch']:.3f}",
                f"{r['batch_ms']:.2f}",
                f"{r['batch_per_sample_ms']:.3f}",
                f"{r['single_ms']:.3f}",
                f"{r['loop_per_sample_ms']:.3f}",
                f"{r['gflops']:.1f}",
                f"{r['rss_delta_mib']:.1f}",
                _fmt(r["accel_mib"], 10).strip(),
                _fmt(r["accel_peak_mib"], 10).strip(),
            )
        )
    _log_table(headers, table_rows, widths, aligns)

    _log.info("")
    _log.info("Notes:")
    _log.info(
        " * GFLOP/s vs device peak = MFU; well below ~30-50% means memory/IO bound."
    )
    _log.info(" * b/samp ms (batched per-sample) << 1samp ms is the batching win.")
    _log.info(" * loop ms is the per-sample cost of running samples one at a time.")
    _log.info(" * accel cols are MPS/CUDA allocator usage; n/a on CPU (see dRSS).")
    _log.info(
        " * peak accel memory is CUDA-only; MPS exposes no reliable peak counter."
    )
    _log.info(" * train FLOPs ~= 3x forward (backward is ~2x forward).")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
