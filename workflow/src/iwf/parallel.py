"""Resource-aware parallel runner for the pipeline's embarrassingly-parallel loops.

Each candidate-dimension autoencoder (Phase 3 dim search) and each model seed (Phase 6/7/8
training) is fully independent, so they can run as separate processes. This module launches
them with a concurrency cap chosen so the CPU is not oversubscribed and the GPU does not run
out of memory, and it only admits new workers while the machine is below a load target (the
"don't start more once the PC is ~75% loaded" rule).

Opt in via the env (read at call time):
  PARALLEL_JOBS        unset / 1 / off / false  -> sequential (default, original code path)
                       auto                     -> pick a safe worker count from cores + VRAM
                       <int>=2..N               -> at most that many workers (still resource-capped)
  PARALLEL_LOAD_TARGET fraction in 0.1..0.95    -> CPU%/GPU% ceiling for admitting more (default 0.75)
  PARALLEL_THREADS_PER_WORKER  >0               -> torch threads per worker (default: auto = 2)

Safety: a single GPU serializes kernels across processes, so expect a solid (not linear)
speed-up; the win is largest for CPU-bound runs and the heavier autoencoder search. If the
parallel machinery raises for any reason, ``run_jobs`` finishes the remaining work
sequentially, so enabling this can never make the pipeline fail (worst case = no faster).
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

_FALSEY = {"", "0", "1", "off", "no", "false", "none", "seq", "sequential"}


def _jobs_setting() -> str:
    return (os.getenv("PARALLEL_JOBS", "1") or "1").strip().lower()


def parallel_enabled() -> bool:
    """True unless PARALLEL_JOBS selects the sequential path."""
    return _jobs_setting() not in _FALSEY


def load_target() -> float:
    try:
        v = float(os.getenv("PARALLEL_LOAD_TARGET", "0.75"))
    except (TypeError, ValueError):
        v = 0.75
    return min(max(v, 0.1), 0.95)


def _threads_setting() -> int:
    try:
        return max(0, int(os.getenv("PARALLEL_THREADS_PER_WORKER", "0")))
    except (TypeError, ValueError):
        return 0


def torch_cuda_available() -> bool:
    """Whether the installed torch build can use the GPU (False for a CPU-only wheel)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_stats() -> tuple[int, int, int, int] | None:
    """(free_MiB, used_MiB, total_MiB, util_pct) from nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.free,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        free, used, total, util = (int(x.strip()) for x in
                                   out.stdout.strip().splitlines()[0].split(","))
        return free, used, total, util
    except Exception:
        return None


def _cpu_percent(interval: float = 0.2) -> float | None:
    """Live CPU% via psutil if installed, else None (caller falls back to the worker cap)."""
    try:
        import psutil
        return float(psutil.cpu_percent(interval=interval))
    except Exception:
        return None


def plan_workers(n_tasks: int, *, use_cuda: bool, vram_per_worker_mib: int) -> tuple[int, int]:
    """Pick (max_workers, threads_per_worker) for ``n_tasks`` independent jobs."""
    cores = os.cpu_count() or 1
    target = load_target()
    threads = _threads_setting() or 2                       # small budget keeps CPU from thrashing

    setting = _jobs_setting()
    explicit = int(setting) if setting.isdigit() and int(setting) >= 2 else None

    cpu_workers = max(1, int(cores * target) // threads)    # total threads ~ target * cores
    cap = explicit if explicit is not None else cpu_workers  # explicit count is the ceiling
    cap = min(cap, n_tasks)

    if use_cuda:                                             # don't let processes OOM the GPU
        g = _gpu_stats()
        if g is not None:
            free_mib = g[0]
            gpu_workers = max(1, int(free_mib * 0.85) // max(1, vram_per_worker_mib))
            cap = min(cap, gpu_workers)

    cap = max(1, min(cap, n_tasks))
    if _threads_setting() == 0:                             # spread threads over the chosen workers
        threads = max(1, int(cores * target) // cap)
    return cap, threads


def _worker_init(threads: int) -> None:
    """Runs once in each worker process: bound its thread budget before any torch/BLAS work."""
    os.environ["WF_TORCH_THREADS"] = str(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))


def _admit(n_running: int, use_cuda: bool, target: float, vram_per_worker_mib: int) -> bool:
    """Whether to start one more worker now (always yes if none are running)."""
    if n_running == 0:
        return True
    cpu = _cpu_percent()
    if cpu is not None and cpu >= target * 100:
        return False
    if use_cuda:
        g = _gpu_stats()
        if g is not None:
            free_mib, _used, _total, util = g
            if free_mib < vram_per_worker_mib or util >= target * 100:
                return False
    return True


def run_jobs(worker, payloads, *, use_cuda: bool, vram_per_worker_mib: int = 1200,
             label: str = "jobs") -> list:
    """Run ``worker(payload)`` for each payload, resource-capped, results in input order.

    Sequential when parallelism is disabled or only one job. Any failure of the parallel
    machinery is caught and the remaining jobs are completed sequentially.
    """
    n = len(payloads)
    results: list = [None] * n
    if n == 0:
        return results
    if not parallel_enabled() or n == 1:
        for i, p in enumerate(payloads):
            results[i] = worker(p)
        return results

    cap, threads = plan_workers(n, use_cuda=use_cuda, vram_per_worker_mib=vram_per_worker_mib)
    if cap <= 1:
        for i, p in enumerate(payloads):
            results[i] = worker(p)
        return results

    target = load_target()
    print(f"[parallel] {label}: {n} jobs -> up to {cap} workers x {threads} threads"
          f"{' sharing 1 GPU' if use_cuda else ''}; admit while load < {int(target * 100)}%")

    done = [False] * n
    try:
        ex = ProcessPoolExecutor(max_workers=cap, initializer=_worker_init, initargs=(threads,))
        try:
            pending = list(range(n))
            running: dict = {}
            while pending or running:
                while pending and len(running) < cap and _admit(len(running), use_cuda,
                                                                 target, vram_per_worker_mib):
                    idx = pending.pop(0)
                    running[ex.submit(worker, payloads[idx])] = idx
                if not running and pending:                  # safety: keep making progress
                    idx = pending.pop(0)
                    running[ex.submit(worker, payloads[idx])] = idx
                if running:
                    finished, _ = wait(list(running), timeout=2.0,
                                       return_when=FIRST_COMPLETED)
                    for fut in finished:
                        i = running.pop(fut)
                        results[i] = fut.result()            # re-raises worker errors -> fallback
                        done[i] = True
        finally:
            ex.shutdown(wait=True)
    except Exception as e:  # never let parallelism break the run
        print(f"[parallel] {label}: parallel run failed ({e!r}); finishing the rest sequentially.")
        for i, p in enumerate(payloads):
            if not done[i]:
                results[i] = worker(p)
    return results
