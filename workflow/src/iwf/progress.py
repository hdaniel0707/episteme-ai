"""Live multi-workflow terminal progress display (plan section 6.3).

``MultiWorkflowProgress`` shows one fixed-width line per workflow, redrawn ~once a second from a
background daemon thread that drains a queue of structured updates. It uses ANSI cursor control
on a capable TTY and falls back to plain line-by-line logging when output is piped, ``NO_COLOR``
is set, or the terminal cannot encode the box characters. Standard library only (no rich/tqdm).

Usage::

    prog = MultiWorkflowProgress(["emb_reg", "tfm_reg"])
    prog.start()
    prog.update("emb_reg", phase=(7, 10), percent=40, seed=(2, 5),
                metric=("val_spearman", 0.412))
    prog.done("emb_reg", metric=("best_spearman", 0.431))
    prog.error("tfm_reg", "CUDA OOM")
    prog.stop()
"""

from __future__ import annotations

import os
import queue
import shutil
import sys
import threading
import time

# Canonical W1..W6 numbering for the labels (plan section 1).
WF_NUMBER = {
    "emb_clf": "W1", "emb_reg": "W2", "emb_meta_clf": "W3",
    "emb_meta_reg": "W4", "tfm_clf": "W5", "tfm_reg": "W6",
}


def _eta_str(seconds: float) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "--"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


class _WFState:
    __slots__ = ("label", "status", "phase", "phase_total", "percent", "seed", "n_runs",
                 "metric_name", "metric_val", "start_time", "end_time", "error_msg", "_last_plain")

    def __init__(self, label: str):
        self.label = label
        self.status = "pending"           # pending | running | done | error
        self.phase = 0
        self.phase_total = 10
        self.percent = 0.0
        self.seed = 0
        self.n_runs = 0
        self.metric_name = ""
        self.metric_val = None
        self.start_time = None
        self.end_time = None
        self.error_msg = ""
        self._last_plain = ""


class MultiWorkflowProgress:
    """Background, queue-driven live progress for several workflows on one terminal."""

    REDRAW_EVERY = 1.0   # seconds

    def __init__(self, workflow_ids, labels: dict | None = None, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._order = list(workflow_ids)
        self.states: dict[str, _WFState] = {}
        for wf in self._order:
            lbl = (labels or {}).get(wf) or f"{WF_NUMBER.get(wf, '  ')} {wf}"
            self.states[wf] = _WFState(self._fixed_label(lbl))
        self._ansi = self._supports_ansi()
        self._unicode = self._supports_unicode()
        self._drawn_lines = 0

    # ----------------------------------------------------------------- capability detection
    def _supports_ansi(self) -> bool:
        if os.getenv("NO_COLOR") is not None:
            return False
        if os.getenv("TERM") == "dumb":
            return False
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if not is_tty:
            return False
        if sys.platform == "win32":
            return self._enable_windows_vt()
        return True

    def _enable_windows_vt(self) -> bool:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)            # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VT_PROCESSING
            return True
        except Exception:
            return False

    def _supports_unicode(self) -> bool:
        enc = (getattr(self.stream, "encoding", None) or "").lower()
        return "utf" in enc

    def _fixed_label(self, label: str) -> str:
        label = label[:12]
        return f"{label:<12}"

    # ----------------------------------------------------------------- public API
    def start(self) -> "MultiWorkflowProgress":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="wf-progress", daemon=True)
        self._thread.start()
        return self

    def update(self, wf_id: str, *, phase=None, percent=None, seed=None,
               metric=None, status="running") -> None:
        msg = {"wf_id": wf_id, "status": status}
        if phase is not None:
            msg["phase"] = phase
        if percent is not None:
            msg["percent"] = percent
        if seed is not None:
            msg["seed"] = seed
        if metric is not None:
            msg["metric"] = metric
        self._q.put(msg)

    def done(self, wf_id: str, *, metric=None) -> None:
        self._q.put({"wf_id": wf_id, "status": "done", "metric": metric})

    def error(self, wf_id: str, message: str) -> None:
        self._q.put({"wf_id": wf_id, "status": "error", "error_msg": str(message)})

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._drain()
        self._render(final=True)

    def __enter__(self) -> "MultiWorkflowProgress":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ----------------------------------------------------------------- worker thread
    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain()
            self._render()
            time.sleep(self.REDRAW_EVERY)

    def _drain(self) -> None:
        while True:
            try:
                msg = self._q.get_nowait()
            except queue.Empty:
                return
            self._apply(msg)

    def _apply(self, msg: dict) -> None:
        wf = msg.get("wf_id")
        st = self.states.get(wf)
        if st is None:
            return
        with self._lock:
            status = msg.get("status", st.status)
            if status == "running" and st.status in ("pending",):
                st.start_time = time.time()
            st.status = status
            if "phase" in msg and msg["phase"] is not None:
                st.phase, st.phase_total = msg["phase"]
            if "percent" in msg and msg["percent"] is not None:
                st.percent = float(msg["percent"])
            if "seed" in msg and msg["seed"] is not None:
                st.seed, st.n_runs = msg["seed"]
            if msg.get("metric") is not None:
                st.metric_name, st.metric_val = msg["metric"]
            if status in ("done", "error") and st.end_time is None:
                st.end_time = time.time()
            if status == "error":
                st.error_msg = msg.get("error_msg", "")

    # ----------------------------------------------------------------- rendering
    def _bar(self, percent: float, width: int = 20) -> str:
        filled = int(round(width * max(0.0, min(100.0, percent)) / 100.0))
        if self._unicode:
            return "█" * filled + "░" * (width - filled)
        return "#" * filled + "-" * (width - filled)

    def _mark(self, ok: bool) -> str:
        if self._unicode:
            return "✓" if ok else "✗"
        return "OK" if ok else "X"

    def _line(self, st: _WFState) -> str:
        if st.status == "done":
            elapsed = (st.end_time or time.time()) - (st.start_time or st.end_time or time.time())
            metric = (f"best_{st.metric_name.split('_')[-1]}={st.metric_val:.3f}"
                      if st.metric_val is not None else "")
            line = f"[{st.label}] {self._mark(True)} DONE | {metric} | {_eta_str(elapsed)}"
        elif st.status == "error":
            line = f"[{st.label}] {self._mark(False)} ERROR: {st.error_msg[:60]}"
        else:
            bar = self._bar(st.percent)
            phase = f"Phase {st.phase}/{st.phase_total}"
            seed = f"seed {st.seed}/{st.n_runs}" if st.n_runs else "seed -/-"
            metric = (f"{st.metric_name}={st.metric_val:.3f}"
                      if (st.metric_name and st.metric_val is not None) else "")
            eta = "--"
            if st.start_time and st.percent > 0:
                elapsed = time.time() - st.start_time
                eta = _eta_str(elapsed * (100.0 - st.percent) / max(1e-6, st.percent))
            line = (f"[{st.label}] [{phase:>10}] [{bar}] {st.percent:3.0f}% | "
                    f"{seed:<10} | {metric:<20} | ETA {eta}")
        # Truncate to terminal width to prevent wrapping that breaks ANSI cursor math
        width = shutil.get_terminal_size((120, 30)).columns
        return (line[:width-1] + "…") if len(line) > width else line

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
        except UnicodeEncodeError:                     # console can't encode the glyphs -> ASCII
            self.stream.write(text.encode("ascii", "replace").decode("ascii"))

    def _render(self, final: bool = False) -> None:
        with self._lock:
            lines = [self._line(self.states[wf]) for wf in self._order]
        if self._ansi:
            buf = ""
            if self._drawn_lines:
                buf += f"\x1b[{self._drawn_lines}A"     # cursor up to the first line
            for ln in lines:
                buf += "\r\x1b[2K" + ln + "\n"          # clear line, write, newline
            self._write(buf)
            self._drawn_lines = len(lines)
            self._flush()
        else:
            # Plain fallback: only print a line when its content changed (avoids log spam).
            for wf in self._order:
                st = self.states[wf]
                ln = self._line(st)
                if ln != st._last_plain:
                    st._last_plain = ln
                    self._write(ln + "\n")
            self._flush()

    def _flush(self) -> None:
        try:
            self.stream.flush()
        except Exception:
            pass
