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


# The currently-running MultiWorkflowProgress (set by start(), cleared by stop()). The
# tqdm shim below forwards fine-grained sub-task bars (e.g. the autoencoder "AE(k=…)"
# loop) to this instance's footer line instead of printing them straight to the terminal,
# where they would collide with the per-workflow block.
_ACTIVE: "MultiWorkflowProgress | None" = None


def active_progress() -> "MultiWorkflowProgress | None":
    return _ACTIVE


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
        self.stream = stream if stream is not None else self._default_stream()
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
        self._subtask = ""                     # current fine-grained sub-task (footer line)
        # Optional live status file: rewritten in full every render so a plain file always
        # holds exactly one current line per workflow (+ footer) with growing bars — the
        # "each line only once, always up to date" log the append-only stream can't give.
        self._status_path = os.getenv("PROGRESS_STATUS_FILE") or None

    # ----------------------------------------------------------------- capability detection
    @staticmethod
    def _env_flag(name: str) -> bool | None:
        """Tri-state env flag: True/False for recognised on/off values, None when unset."""
        v = os.getenv(name)
        if v is None:
            return None
        v = v.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        return None

    def _default_stream(self):
        """Prefer whichever of stdout/stderr is an interactive TTY, mirroring tqdm.

        When the pipeline's stdout is piped through ``tee`` / ``Tee-Object`` (as the deploy
        scripts do) stdout is no longer a TTY, but stderr often still is — rendering there
        keeps the live in-place refresh working on the console."""
        out, err = sys.stdout, sys.stderr
        if not bool(getattr(out, "isatty", lambda: False)()) \
                and bool(getattr(err, "isatty", lambda: False)()):
            return err
        return out

    def _supports_ansi(self) -> bool:
        if os.getenv("NO_COLOR") is not None:
            return False
        if os.getenv("TERM") == "dumb":
            return False
        # Explicit override: FORCE_PROGRESS_ANSI=1 forces in-place rendering even when the
        # stream isn't a detected TTY (e.g. behind `tee`, which forwards the escape codes to
        # the real console). FORCE_PROGRESS_ANSI=0 forces the plain fallback.
        forced = self._env_flag("FORCE_PROGRESS_ANSI")
        if forced is False:
            return False
        if forced is True:
            if sys.platform == "win32":
                self._enable_windows_vt()   # best-effort; force anyway
            return True
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
        global _ACTIVE
        _ACTIVE = self
        self._thread = threading.Thread(target=self._run, name="wf-progress", daemon=True)
        self._thread.start()
        return self

    def set_subtask(self, text: str) -> None:
        """Set the fine-grained sub-task shown on the footer line (below the block)."""
        with self._lock:
            self._subtask = str(text)

    def clear_subtask(self) -> None:
        with self._lock:
            self._subtask = ""

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
        global _ACTIVE
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            self._subtask = ""             # no sub-task once the block is finalised
        self._drain()
        self._render(final=True)
        if _ACTIVE is self:
            _ACTIVE = None

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

    def _footer_line(self, subtask: str) -> str:
        """The sub-task footer (the AE bar etc.) rendered below the workflow block."""
        prefix = "  └─ " if self._unicode else "  |- "
        line = prefix + subtask
        width = shutil.get_terminal_size((120, 30)).columns
        return (line[:width - 1] + "…") if len(line) > width else line

    def _write_status_file(self, block: list[str]) -> None:
        """Rewrite the whole status file (truncate + write) so it always contains exactly
        the current block — one line per workflow plus the footer, each appearing once."""
        if not self._status_path:
            return
        try:
            with open(self._status_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(block) + "\n")
        except OSError:
            pass

    def _render(self, final: bool = False) -> None:
        with self._lock:
            lines = [self._line(self.states[wf]) for wf in self._order]
            subtask = "" if final else self._subtask
        block = lines + ([self._footer_line(subtask)] if subtask else [])
        # The status file always shows the current block (in-place semantics for a file).
        self._write_status_file(block)
        if self._ansi:
            buf = ""
            if self._drawn_lines:
                buf += f"\x1b[{self._drawn_lines}A"     # cursor up to the first line
            for ln in block:
                buf += "\r\x1b[2K" + ln + "\n"          # clear line, write, newline
            # If the block shrank (e.g. the footer just cleared), wipe the leftover lines
            # so no stale footer remains, then step the cursor back to just below the block.
            extra = self._drawn_lines - len(block)
            if extra > 0:
                buf += ("\r\x1b[2K\n" * extra) + f"\x1b[{extra}A"
            self._write(buf)
            self._drawn_lines = len(block)
            self._flush()
        else:
            # Plain fallback (piped/redirected output can't refresh in place): print a line
            # only when its MEANINGFUL state changes. The ETA is excluded from the change
            # test, otherwise every 1 s redraw looks "changed" (ETA jitter) and floods the
            # log with near-identical lines — the exact spam this avoids.
            for wf in self._order:
                st = self.states[wf]
                sig = self._plain_signature(st)
                if sig != st._last_plain:
                    st._last_plain = sig
                    self._write(self._line(st) + "\n")
            self._flush()

    def _plain_signature(self, st: _WFState) -> tuple:
        """State fingerprint for the plain fallback — everything the reader cares about
        EXCEPT the ETA (which changes every tick and would otherwise spam the log)."""
        if st.status in ("done", "error"):
            return (st.status, st.error_msg)
        mval = None if st.metric_val is None else round(float(st.metric_val), 3)
        return (st.status, st.phase, st.phase_total, int(round(st.percent)),
                st.seed, st.n_runs, st.metric_name, mval)

    def _flush(self) -> None:
        try:
            self.stream.flush()
        except Exception:
            pass


class _FooterTqdm:
    """Minimal tqdm-compatible shim used for fine-grained sub-task bars (e.g. the
    autoencoder ``AE(k=…)`` loop). When a MultiWorkflowProgress is active it forwards the
    bar to that display's single footer line — so the sub-task shows as ONE separate
    bottom line that never collides with the per-workflow block — and prints nothing to the
    terminal itself. When no progress display is active it delegates to the real ``tqdm``,
    so standalone scripts keep their normal bars."""

    _MIN_INTERVAL = 0.2   # seconds between footer updates (avoid thrashing the redraw)

    def __init__(self, iterable=None, total=None, desc="", leave=False, **kwargs):
        self._iterable = iterable
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self._total = total
        self._desc = desc or ""
        self._n = 0
        self._last = 0.0
        self._active = _ACTIVE
        self._real = None
        if self._active is None:                       # no live block → real tqdm
            from tqdm import tqdm as _real_tqdm
            self._real = _real_tqdm(iterable, total=total, desc=desc, leave=leave, **kwargs)

    # ------- footer text + emit
    def _text(self) -> str:
        if self._total:
            pct = 100.0 * self._n / max(1, self._total)
            return f"{self._desc} {pct:3.0f}% ({self._n}/{self._total})"
        return f"{self._desc} {self._n}"

    def _emit(self, force: bool = False) -> None:
        if self._active is None:
            return
        now = time.time()
        if force or now - self._last >= self._MIN_INTERVAL:
            self._last = now
            self._active.set_subtask(self._text())

    # ------- iterator protocol
    def __iter__(self):
        if self._real is not None:
            yield from self._real
            return
        for item in (self._iterable if self._iterable is not None else iter(int, 1)):
            yield item
            self._n += 1
            self._emit()
        if self._active is not None:
            self._active.clear_subtask()

    # ------- manual-update protocol
    def update(self, n: int = 1) -> None:
        if self._real is not None:
            self._real.update(n)
            return
        self._n += n
        self._emit()

    def set_description(self, desc: str = "", refresh: bool = True) -> None:
        self._desc = desc or ""
        if self._real is not None:
            self._real.set_description(desc, refresh=refresh)
        else:
            self._emit(force=True)

    set_description_str = set_description

    def close(self) -> None:
        if self._real is not None:
            self._real.close()
        elif self._active is not None:
            self._active.clear_subtask()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def tqdm(*args, **kwargs) -> _FooterTqdm:
    """Drop-in ``tqdm`` replacement that renders through the active MultiWorkflowProgress
    footer when one is running, and falls back to the real ``tqdm`` otherwise. Import this
    instead of ``from tqdm import tqdm`` for sub-task bars that must not fight the block."""
    return _FooterTqdm(*args, **kwargs)
