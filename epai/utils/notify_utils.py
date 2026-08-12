"""Push pipeline progress to a phone via ntfy.

The pipeline runners in this repo take hours — a science corpus build is a
fetch, thousands of API calls and two GPU stages — so the useful question while
one is running is not "what is it printing" but "is it still alive, and did it
get past the expensive stage". This module answers that on a phone, using
`ntfy <https://ntfy.sh>`_: an HTTP POST to ``<server>/<topic>`` becomes a push
notification on every device subscribed to that topic. No account, no SDK.

Configured entirely by the environment, normally through ``.env``:

``NTFY_PIPELINE_TOPIC``
    The topic to publish to. **No default, and unset means silence** — the
    topic name is the only thing gating who can read these pushes, so it must
    not live in a tracked file, and a fresh clone must not start POSTing
    somewhere. See ``.env_sample``.
``NTFY_SERVER``
    Base URL of the ntfy server. Defaults to ``https://ntfy.sh``.
``EPAI_NOTIFY_NESTED``
    Set by a runner that shells out to another runner (pipeline 3's ``science``
    step is the whole of ``run_pipeline.py``). The inner process still reports
    its own stages, but drops its "started" and "finished" pushes, which would
    otherwise duplicate the outer run's.

This is a *deliberately separate topic* from the Claude Code hook's
(``.claude/hooks/notify.env``). Same app, two subscriptions: pipeline progress
and "Claude finished a turn" are different feeds with different lifetimes, and
one of them is useful precisely when you are nowhere near a terminal.

**Nothing here can fail a run.** Every send is wrapped, has a short timeout, and
returns a bool nobody is obliged to check. A missing topic, a DNS failure, a
502 from ntfy and a rate-limit rejection are all the same non-event: the
pipeline prints what it always printed and carries on.

Sends are synchronous. They happen only at stage boundaries — at most a dozen
per run — so the worst case is ``TIMEOUT_S`` seconds added to a run measured in
hours, which is cheaper than the thread and shutdown handling needed to avoid
it.

Typical use, from a runner's ``main()``::

    notify = notify_utils.configure(
        "science_data", f"{cfg['dataset_id']} {variant}", keys,
        enabled=not args.dry_run,
    )
    notify.started(f"{len(keys)} stages")
    ...
    notify.stage(key, code, seconds)
    notify.failed(key, code, elapsed, resume="... --from fetch")
    notify.finished("corpus ready: data/parquet/x.parquet", total_seconds)
"""

from __future__ import annotations

import os
import re
import urllib.request
from collections.abc import Iterable

TOPIC_VAR = "NTFY_PIPELINE_TOPIC"
SERVER_VAR = "NTFY_SERVER"
NESTED_VAR = "EPAI_NOTIFY_NESTED"

DEFAULT_SERVER = "https://ntfy.sh"

# Long enough to cover a slow round trip on a home connection, short enough that
# a black-holed connection is a hiccup between stages rather than a hang.
TIMEOUT_S = 5.0

# ntfy puts the title in a header, and headers are latin-1 on the wire: a
# non-ASCII title raises at send time rather than arriving mangled. Bodies are
# UTF-8 and have no such limit.
MAX_TITLE = 80
MAX_BODY = 400

# A stage that exits 130 was Ctrl-C'd, which is a different message from a
# crash: someone was there, and they meant it.
INTERRUPTED = 130

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

_dotenv_loaded = False


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _plain(text: str) -> str:
    """Strip ANSI colour and tidy whitespace, keeping the line structure.

    Call sites pass the same strings they print, which in these runners are
    full of ``{YELLOW}...{RESET}`` — escape codes that a phone renders as
    literal garbage. Line breaks are kept because a notification body is read
    at a glance and one fact per line is the readable shape; blank lines are
    dropped, since the printed strings lead with ``\\n`` to space the terminal
    output.
    """
    lines = (" ".join(line.split()) for line in _ANSI.sub("", text).splitlines())
    return "\n".join(line for line in lines if line)


def _ascii(text: str, limit: int) -> str:
    """A header-safe title: ASCII only, single line, clipped."""
    flat = " ".join(_plain(text).split())
    return flat.encode("ascii", "replace").decode("ascii")[:limit]


def elapsed(seconds: float) -> str:
    """``2h 41m`` / ``17m 08s`` / ``43s`` — two units are enough to act on."""
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _topic() -> str:
    """The configured topic, loading ``.env`` on first ask.

    Lazily, because importing this module must not have side effects: the
    runners that use it are also imported by other things. ``load_dotenv`` does
    not override variables already in the environment, so an explicit
    ``NTFY_PIPELINE_TOPIC=... python ...`` still wins.
    """
    global _dotenv_loaded
    if not _dotenv_loaded:
        _dotenv_loaded = True
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
    return os.environ.get(TOPIC_VAR, "").strip()


def push(
    title: str,
    message: str,
    *,
    tags: str = "",
    priority: str = "default",
) -> bool:
    """POST one notification. Returns whether it was sent; never raises.

    ``tags`` is ntfy's comma-separated list of emoji short codes — the first
    one becomes the notification's icon. ``priority`` is ntfy's 1–5 scale by
    name; ``high`` is the one that vibrates insistently, so it is reserved for
    a run that has stopped.
    """
    topic = _topic()
    if not topic:
        return False

    server = (os.environ.get(SERVER_VAR) or DEFAULT_SERVER).rstrip("/")
    body = _plain(message)[:MAX_BODY] or title

    request = urllib.request.Request(
        f"{server}/{topic}",
        data=body.encode("utf-8"),
        headers={
            "Title": _ascii(title, MAX_TITLE),
            "Tags": tags,
            "Priority": priority,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S):
            return True
    except Exception:
        # Every failure mode is the same non-event; see the module docstring.
        return False


# ---------------------------------------------------------------------------
# The per-run notifier
# ---------------------------------------------------------------------------


class Notifier:
    """Stage-boundary pushes for one run of one pipeline.

    Created through :func:`configure`, which also parks it where
    :func:`active` can find it — so a helper like ``confirm()``, called deep in
    a step with no notifier in scope, can still report that the run is now
    sitting at a prompt.
    """

    def __init__(
        self,
        pipeline: str,
        label: str,
        steps: Iterable[str],
        *,
        enabled: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.label = label
        self.steps = list(steps)
        self.nested = bool(os.environ.get(NESTED_VAR))
        # Resolved once: with no topic configured every method is a no-op, and
        # there is no point re-reading the environment a dozen times to
        # rediscover that.
        self.enabled = bool(enabled and _topic())
        self.done = 0

    # -- helpers ------------------------------------------------------------

    def _body(self, *lines: str) -> str:
        """Prefix every push with which corpus this is, then the detail."""
        return "\n".join([self.label, *(line for line in lines if line)])

    def _next(self) -> str:
        return self.steps[self.done] if self.done < len(self.steps) else ""

    # -- events -------------------------------------------------------------

    def started(self, detail: str = "") -> bool:
        """The run has begun. Low priority: you are still at the keyboard."""
        if not self.enabled or self.nested:
            return False
        return push(
            f"{self.pipeline}: started",
            self._body(detail, " -> ".join(self.steps)),
            tags="rocket",
            priority="low",
        )

    def stage(
        self,
        key: str,
        code: int,
        seconds: float,
        *,
        critical: bool = True,
    ) -> bool:
        """One stage finished — the push you actually wanted.

        A non-critical stage that failed is a warning, not a stop: the analysis
        stages are diagnostics and the pipeline deliberately carries on past
        them, so they get a different tag and the same priority rather than an
        alarm. A *critical* failure is reported by :meth:`failed`, which the
        runner calls with the resume command; this method stays quiet about it
        rather than pushing twice.

        An interrupted stage is never a warning, whatever its ``critical``
        flag: the runners stop on a Ctrl-C from any stage, because what it
        saved is partial and everything downstream would consume it.
        """
        if not self.enabled:
            return False
        self.done += 1
        position = f"{self.done}/{len(self.steps)}"

        if code == 0:
            nxt = self._next()
            return push(
                f"{self.pipeline}: {key} ok ({position})",
                self._body(
                    elapsed(seconds),
                    f"next: {nxt}" if nxt else "last stage",
                ),
                tags="white_check_mark",
                priority="default",
            )
        if not critical and code != INTERRUPTED:
            return push(
                f"{self.pipeline}: {key} failed, non-fatal ({position})",
                self._body(
                    f"exit {code} after {elapsed(seconds)}",
                    "Diagnostic stage - the pipeline is carrying on.",
                ),
                tags="warning",
                priority="default",
            )
        return False

    def blocked(self, question: str) -> bool:
        """The run has stopped to ask something and is waiting on stdin.

        Default priority, not high: nothing has gone wrong, the run is just
        parked until someone answers. It matters because an unattended run that
        hits this does nothing at all until you notice.
        """
        if not self.enabled:
            return False
        return push(
            f"{self.pipeline}: waiting for input",
            self._body(question, "The run is parked until you answer."),
            tags="raising_hand",
            priority="default",
        )

    def failed(
        self,
        key: str,
        code: int,
        seconds: float,
        *,
        resume: str = "",
    ) -> bool:
        """A critical stage failed and the run stopped. The one alarm."""
        if not self.enabled:
            return False
        state = "interrupted at" if code == INTERRUPTED else "FAILED at"
        return push(
            f"{self.pipeline}: {state} {key}",
            self._body(
                f"exit {code} after {elapsed(seconds)}",
                f"Resume: {resume}" if resume else "",
            ),
            tags="rotating_light",
            priority="high",
        )

    def finished(self, detail: str = "", seconds: float = 0.0) -> bool:
        """Every selected stage ran and the run is done."""
        if not self.enabled or self.nested:
            return False
        return push(
            f"{self.pipeline}: done ({len(self.steps)} stages)",
            self._body(f"total {elapsed(seconds)}" if seconds else "", detail),
            tags="checkered_flag",
            priority="default",
        )


class _Disabled(Notifier):
    """What :func:`active` returns before any runner has configured one."""

    def __init__(self) -> None:
        super().__init__("epai", "", (), enabled=False)


_ACTIVE: Notifier = _Disabled()


def configure(
    pipeline: str,
    label: str,
    steps: Iterable[str],
    *,
    enabled: bool = True,
) -> Notifier:
    """Build the notifier for this run and make it the process-wide one."""
    global _ACTIVE
    _ACTIVE = Notifier(pipeline, label, steps, enabled=enabled)
    return _ACTIVE


def active() -> Notifier:
    """The current run's notifier, or a no-op one if none was configured."""
    return _ACTIVE


def self_test() -> int:
    """Send one push and say what happened. See ``__main__`` below."""
    topic = _topic()
    if not topic:
        print(
            f"{TOPIC_VAR} is not set, so the pipelines will send nothing.\n"
            f"Add it to .env — see .env_sample."
        )
        return 1

    server = (os.environ.get(SERVER_VAR) or DEFAULT_SERVER).rstrip("/")
    ok = push(
        "epai: test",
        "If this reached your phone, the pipeline runners can too.",
        tags="satellite",
        priority="default",
    )
    print(f"topic  {topic}\nserver {server}\nsent   {'yes' if ok else 'NO'}")
    return 0 if ok else 1


def nested_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """A child process's environment, marked as a nested run.

    For a runner that shells out to another runner: the child reports its own
    stages, but suppresses the "started" and "finished" pushes that would
    duplicate this run's.
    """
    child = dict(env if env is not None else os.environ)
    child[NESTED_VAR] = "1"
    return child


if __name__ == "__main__":
    # Check the setup without waiting for a five-hour pipeline to reach a stage
    # boundary:  uv run python -m epai.utils.notify_utils
    raise SystemExit(self_test())
