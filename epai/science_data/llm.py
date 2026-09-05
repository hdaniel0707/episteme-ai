"""Resumable, concurrent LLM calls with append-only JSONL checkpointing.

Stages 3 and 4 are the expensive parts of this pipeline: one call per paper for
the prompt spec, then one per (paper × generator) for the abstracts. At a few
thousand papers that is enough work that a crash near the end must not cost the
whole run, and adding a second generator later must not re-pay for the first.

So every result is appended to a JSONL shard the moment it lands, keyed by a
caller-supplied string. On restart, completed keys are skipped. Adding a
generator only issues the calls whose keys are missing.

Failures are recorded too, with ``ok: false`` and the error text, rather than
being retried forever or silently dropped. That keeps the shard an honest
record of what happened, and lets a re-run pick up only the failures (see
``retry_failed``).
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

YELLOW = "\033[93m"
RESET = "\033[0m"

DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF = 2.0


class Shard:
    """Append-only JSONL checkpoint, safe to write from several threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.records: dict[str, dict] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A crash mid-write can leave one truncated final line.
                    print(f"{YELLOW}  {self.path.name}:{line_no} unreadable, skipped{RESET}")
                    continue
                self.records[rec["key"]] = rec
        n_ok = sum(1 for r in self.records.values() if r.get("ok"))
        print(f"  Resuming from {self.path.name}: {n_ok} done, {len(self.records) - n_ok} failed")

    def done_keys(self, include_failed: bool = False) -> set[str]:
        """Keys that need no further work.

        Failures count as done by default, so a re-run does not hammer records
        that fail deterministically (a content filter, an over-length input).
        Pass ``include_failed=True`` — via ``--retry-failed`` on the stages — to
        take another pass at them.
        """
        if include_failed:
            return {k for k, r in self.records.items() if r.get("ok")}
        return set(self.records)

    def append(self, key: str, ok: bool, payload: dict) -> None:
        rec = {"key": key, "ok": ok, **payload}
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            self.records[key] = rec

    def ok_records(self) -> dict[str, dict]:
        return {k: r for k, r in self.records.items() if r.get("ok")}


def call_with_retry(
    fn: Callable[[], str],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> tuple[bool, str]:
    """Run *fn*, retrying transient failures with exponential backoff.

    Returns ``(ok, text_or_error)``. Every exception type is retried: the
    provider SDKs raise a wide range for rate limits and transient server
    errors, and distinguishing them reliably is not worth the coupling. A
    deterministic failure simply burns its retries once and is then recorded.

    The wait is randomised over ``[backoff * 2**attempt / 2, backoff *
    2**attempt]``. Without the jitter, ``max_workers`` calls that hit one rate
    limit together sleep for exactly the same interval and then retry in
    lockstep, re-creating the burst that caused the limit — the more workers,
    the worse it gets, which is the opposite of what raising them is for.
    """
    last_error = ""
    for attempt in range(max_retries):
        try:
            result = fn()
            if result and result.strip():
                return True, result.strip()
            last_error = "empty response"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < max_retries - 1:
            delay = backoff * (2**attempt)
            time.sleep(random.uniform(delay / 2, delay))

    return False, last_error


def run_resumable(
    tasks: Iterable[tuple[str, Callable[[], str]]],
    shard: Shard,
    max_workers: int = 8,
    desc: str = "calling model",
    retry_failed: bool = False,
    payload_of: Callable[[str, str], dict] | None = None,
) -> Shard:
    """Run ``(key, thunk)`` tasks concurrently, checkpointing each result.

    Tasks whose key is already complete are skipped without being submitted.
    ``payload_of(key, text)`` builds the extra fields stored alongside the
    result; it defaults to ``{"text": text}``.

    Each thunk must be self-contained — closed over its own inputs, reading no
    shared mutable state — because that is what makes the concurrency safe: the
    workers share only the HTTP client (thread-safe) and the API is stateless,
    so one call cannot see another's content. Results are matched by future
    identity, never by completion order, so a reply cannot land under the wrong
    key.
    """
    payload_of = payload_of or (lambda _key, text: {"text": text})
    skip = shard.done_keys(include_failed=retry_failed)

    pending = [(k, thunk) for k, thunk in tasks if k not in skip]

    # Two tasks under one key would both run, both write to the shard, and the
    # slower one would win — so the corpus would silently hold whichever reply
    # finished last, and a re-run could pick the other. That is a bug in the
    # caller's key scheme (a duplicate id in the frame, say), and it is worth
    # failing loudly on rather than paying for twice.
    seen: set[str] = set()
    repeated: set[str] = set()
    for key, _ in pending:
        if key in seen:
            repeated.add(key)
        seen.add(key)
    if repeated:
        duplicates = sorted(repeated)
        shown = ", ".join(duplicates[:5])
        more = f" (+{len(duplicates) - 5} more)" if len(duplicates) > 5 else ""
        raise ValueError(
            f"{len(duplicates)} duplicate task key(s): {shown}{more}. "
            f"Each key identifies one checkpointed call, so duplicates would "
            f"race to overwrite each other in {shard.path.name}."
        )

    if not pending:
        print(f"  Nothing to do — all {len(skip)} results already in {shard.path.name}")
        return shard

    print(f"  {len(pending)} call(s) to make ({len(skip)} already done)")

    n_ok = n_fail = 0
    # Not a `with` block: its __exit__ is shutdown(wait=True), which on Ctrl-C
    # waits for every one of the (possibly thousands of) queued tasks to run
    # before the interrupt propagates — the run looks hung at exactly the moment
    # someone is trying to stop it. Cancelling the unstarted futures instead
    # loses nothing: each finished result is already durable in the shard, and
    # a re-run skips it.
    #
    # `except BaseException` and not `except KeyboardInterrupt`, because Ctrl-C
    # is not the only way out of the loop below — `shard.append` writes to disk
    # and can raise. Any escape that left the pool un-shut-down would hand the
    # queue to ThreadPoolExecutor's atexit hook, which drains it in full: the
    # same apparent hang, on the path where something has already gone wrong.
    # Every exit shuts the pool down, and only the clean one waits.
    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {pool.submit(call_with_retry, thunk): key for key, thunk in pending}
    abandoned = False
    try:
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="call"):
            key = futures[fut]
            try:
                ok, text = fut.result()
            except Exception as e:  # pragma: no cover — call_with_retry catches
                ok, text = False, f"{type(e).__name__}: {e}"

            if ok:
                shard.append(key, True, payload_of(key, text))
                n_ok += 1
            else:
                shard.append(key, False, {"error": text})
                n_fail += 1
    except BaseException as e:
        abandoned = True
        why = (
            "Interrupted"
            if isinstance(e, KeyboardInterrupt)
            else f"Stopped by {type(e).__name__}"
        )
        print(f"\n{YELLOW}  {why} after {n_ok + n_fail} of {len(pending)} "
              f"call(s). Everything finished is in {shard.path.name}; re-run to "
              f"continue from there.{RESET}")
        raise
    finally:
        # In-flight calls (at most `max_workers`) are not interruptible and are
        # left to finish; it is the unstarted queue that would cost minutes.
        pool.shutdown(wait=not abandoned, cancel_futures=abandoned)

    print(f"  {n_ok} succeeded, {n_fail} failed")
    if n_fail:
        print(
            f"{YELLOW}  Re-run with --retry-failed to retry the {n_fail} failure(s).{RESET}"
        )
    return shard


def openai_text_call(client, model: str, system: str, user: str) -> Callable[[], str]:
    """Build a zero-argument thunk that returns the model's text response.

    Deliberately minimal: only ``model`` and ``messages`` are sent. Sampling and
    token-limit parameter names vary across model generations, and the prompts
    already state the required length, so there is nothing to gain by pinning
    them here.

    The one model-specific touch: ``qwen3_thinking_off`` appends ``/no_think``
    to a Qwen3 model's prompt (a no-op for every other model), because its
    default ``<think>`` pass 10×'s the tokens per abstract for no gain.
    """
    from epai.utils.env_utils import qwen3_thinking_off

    messages = qwen3_thinking_off(model, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    def _call() -> str:
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""

    return _call
