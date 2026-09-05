# env_utils.py
import os
import sys
import time
from dotenv import dotenv_values

_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# GenAI4Science (HUN-REN SZTAKI) — an OpenAI-compatible endpoint for open-weight
# models. Two deployments:
#   - the shared default endpoint: rate-limited, sheds load, kept for model
#     trials and light use;
#   - the performance endpoint: dedicated capacity, up to 4 concurrent calls
#     (a 5th trips a ~1-minute key block), reached at a different base URL.
GENAI4SCIENCE_BASE_URL = "https://genai.science-cloud.hu/api"
GENAI4SCIENCE_PERFORMANCE_BASE_URL = "https://genai.science-cloud.hu/performance-api/"


def genai4science_endpoint() -> tuple[str, str]:
    """``(base_url, api_key_env)`` for the GenAI4Science endpoint.

    Prefers the **performance** endpoint whenever
    ``GENAI4SCIENCE_PERFORMANCE_API_KEY`` is set, and falls back to the shared
    default endpoint otherwise. The key and the base URL always move together,
    so a performance key is never sent to the default host or the reverse.
    Either base URL can still be overridden by its own ``*_BASE_URL`` env var.

    ``api_key_env`` is a variable *name*, not a value — the caller reads it from
    the environment when it builds its client.
    """
    if os.environ.get("GENAI4SCIENCE_PERFORMANCE_API_KEY"):
        return (
            os.environ.get("GENAI4SCIENCE_PERFORMANCE_BASE_URL",
                           GENAI4SCIENCE_PERFORMANCE_BASE_URL),
            "GENAI4SCIENCE_PERFORMANCE_API_KEY",
        )
    return (
        os.environ.get("GENAI4SCIENCE_BASE_URL", GENAI4SCIENCE_BASE_URL),
        "GENAI4SCIENCE_API_KEY",
    )


def genai4science_tier() -> str:
    """``"performance"`` when ``GENAI4SCIENCE_PERFORMANCE_API_KEY`` is set in the
    environment right now, else ``"shared"`` — the endpoint
    :func:`genai4science_endpoint` would pick."""
    return "performance" if os.environ.get("GENAI4SCIENCE_PERFORMANCE_API_KEY") else "shared"


def probe_genai4science(base_url: str, api_key: str, model: str,
                        *, timeout: float = 45.0) -> tuple[bool, str]:
    """Send one 1-token completion to ``model``. Returns ``(ok, message)``.

    A cold open-weight model on the shared endpoint can take most of a minute to
    load, hence the generous timeout and ``max_retries=0`` (the caller decides
    what a failure means).
    """
    try:
        from openai import OpenAI
    except ImportError:
        return False, "openai package not importable — cannot probe"
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    t0 = time.monotonic()
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
        )
        return True, f"OK in {time.monotonic() - t0:.1f}s"
    except Exception as exc:  # noqa: BLE001 — any transport/HTTP error is a failed probe
        first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return False, f"{exc.__class__.__name__}: {first[:200]}"


def report_genai4science_endpoint(*, probe_model: str | None = None,
                                  timeout: float = 45.0,
                                  allow_fallback: bool = True,
                                  stream=None) -> tuple[str, str]:
    """Announce which GenAI4Science key/endpoint a run will use, and check it.

    Prints, every time:
      - ``performance endpoint`` in green when ``GENAI4SCIENCE_PERFORMANCE_API_KEY``
        is set, or
      - a **warning** that the performance key is missing and the run has fallen
        back to the rate-limited shared endpoint (``GENAI4SCIENCE_API_KEY``).

    When ``probe_model`` is given, sends a 1-token completion to confirm the
    endpoint actually answers and prints the result. If the **performance**
    endpoint fails the probe and ``allow_fallback`` is set and a
    ``GENAI4SCIENCE_API_KEY`` is available that DOES answer, this pops the
    performance key from ``os.environ`` for the rest of the process — so every
    later :func:`genai4science_endpoint` call (client builds, subprocess envs)
    transparently uses the shared endpoint — and says so loudly.

    Returns ``(base_url, api_key_env)`` for the endpoint that won, same shape as
    :func:`genai4science_endpoint`.
    """
    out = stream or sys.stdout

    def emit(msg: str) -> None:
        print(msg, file=out, flush=True)

    base_url, api_key_env = genai4science_endpoint()
    tier = genai4science_tier()
    key = os.environ.get(api_key_env)

    if tier == "performance":
        emit(f"{_GREEN}GenAI4Science: performance endpoint{_RESET} "
             f"{_DIM}({api_key_env}={summarize_value(key) if key else '<not set>'}  {base_url}){_RESET}")
    else:
        emit(f"{_YELLOW}GenAI4Science: GENAI4SCIENCE_PERFORMANCE_API_KEY is not set "
             f"— using the SHARED endpoint via {api_key_env}.{_RESET}")
        emit(f"{_YELLOW}  The shared endpoint is rate-limited and drops calls under "
             f"batch load; set the performance key for dedicated (4-concurrent) "
             f"capacity.{_RESET}")

    if not key:
        emit(f"{_RED}  {api_key_env} is not set — GenAI4Science calls will fail.{_RESET}")
        return base_url, api_key_env

    if not probe_model:
        return base_url, api_key_env

    ok, msg = probe_genai4science(base_url, key, probe_model, timeout=timeout)
    emit(f"{_GREEN if ok else _RED}  probe {probe_model}: {msg}{_RESET}")
    if ok:
        return base_url, api_key_env

    # The chosen endpoint did not answer.
    shared_key = os.environ.get("GENAI4SCIENCE_API_KEY")
    if tier != "performance" or not allow_fallback or not shared_key:
        if tier == "performance":
            emit(f"{_YELLOW}  The performance endpoint is not answering and there "
                 f"is no GENAI4SCIENCE_API_KEY to fall back to. It will NOT switch "
                 f"on its own — fix the endpoint or set the shared key.{_RESET}")
        return base_url, api_key_env

    emit(f"{_YELLOW}  Performance endpoint down — trying the shared endpoint "
         f"(GENAI4SCIENCE_API_KEY)…{_RESET}")
    shared_base = os.environ.get("GENAI4SCIENCE_BASE_URL", GENAI4SCIENCE_BASE_URL)
    ok2, msg2 = probe_genai4science(shared_base, shared_key, probe_model, timeout=timeout)
    emit(f"{_GREEN if ok2 else _RED}  probe {probe_model} (shared): {msg2}{_RESET}")
    if not ok2:
        emit(f"{_RED}  Neither GenAI4Science endpoint answered for {probe_model}.{_RESET}")
        return base_url, api_key_env

    # Fall back for the rest of this process: drop the performance key so every
    # later genai4science_endpoint() resolves to the shared one.
    os.environ.pop("GENAI4SCIENCE_PERFORMANCE_API_KEY", None)
    emit(f"{_YELLOW}  → FALLING BACK to the shared endpoint (GENAI4SCIENCE_API_KEY) "
         f"for the rest of this run.{_RESET}")
    return shared_base, "GENAI4SCIENCE_API_KEY"


def qwen3_thinking_off(model: str, messages: list[dict]) -> list[dict]:
    """Append the ``/no_think`` soft switch to a Qwen3 call's last user turn.

    Qwen3 models emit a ``<think>…</think>`` reasoning pass by default, which
    roughly 10×'s the tokens per call. It buys nothing for the two things this
    project asks a Qwen3 model to do — write an abstract from a title, and rate
    a document 0–1 — and ``clean_generated`` strips the block from the corpus
    anyway, so the cost is pure wall-clock.

    ``/no_think`` is Qwen3's own chat-template switch, honoured by every backend
    that uses that template (this endpoint is LiteLLM → Ollama). A model that
    ignores it just sees nine extra characters. **Non-Qwen3 models are returned
    unchanged** — the check is a substring match on the model id, so this is
    safe to call unconditionally on every generation / judge call.

    A new list is returned; the input messages are not mutated.
    """
    if "qwen3" not in model.lower():
        return messages
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] = m["content"].rstrip() + " /no_think"
            return out
    for m in out:  # no user turn (unusual) — put it on the system message
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = m["content"].rstrip() + "\n\n/no_think"
            return out
    return out


def summarize_value(value: str) -> str:
    """Return masked form: ****last4 or boolean string."""
    lower = value.lower()
    if lower in ("true", "false"):
        return lower
    return "****" + value[-4:] if len(value) > 4 else "****" + value


def doublecheck_env(file_path: str):
    """Check environment variables against a .env file and print summaries."""
    if not os.path.exists(file_path):
        print(f"Did not find file {file_path}.")
        print("This is used to double check the key settings for the notebook.")
        print("This is just a check and is not required.\n")
        return

    parsed = dotenv_values(file_path)
    for key in parsed.keys():
        current = os.getenv(key)
        if current is not None:
            print(f"{key}={summarize_value(current)}")
        else:
            print(f"{key}=<not set>")


# ========== utility to check packages and python based on pyproject.toml  =====================================

# Requires: pip install packaging
import sys, tomllib
from pathlib import Path
from importlib import metadata
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def _fmt_row(cols, widths):
    return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def doublecheck_pkgs(pyproject_path="pyproject.toml", verbose=False):
    p = Path(pyproject_path)
    if not p.exists():
        print(f"ERROR: {pyproject_path} not found.")
        return None

    # Load pyproject + python requirement
    with p.open("rb") as f:
        data = tomllib.load(f)
    project = data.get("project", {})
    python_spec_str = project.get("requires-python") or ">=3.11"

    py_ver = Version(
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    py_ok = py_ver in SpecifierSet(python_spec_str)

    # Load deps (PEP 621)
    deps = project.get("dependencies", [])
    if not deps:
        if verbose or not py_ok:
            print("No [project].dependencies found in pyproject.toml.")
            print(
                f"Python {py_ver} {'satisfies' if py_ok else 'DOES NOT satisfy'} requires-python: {python_spec_str}"
            )
            print(f"Executable: {sys.executable}")
        return None

    # Evaluate deps
    results = []
    problems = []
    for dep in deps:
        try:
            req = Requirement(dep)
            name = req.name
            spec = str(req.specifier) if req.specifier else "(any)"
        except Exception:
            name, spec = dep, "(unparsed)"

        rec = {
            "package": name,
            "required": spec,
            "installed": "-",
            "path": "-",
            "status": "❌ Missing",
        }

        try:
            installed_ver = metadata.version(name)
            rec["installed"] = installed_ver
            try:
                dist = metadata.distribution(name)
                rec["path"] = str(dist.locate_file(""))
            except Exception:
                rec["path"] = "(unknown)"

            if spec not in ("(any)", "(unparsed)") and any(op in spec for op in "<>="):
                sset = SpecifierSet(spec)
                if Version(installed_ver) in sset:
                    rec["status"] = "✅ OK"
                else:
                    rec["status"] = "⚠️ Version mismatch"
            else:
                rec["status"] = "✅ OK"

        except metadata.PackageNotFoundError:
            # keep defaults: installed "-", status "❌ Missing"
            pass

        results.append(rec)
        if rec["status"] != "✅ OK":
            problems.append(rec)

    should_print = verbose or (not py_ok) or bool(problems)
    if should_print:
        # Python status
        print(
            f"Python {py_ver} {'satisfies' if py_ok else 'DOES NOT satisfy'} requires-python: {python_spec_str}"
        )

        # Table (no hints column)
        headers = ["package", "required", "installed", "status", "path"]

        def short_path(s, maxlen=80):
            s = str(s)
            return s if len(s) <= maxlen else ("…" + s[-(maxlen - 1) :])

        rows = [
            [
                r["package"],
                r["required"],
                r["installed"],
                r["status"],
                short_path(r["path"]),
            ]
            for r in results
        ]
        widths = [
            max(len(h), *(len(str(row[i])) for row in rows))
            for i, h in enumerate(headers)
        ]
        print(_fmt_row(headers, widths))
        print(_fmt_row(["-" * w for w in widths], widths))
        for row in rows:
            print(_fmt_row(row, widths))

        # Summarize issues without prescribing a tool
        if problems:
            print("\nIssues detected:")
            for r in problems:
                print(
                    f"- {r['package']}: {r['status']} (required {r['required']}, installed {r['installed']}, path {r['path']})"
                )

        if verbose or problems or not py_ok:
            print("\nEnvironment:")
            print(f"- Executable: {sys.executable}")

    return None
