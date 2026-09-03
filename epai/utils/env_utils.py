# env_utils.py
import os
from dotenv import dotenv_values

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
