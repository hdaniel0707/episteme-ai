r"""Does our EditLens preprocessing still match upstream's?

    python3 epai/ai_detection/score/check_editlens_cleaning.py

Runs on a bare interpreter -- no venv, no numpy, no torch -- because everything
it touches is stdlib plus a stub, and a check you cannot run is a check you
will not run.

WHY THIS EXISTS
---------------
``add_detector_scores.py`` reimplements ``clean_text`` from
``external/EditLens/scripts/preprocess.py`` instead of importing it. That is a
licence decision, not a technical one: EditLens is CC BY-NC-SA (share-alike),
and importing it would propagate the licence into this tree.

The cost of a reimplementation is drift, and drift here is silent and total.
``pangram/editlens_roberta-large`` was TRAINED on text put through that
function; feed it text cleaned even slightly differently and the scores are
quietly wrong, with nothing in the output to say so. So the equivalence gets
checked rather than assumed. If a ``git pull`` in the submodule changes
preprocess.py, this is what tells you.

The `emoji` package is stubbed to the identity, so this compares everything
EXCEPT emoji normalisation -- which is the one part that cannot differ, being a
direct call into the same library in both implementations.

WHAT THE FIXTURES ARE FOR
-------------------------
Half of them are regression cases for behaviour that is surprising but real, and
they are worth reading before trusting the column:

  * "Abstract\n..."   -- upstream deletes the first line as assistant
                         boilerplate. On a science corpus that is a section
                         heading, and its human documents are the ones that
                         have it.
  * "Herewith I..."   -- startswith("Here") matches, so an ordinary human
                         sentence loses its first line.
  * "Titles are..."   -- startswith("Title") matches. Same problem.

None of these are bugs in OUR code -- they are upstream's prefix rule, and
matching it is the whole point. They are the reason add_detector_scores.py
counts stripped rows per file and prints the count.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
UPSTREAM = BASE_DIR / "external" / "EditLens" / "scripts" / "preprocess.py"
OURS = BASE_DIR / "epai" / "ai_detection" / "score" / "add_detector_scores.py"

# Functions to lift out of add_detector_scores.py. Pulled by AST rather than
# imported, because importing that module pulls in torch, transformers and a
# command line.
WANTED = ("_demojize", "_strip_emoji", "remove_think_tag", "remove_ai_header", "clean_text")

FIXTURES = [
    # assistant boilerplate -- the case the rule is actually for
    "Sure! Here is the essay you asked for.\nThe actual content begins here.\nAnd continues.",
    "Certainly.\nSecond line.",
    "I'm happy to help!\nHere is the thing.",
    "**Here** is a bolded header\nbody text",
    "   \n\n  Here we go\nreal content",
    # academic prose the rule misfires on
    "Abstract\nWe present a novel method for detecting machine-generated text.\nResults follow.",
    "Title: On the Detection of Synthetic Prose\nThe body of the paper.",
    # ordinary human prose the rule ALSO misfires on, via prefix matching
    "Herewith I submit my findings.\nSecond line.",
    "Titles are hard.\nSecond line.",
    # things that must pass through untouched
    "A perfectly ordinary human paragraph.\n\nWith a second one.",
    "<think>internal reasoning</think>\nThe visible answer.",
    "MiXeD CaSe   and\t\tirregular\n\n whitespace ",
    # degenerate inputs
    "Here",
    "Abstract",
    "",
    "\n\n\n",
]


def load_upstream():
    if not UPSTREAM.is_file():
        raise SystemExit(
            f"Not found: {UPSTREAM.relative_to(BASE_DIR)}\n"
            f"  The EditLens submodule is not checked out. Run:\n"
            f"    git submodule update --init --recursive"
        )
    # Stub `emoji` so upstream imports on a bare interpreter. Both sides call the
    # same library for that step, so identity here compares everything else.
    stub = types.ModuleType("emoji")
    stub.demojize = lambda t: t
    stub.replace_emoji = lambda t, r="": t
    sys.modules["emoji"] = stub

    module = types.ModuleType("upstream_preprocess")
    exec(UPSTREAM.read_text(), module.__dict__)
    return module


def load_ours():
    source = OURS.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks = ["import re"]
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "BOILERPLATE_STARTS" for t in node.targets
        ):
            chunks.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
        if isinstance(node, ast.FunctionDef) and node.name in WANTED:
            chunks.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
            found.add(node.name)
    missing = set(WANTED) - found
    if missing:
        raise SystemExit(
            f"Could not find {', '.join(sorted(missing))} in {OURS.name}. "
            f"Were they renamed? This check needs updating."
        )
    module = types.ModuleType("ours")
    exec("\n\n".join(chunks), module.__dict__)
    return module


def main() -> int:
    upstream = load_upstream()
    ours = load_ours()

    failures = 0
    stripped_count = 0
    for i, text in enumerate(FIXTURES):
        theirs = upstream.clean_text(text)
        mine, stripped = ours.clean_text(text, emoji_enabled=False)
        stripped_count += bool(stripped)
        if theirs != mine:
            failures += 1
            print(f"MISMATCH [{i}] input={text!r}")
            print(f"    upstream: {theirs!r}")
            print(f"    ours    : {mine!r}")
        else:
            flag = "   <- first line stripped" if stripped else ""
            shown = mine if len(mine) <= 56 else mine[:53] + "..."
            print(f"ok [{i:>2}] {shown!r}{flag}")

    print()
    print(f"{len(FIXTURES) - failures}/{len(FIXTURES)} fixtures identical to upstream; "
          f"{stripped_count} had their first line removed.")
    if failures:
        print()
        print("Our reimplementation has drifted from EditLens's preprocess.py.")
        print("Until it matches, detector_score_pangram_* is scoring text the model")
        print("was not trained on. Fix add_detector_scores.py::clean_text.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
