r"""How one generator is spelled on disk and in a corpus.

A generated row names the model and the prompts behind it, as
``<model_slug>:<prompt_id>`` — ``gpt56luna:0701A``. That is the corpus-side
spelling, produced by :func:`epai.science_data.config.generated_by_label` for
the science pipeline and by this module's rule for the ghostbuster one.

A directory cannot be spelled that way. ``:`` is legal in a POSIX path and
nowhere else, and these trees get copied, rsynced and archived, so the same
generator is a directory named ``gpt56luna_0701A`` — one underscore instead of
the colon. This module owns both spellings and the map between them, because
the two ends are written in different places: the ghostbuster pipeline composes
the directory name and hands it to the fork's ``generate.py``, while
``prepare_{essay,wp,reuter}.py`` read the name back off disk a stage later and
turn it into the label. A rule that lived in only one of them would be a rule
the other could contradict.

WHY NOT JUST SPLIT ON THE UNDERSCORE: the same trees hold directories from the
original Ghostbuster release — ``gpt``, ``gpt_prompt1``, ``gpt_writing``,
``gpt_semantic`` — and ``gpt_prompt1`` is not a model called ``gpt`` under a
prompt called ``prompt1``. So the map keys on the *shape of a prompt id*
(:data:`GENERATOR_DIR_RE`) rather than on the separator, and anything that does
not match that shape is passed through untouched. ``model_slug`` drops every
non-alphanumeric character, so the model half can never contain an underscore
and the two cases cannot overlap.
"""

from __future__ import annotations

import re
from collections.abc import Container, Sequence
from pathlib import Path

# The separator in a corpus label, and the one that stands in for it on disk.
LABEL_SEP = ":"
DIR_SEP = "_"

# A prompt config id: <MMDD><letter>, the date the prompts were written plus a
# letter for the variants written that day — the convention
# cfg/science_data/prompts/ already uses (0807A), reused here so one reader of
# `generated_by` learns one thing.
PROMPT_ID_RE = re.compile(r"^[0-9]{4}[A-Z]$")

# A generator directory written by this pipeline: model slug, underscore,
# prompt id. Deliberately narrow — everything it does not match is a directory
# name from somewhere else and must survive unchanged.
GENERATOR_DIR_RE = re.compile(r"^([a-z0-9]+)_([0-9]{4}[A-Z])$")

# The same generator as a corpus label: the ':' spelling of GENERATOR_DIR_RE,
# and the only shape `generated_by` should hold beside 'human'. Readers that
# take a generator from a user (the analyse scripts) check against this, so a
# mistyped label fails where it was typed rather than selecting no rows.
GENERATOR_LABEL_RE = re.compile(r"^([a-z0-9]+):([0-9]{4}[A-Z])$")

PROMPT_ID_HINT = (
    "a prompt config id is <MMDD><letter> — four digits and one capital "
    "letter, e.g. \"0701A\" — the same convention as cfg/science_data/prompts/"
)


def check_prompt_id(prompt_id: object, where: str) -> str:
    """Validate one prompt id, returning it; raises ``ValueError`` otherwise.

    Worth being strict about: the id's *shape* is what
    :func:`label_from_dir_name` keys on, so an id like ``plain`` or ``0701a``
    would produce a directory the prepare stage silently declines to relabel,
    and a corpus labelled ``gpt56luna_0701a`` — with an underscore — three
    stages later, long after the generation was paid for.
    """
    if not isinstance(prompt_id, str) or not PROMPT_ID_RE.match(prompt_id):
        raise ValueError(f"{where}: {prompt_id!r} is not a prompt id. {PROMPT_ID_HINT}")
    return prompt_id


def check_generator_label(label: object, where: str) -> str:
    """Validate one corpus label, returning it; raises ``ValueError`` otherwise.

    The counterpart of :func:`check_prompt_id` for the whole label. Used where a
    generator arrives from outside the pipeline — a command line naming the
    generator to analyse — so that ``gpt``, ``gpt_plain`` or ``gpt:000A`` are
    rejected as the wrong shape, with the shape spelled out, instead of quietly
    matching no rows in the parquet.
    """
    if not isinstance(label, str) or not GENERATOR_LABEL_RE.match(label):
        raise ValueError(
            f"{where}: {label!r} is not a generator label. A generator is "
            f"<model_slug>{LABEL_SEP}<prompt_id> — lowercase letters and digits, "
            f"{LABEL_SEP!r}, then the prompt id, e.g. \"gpt56luna:0807A\"; "
            f"{PROMPT_ID_HINT}"
        )
    return label


def generator_dir_name(model_token: str, prompt_id: str) -> str:
    """``("gpt56luna", "0701A")`` -> ``"gpt56luna_0701A"``.

    ``model_token`` is an already-slugged model name (see
    :func:`epai.science_data.config.model_slug`); this module takes the token
    rather than the raw model name so it stays free of pipeline imports. The
    result is checked against :data:`GENERATOR_DIR_RE` before it is returned,
    which is what keeps the two directions of this map honest: a token that
    could not be parsed back fails here, where the name is being made, instead
    of at the prepare stage, where it would just look like a directory nobody
    recognises.
    """
    check_prompt_id(prompt_id, "prompt id")
    name = f"{model_token}{DIR_SEP}{prompt_id}"
    if not GENERATOR_DIR_RE.match(name):
        raise ValueError(
            f"{name!r} is not a usable generator directory name: the model part "
            f"{model_token!r} must be lowercase letters and digits only "
            "(model_slug produces exactly that)."
        )
    return name


def label_from_dir_name(name: str) -> str:
    """``"gpt56luna_0701A"`` -> ``"gpt56luna:0701A"``; anything else unchanged.

    The pass-through half is the important one. ``human``, ``gpt``,
    ``gpt_prompt1``, ``gpt_writing`` and the rest of the original Ghostbuster
    generators are directory names this convention never applied to, and a
    corpus built from those trees has to keep calling them what it always
    called them.
    """
    match = GENERATOR_DIR_RE.match(name)
    if not match:
        return name
    return f"{match.group(1)}{LABEL_SEP}{match.group(2)}"


def dir_name_from_label(label: str) -> str:
    """``"gpt56luna:0701A"`` -> ``"gpt56luna_0701A"``; anything else unchanged.

    :func:`label_from_dir_name` the other way round, for code that HAS a label
    and needs a path out of it — a results directory named after the generator
    it describes, so two runs over two generators cannot land in the same one.
    The pass-through half is the same promise: a name that is not a label of
    this convention is not this module's to rewrite.
    """
    match = GENERATOR_LABEL_RE.match(label)
    if not match:
        return label
    return f"{match.group(1)}{DIR_SEP}{match.group(2)}"


def generator_dirs(
    root: Path,
    requested: Sequence[str] | None = None,
    ignore: Container[str] = (),
) -> list[Path]:
    """The generator directories under ``root`` a corpus should read, sorted.

    ``requested`` is the important argument. Left as None every directory is
    taken, which is what a hand-run over a corpus tree wants — but it makes the
    contents of the parquet a function of whatever happens to be on disk, and
    these trees are written to more than once. A corpus directory is *added to*:
    the ghostbuster collect stage copies into it without deleting, so a
    generator from an earlier run stays behind and silently contributes rows
    under its own old label. Naming the directories makes the config decide what
    the corpus holds, and turns that stale directory into an error instead of
    extra rows.

    A requested directory that does not exist is a failure, not an empty set:
    it means the tree is not the one the caller thinks it is.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    if requested is None:
        return sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ignore)

    names = sorted(set(requested))
    missing = [n for n in names if not (root / n).is_dir()]
    if missing:
        present = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"{root}: no such generator directory: {', '.join(missing)}. "
            f"The tree holds: {', '.join(present) or '(nothing)'}"
        )
    return [root / n for n in names]
