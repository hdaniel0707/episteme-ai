"""Peek at the in-progress science_v3 generate shard. Run from the repo root.

    .venv/bin/python scratchpad_check_gen.py            # summary + 3 samples each
    .venv/bin/python scratchpad_check_gen.py 6          # 6 samples each

Reads data/science/science_v3_generate.jsonl, which is flushed after every call
and already holds the cleaned text. Safe to run repeatedly while generation
runs. Throwaway — delete when the run is done.
"""

import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

SHARD = Path("data/science/science_v3_generate.jsonl")
n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3

if not SHARD.is_file():
    sys.exit(f"not found: {SHARD} (run from the repo root, after generation starts)")

rows = []
for line in SHARD.read_text().splitlines():
    line = line.strip()
    if line:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a crash mid-write can leave one truncated final line

ok, fail = Counter(), Counter()
lengths: dict[str, list[int]] = {}
samples: dict[str, list[str]] = {}
errors: Counter = Counter()

for r in rows:
    label = r["key"].split("|", 1)[1]
    if r.get("ok"):
        ok[label] += 1
        words = len((r.get("abstract") or "").split())
        lengths.setdefault(label, []).append(words)
        samples.setdefault(label, []).append(r.get("abstract") or "")
    else:
        fail[label] += 1
        errors[(label, (r.get("error") or "")[:70])] += 1

print(f"{len(rows)} records in {SHARD}\n")
print(f"{'label':<22}{'ok':>7}{'fail':>7}   words min/median/max")
for label in sorted(set(ok) | set(fail)):
    ls = sorted(lengths.get(label, []))
    span = f"{ls[0]}/{ls[len(ls) // 2]}/{ls[-1]}" if ls else "-"
    print(f"{label:<22}{ok[label]:>7}{fail[label]:>7}   {span}")

if errors:
    print("\nfailure reasons:")
    for (label, msg), count in errors.most_common(10):
        print(f"  {count:>4}  {label:<22} {msg}")

new = [lab for lab in sorted(samples) if not lab.startswith(("human", "gpt56luna"))]
if not new:
    print("\n(no open-weight generations landed yet)")
for label in new:
    print(f"\n{'=' * 70}\n{label}  —  last {n_samples}\n{'=' * 70}")
    for text in samples[label][-n_samples:]:
        print(textwrap.fill(text, 92))
        print("-" * 70)
