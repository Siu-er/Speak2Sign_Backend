"""Generate the english->gloss lexicon from the canonical vocabulary.

The vocabulary (data/vocabulary.json) is the single source of truth: each entry
lists a canonical gloss and the english words that map to it. This flattens that
into data/lexicon.json, the form the glosser consumes, and preserves any
existing lexicon entries that are not yet represented in the vocabulary (so
grammar glosses and extra synonyms are not lost).

Where the vocabulary and the existing lexicon disagree on a word's gloss, the
vocabulary wins, because it is the contract that synthesis assets are aligned to.

Run from the backend root: python tools/build_lexicon.py
"""

import json
import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    vocab = load(os.path.join(DATA_DIR, "vocabulary.json"))
    lex_path = os.path.join(DATA_DIR, "lexicon.json")
    existing = load(lex_path)

    # Start from existing so non-target entries (grammar glosses, extra
    # synonyms) survive, then overlay the vocabulary as authoritative.
    merged = dict(existing)
    added, changed = 0, 0
    for entry in vocab["entries"]:
        gloss = entry["gloss"].upper()
        for word in entry.get("english", []):
            w = word.lower()
            if w not in merged:
                added += 1
            elif merged[w] != gloss:
                changed += 1
            merged[w] = gloss

    # Stable ordering: vocabulary order first, then leftover existing words.
    ordered = {}
    for entry in vocab["entries"]:
        for word in entry.get("english", []):
            w = word.lower()
            if w in merged:
                ordered[w] = merged[w]
    for w, g in merged.items():
        if w not in ordered:
            ordered[w] = g

    with open(lex_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    print(f"Wrote {lex_path}")
    print(f"words: {len(ordered)} (added {added}, regloss {changed})")


if __name__ == "__main__":
    main()
