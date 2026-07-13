"""Enrich data/signs with real BSL signs from the Dicta-Sign Basic Lexicon.

The avatar already renders BSL; this fills the everyday vocabulary gaps that
currently fingerspell by fetching genuine SiGML signs (Dicta-Sign project,
hosted by UEA Virtual Humans) rather than authoring them.

Crawls the BSL index, maps english word -> sign file, matches the target
vocabulary's synthesis gaps, and (with --apply) saves each as
data/signs/<GLOSS>.sigml. Dry run by default.

Run from the backend root:
  python tools/fetch_signs.py            # dry run: show matches
  python tools/fetch_signs.py --apply    # fetch and write
"""

import argparse
import json
import os
import re
import sys
import time

import requests

BASE = "https://vhg.cmp.uea.ac.uk/demo/DictaSignLexicon/BSL"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SIGNS_DIR = os.path.join(DATA_DIR, "signs")

PAIR_RE = re.compile(
    r'href="\.\./SignIndex/([a-z0-9_]+)\.html".*?(BSL/SignFiles/[A-Za-z0-9_]+\.BSL\.sigml)',
    re.DOTALL | re.IGNORECASE,
)


def index_pages():
    html = requests.get(f"{BASE}/DictIndex.html", timeout=30).text
    return sorted(set(re.findall(r'href="(DictIndex/P\d+\.html)"', html)))


def build_word_map():
    """english word -> absolute SiGML url, preferring the primary (_1n) variant."""
    word_map = {}
    for page in index_pages():
        html = requests.get(f"{BASE}/{page}", timeout=30).text
        for ref, sigml_rel in PAIR_RE.findall(html):
            word = re.sub(r"_\d+[a-z]*$", "", ref).lower()
            url = f"https://vhg.cmp.uea.ac.uk/demo/DictaSignLexicon/{sigml_rel}"
            primary = ref.endswith("_1n")
            if word not in word_map or (primary and "_1n" not in word_map[word]):
                word_map[word] = url
    return word_map


def existing_assets():
    return set(os.path.splitext(f)[0].upper() for f in os.listdir(SIGNS_DIR) if f.endswith(".sigml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    vocab = json.load(open(os.path.join(DATA_DIR, "vocabulary.json"), encoding="utf-8"))
    have = existing_assets()
    word_map = build_word_map()
    print(f"Dicta-Sign BSL lexicon: {len(word_map)} signs indexed")

    matched, missing = [], []
    for entry in vocab["entries"]:
        gloss = entry["gloss"].upper()
        if gloss in have:
            continue  # already renders
        candidates = [gloss.lower()] + [w.lower() for w in entry.get("english", [])]
        url = next((word_map[c] for c in candidates if c in word_map), None)
        if url:
            matched.append((gloss, url))
        else:
            missing.append(gloss)

    print(f"Gap signs matched in lexicon: {len(matched)}")
    print(f"Gap signs still missing:      {len(missing)}")
    print("MISSING:", ", ".join(sorted(missing)))

    if not args.apply:
        print("\n(dry run; re-run with --apply to fetch)")
        for g, u in matched:
            print(f"  {g} <- {u.rsplit('/', 1)[1]}")
        return

    added = 0
    for gloss, url in matched:
        try:
            content = requests.get(url, timeout=30).text
            if "<sigml" not in content:
                print(f"  skip {gloss}: unexpected content")
                continue
            with open(os.path.join(SIGNS_DIR, f"{gloss}.sigml"), "w", encoding="utf-8") as f:
                f.write(content)
            added += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"  fail {gloss}: {e}")
    print(f"\nAdded {added} signs to {SIGNS_DIR}")


if __name__ == "__main__":
    main()
