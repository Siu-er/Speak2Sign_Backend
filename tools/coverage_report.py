"""Coverage report for the Speak2Sign vocabularies.

Reads data/registry.json (run build_registry.py first) and the raw lexicon,
and prints a human-readable coverage breakdown: recognition and synthesis
coverage of the target vocabulary by tier and category, plus the raw
lexicon-to-asset health (how many glosses fingerspell).
"""

import json
import os
from collections import defaultdict

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def asset_keys():
    d = os.path.join(DATA_DIR, "signs")
    return set(os.path.splitext(f)[0].upper() for f in os.listdir(d) if f.endswith(".sigml"))


def bar(ok, total):
    pct = (100 * ok // total) if total else 0
    return f"{ok}/{total} ({pct}%)"


def main():
    reg = load(os.path.join(DATA_DIR, "registry.json"))
    lexicon = load(os.path.join(DATA_DIR, "lexicon.json"))
    assets = asset_keys()
    entries = reg["entries"]

    print("=" * 60)
    print("SPEAK2SIGN VOCABULARY COVERAGE")
    print("=" * 60)
    print(f"target language: {reg.get('target_language')}")
    print(f"counts: {json.dumps(reg['counts'])}")

    # Raw lexicon health (independent of target)
    glosses = set(g.upper() for g in lexicon.values() if g)
    with_asset = sorted(g for g in glosses if g in assets)
    fingerspell = sorted(g for g in glosses if g not in assets)
    orphan_assets = sorted(a for a in assets if a not in glosses and not a.isdigit() and len(a) > 1)
    print("\n-- LEXICON HEALTH --")
    print(f"unique glosses: {len(glosses)}")
    print(f"  render as a sign:  {bar(len(with_asset), len(glosses))}")
    print(f"  fingerspelled:     {bar(len(fingerspell), len(glosses))}")
    print(f"orphan word-signs (asset exists, no lexicon word): {len(orphan_assets)}")

    # Target coverage overall
    recog_ok = sum(1 for e in entries if e["recog_status"] == "covered")
    synth_ok = sum(1 for e in entries if e["synth_status"] == "sign")
    n = len(entries)
    print("\n-- TARGET VOCABULARY COVERAGE --")
    print(f"recognition (sign->text): {bar(recog_ok, n)}")
    print(f"synthesis (text->sign):   {bar(synth_ok, n)}")

    # By tier
    print("\n-- BY TIER --")
    tiers = defaultdict(lambda: [0, 0, 0])  # recog_ok, synth_ok, total
    for e in entries:
        t = tiers[e["tier"]]
        t[0] += e["recog_status"] == "covered"
        t[1] += e["synth_status"] == "sign"
        t[2] += 1
    for tier in sorted(tiers):
        r, s, tot = tiers[tier]
        print(f"tier {tier}: recog {bar(r, tot)}  synth {bar(s, tot)}")

    # By category
    print("\n-- BY CATEGORY (synthesis) --")
    cats = defaultdict(lambda: [0, 0])
    for e in entries:
        c = cats[e["category"]]
        c[0] += e["synth_status"] == "sign"
        c[1] += 1
    for cat in sorted(cats):
        s, tot = cats[cat]
        print(f"  {cat:12} {bar(s, tot)}")

    # Gaps
    synth_gap = sorted(e["gloss"] for e in entries if e["synth_status"] != "sign")
    recog_gap = sorted(e["gloss"] for e in entries if e["recog_status"] != "covered")
    print("\n-- SYNTHESIS GAPS (fingerspelled, need a sign asset) --")
    print(", ".join(synth_gap) or "none")
    print("\n-- RECOGNITION GAPS (not in the 250-class model) --")
    print(", ".join(recog_gap) or "none")


if __name__ == "__main__":
    main()
