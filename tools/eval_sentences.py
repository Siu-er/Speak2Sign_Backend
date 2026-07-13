"""End-to-end synthesis eval: how much of everyday speech renders as real signs
versus degrades to fingerspelling.

Runs each sentence in data/eval_sentences.txt through the real ASLGlosser and
classifies every gloss token by how the SiGMLGenerator would render it: a known
sign, a number, or fingerspelled. Prints per-sentence and aggregate rates.

This is the regression harness: the fingerspell rate is the number to drive
down and to gate in CI.

Run from the backend root: python tools/eval_sentences.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from asl_glosser import ASLGlosser
from sigml_generator import SiGMLGenerator

DATA_DIR = os.path.join(ROOT, "data")


def classify(token, cache):
    clean = token.replace("++", "")
    up = clean.upper()
    if up in cache:
        return "sign"
    if clean.startswith("FS-"):
        return "fingerspell"
    if clean.isdigit():
        return "number" if clean in cache else "fingerspell"
    if "-" in clean and not clean.startswith("FS-"):
        return "sign" if all(p.upper() in cache for p in clean.split("-")) else "fingerspell"
    return "fingerspell"


def main():
    glosser = ASLGlosser(DATA_DIR)
    glosser.DEBUG = False
    gen = SiGMLGenerator(os.path.join(DATA_DIR, "signs"))
    cache = gen.sigml_cache

    with open(os.path.join(DATA_DIR, "eval_sentences.txt"), "r", encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]

    tot_sign = tot_fs = tot_tok = 0
    print("=" * 70)
    for s in sentences:
        res = glosser.gloss(s)
        toks = res.gloss_tokens
        kinds = [classify(t, cache) for t in toks]
        sign = sum(k == "sign" for k in kinds)
        fs = sum(k == "fingerspell" for k in kinds)
        tot_sign += sign
        tot_fs += fs
        tot_tok += len(toks)
        marked = " ".join(
            t if classify(t, cache) == "sign" else f"[{t}]" for t in toks
        )
        print(f"{s}")
        print(f"  -> {marked}")
        print(f"  signs {sign}/{len(toks)}  fingerspelled {fs}/{len(toks)}")
    print("=" * 70)
    if tot_tok:
        print(f"AGGREGATE: signs {tot_sign}/{tot_tok} ({100*tot_sign//tot_tok}%), "
              f"fingerspelled {tot_fs}/{tot_tok} ({100*tot_fs//tot_tok}%)")
        print("(bracketed tokens render as fingerspelling)")


if __name__ == "__main__":
    main()
