"""Build the canonical vocabulary registry.

Ties the three independently-sourced vocabularies together against the target
everyday vocabulary, so coverage is a derived fact rather than a guess:

  - recognition classes   (data/models/sign_to_prediction_index_map.json)
  - synthesis assets       (data/signs/*.sigml)
  - english->gloss lexicon (data/lexicon.json)
  - target vocabulary      (data/vocabulary.json)

Writes data/registry.json and prints a coverage summary.
"""

import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def asset_keys(signs_dir):
    keys = set()
    for fn in os.listdir(signs_dir):
        if fn.endswith(".sigml"):
            keys.add(os.path.splitext(fn)[0].upper())
    return keys


def recognizer_classes(label_map_path):
    m = load_json(label_map_path)
    return set(k.lower() for k in m.keys())


def build():
    data = os.path.abspath(DATA_DIR)
    lexicon = load_json(os.path.join(data, "lexicon.json"))
    vocab = load_json(os.path.join(data, "vocabulary.json"))
    assets = asset_keys(os.path.join(data, "signs"))
    recog = recognizer_classes(os.path.join(data, "models", "sign_to_prediction_index_map.json"))

    # gloss -> english words pointing at it, from the lexicon
    gloss_to_words = {}
    for word, gloss in lexicon.items():
        if gloss:
            gloss_to_words.setdefault(gloss.upper(), []).append(word)

    entries = []
    for v in vocab["entries"]:
        gloss = v["gloss"].upper()
        english = [w.lower() for w in v.get("english", [])]
        asset = gloss if gloss in assets else None
        # recognition: any synonym (or the gloss itself) is a model class
        recog_class = None
        for cand in [gloss.lower()] + english:
            if cand in recog:
                recog_class = cand
                break
        lexicon_words = sorted(set(gloss_to_words.get(gloss, [])) | set(w for w in english if lexicon.get(w, "").upper() == gloss))
        entries.append({
            "gloss": gloss,
            "tier": v.get("tier"),
            "category": v.get("category"),
            "english": english,
            "recognizer_class": recog_class,
            "asset": asset,
            "lexicon_words": lexicon_words,
            "recog_status": "covered" if recog_class else "missing",
            "synth_status": "sign" if asset else "fingerspell",
        })

    registry = {
        "version": vocab.get("version", 1),
        "target_language": vocab.get("language"),
        "counts": {
            "recognizer_classes": len(recog),
            "synthesis_assets": len(assets),
            "lexicon_words": len(lexicon),
            "target_entries": len(entries),
        },
        "entries": entries,
    }

    out = os.path.join(data, "registry.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    return registry, out


if __name__ == "__main__":
    reg, out = build()
    print(f"Wrote {out}")
    print(f"Counts: {json.dumps(reg['counts'])}")
    recog_ok = sum(1 for e in reg["entries"] if e["recog_status"] == "covered")
    synth_ok = sum(1 for e in reg["entries"] if e["synth_status"] == "sign")
    n = len(reg["entries"])
    print(f"Target recognition coverage: {recog_ok}/{n} ({100*recog_ok//n}%)")
    print(f"Target synthesis coverage:   {synth_ok}/{n} ({100*synth_ok//n}%)")
