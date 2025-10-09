#!/usr/bin/env python3
"""
test_gloss_local.py — Direct tests for ASLGlosser (no API)

- Imports ASLGlosser from asl_glosser.py
- Loads lexicon/config from --data (default: ./data)
- Compares returned gloss to regex expectations
- Tolerates minor variations (optional BE/duplicate FINISH/etc.)
- Exits nonzero if unexpected failures occur (good for CI)

Usage:
  python test_gloss_local.py
  python test_gloss_local.py --data ./data
  python test_gloss_local.py --filter "future"
  python test_gloss_local.py --show-tokens
"""

import argparse
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Pattern

# Import your glosser locally
from asl_glosser import ASLGlosser


@dataclass
class Case:
    name: str
    text: str
    pattern: Pattern[str]  # compiled regex
    xfail: bool = False  # known issue; won't fail the run


# ---------- Utilities ----------

SPACE_RE = re.compile(r"\s+")
TRAILING_SPACE_BEFORE_Q = re.compile(r"\s+\?")


def normalize_gloss(s: str) -> str:
    """Normalize for stable comparisons."""
    s = s.strip()
    s = SPACE_RE.sub(" ", s)
    s = TRAILING_SPACE_BEFORE_Q.sub(" ?", s)
    return s


def compile_pat(pat: str) -> Pattern[str]:
    return re.compile(pat)


def pretty_diff(text: str, expected_rx: str, got: str) -> str:
    return textwrap.dedent(f"""
        Text: {text}
        Expected (regex): {expected_rx}
        Got            : {got}
    """).strip()


# ---------- Test Catalog ----------
# Patterns are anchored and allow minor variation where appropriate.

RAW_CASES: List[Case] = [
    Case(
        "Declarative + future + possessives",
        "When I finish my homework, I will call my friend to play basketball.",
        compile_pat(r'^ME FINISH (?:ME )?HOMEWORK ME FUTURE CALL (?:ME )?FRIEND PLAY BASKETBALL$')
    ),
    Case(
        "Yes/No Q",
        "Do you like coffee?",
        compile_pat(r'^YOU LIKE COFFEE \?$')
    ),
    Case(
        "WH Q",
        "Where are you going?",
        compile_pat(r'^YOU GO WHERE \?$')
    ),
    Case(
        "YN without question mark",
        "Do you have time",
        compile_pat(r'^YOU HAVE TIME \?$')
    ),
    Case(
        "Negation (structural NOT)",
        "I don’t want to go.",
        compile_pat(r'^ME NOT WANT GO$')
    ),
    # Perfect aspect — allow duplicate FINISH depending on pipeline variants
    Case(
        "Perfect aspect: has + VBN",
        "He has finished the report.",
        compile_pat(r'^HE FINISH (?:FINISH )?REPORT$')
    ),
    Case(
        "Perfect aspect: already + have",
        "I have already eaten.",
        compile_pat(r'^ME FINISH (?:FINISH )?EAT$')
    ),
    # FUTURE: "going to + VB" vs literal "go to"
    Case(
        "Future: going to + VB + time + pm",
        "I’m going to meet John tomorrow at 7 pm.",
        compile_pat(r'^TOMORROW ME (?:BE )?FUTURE (?:MEET )?FS-JOHN 7 PM$')
    ),
    Case(
        "Literal go to (not future MWE)",
        "I’m going to the store now.",
        compile_pat(r'^NOW ME (?:BE )?GO STORE$')
    ),
    # MWEs
    Case(
        "MKE: make a call",
        "She made a call after work.",
        compile_pat(r'^SHE CALL WORK$')
    ),
    Case(
        "MWE: take a photo",
        "We will take a photo at the park.",
        compile_pat(r'^WE FUTURE PHOTO-TAKE PARK$')
    ),
    Case(
        "MWE: long time",
        "I waited long time.",
        compile_pat(r'^ME WAIT LONG-TIME$')
    ),
    # Intensifiers
    Case(
        "Pre-intensifier",
        "I really like that movie.",
        compile_pat(r'^ME LIKE\+\+ MOVIE$')
    ),
    Case(
        "Post-intensifier",
        "He studies a lot.",
        compile_pat(r'^HE STUDY\+\+$')
    ),
    Case(
        "Pre + Post intensifiers",
        "She really likes it a lot.",
        compile_pat(r'^(?:SHE LIKE\+\+\+\+ IT|SHE LIKE\+\+ IT\+\+)$')
    ),
    # Numbers + units, AM/PM, currency, percent, temperature
    Case(
        "Time: 9 am",
        "We arrive at 9 am.",
        compile_pat(r'^WE ARRIVE 9 AM$')
    ),
    Case(
        "Distance: 3 kilometers",
        "The trail is 3 kilometers long.",
        compile_pat(r'^TRAIL (?:BE )?3-KILOMETER LONG$')
    ),
    Case(
        "Money",
        "The ticket costs 20 dollars.",
        compile_pat(r'^TICKET COST 20-DOLLAR$')
    ),
    Case(
        "Temperature + unit + time-front",
        "It will be 30 degrees celsius tomorrow.",
        compile_pat(r'^TOMORROW IT (?:BE )?FUTURE 30-DEGREE CELSIUS$')
    ),
    Case(
        "Percent",
        "Battery is at 15 percent.",
        compile_pat(r'^BATTERY (?:BE )?15-PERCENT$')
    ),
    # Time-fronting
    Case(
        "Day fronting",
        "On Monday, we have a meeting.",
        compile_pat(r'^MONDAY WE HAVE MEETING$')
    ),
    Case(
        "Month fronting + PROPN FS",
        "In October, I travel to Japan.",
        compile_pat(r'^OCTOBER ME TRAVEL FS-JAPAN$')
    ),
    Case(
        "Front NOW + MUST",
        "Now I must leave.",
        compile_pat(r'^NOW ME MUST LEAVE$')
    ),
    Case(
        "Two time anchors",
        "Tonight we pack; tomorrow we fly.",
        compile_pat(r'^TONIGHT TOMORROW WE PACK WE FLY$')
    ),
    # Modals & YN-Q
    Case(
        "Need to -> MUST",
        "You need to study more.",
        compile_pat(r'^YOU MUST STUDY MORE$')
    ),
    Case(
        "Have to -> MUST + NOW front",
        "I have to leave now.",
        compile_pat(r'^NOW ME MUST LEAVE$')
    ),
    Case(
        "Can (YN)",
        "Can you help me",
        compile_pat(r'^YOU CAN HELP ME \?$')
    ),
    # Subordinator no question
    Case(
        "Because ... (not a question)",
        "Because you were late, we missed the train.",
        compile_pat(r'^YOU LATE WE MISS TRAIN$')
    ),
    Case(
        "When ... (not a question)",
        "When the movie ends, we go home.",
        compile_pat(r'^MOVIE END WE GO HOME$')
    ),
    # Coordination & relative clause
    Case(
        "Coordination",
        "We cooked dinner and watched a movie.",
        compile_pat(r'^WE COOK DINNER WATCH MOVIE$')
    ),
    Case(
        "Relative clause (known limitation)",
        "The person who called me yesterday is my teacher.",
        compile_pat(r'^YESTERDAY PERSON CALL ME (?:TEACHER ME|ME TEACHER)$'),
        xfail=True
    ),
    # Proper nouns & unknown words
    Case(
        "Proper noun with apostrophe (FS)",
        "I met O’Connor yesterday.",
        compile_pat(r'^YESTERDAY ME MEET FS-OCONNOR$')
    ),
    Case(
        "Unknown common noun (FS)",
        "I like foobarbaz.",
        compile_pat(r'^ME LIKE FS-FOOBARBAZ$')
    ),
    # Ambiguous FUTURE
    Case(
        "Going to play (future MWE)",
        "We are going to play soccer.",
        compile_pat(r'^WE (?:BE )?FUTURE (?:PLAY )?SOCCER$')
    ),
    Case(
        "Will + verb + time-front",
        "We will play soccer tomorrow.",
        compile_pat(r'^TOMORROW WE FUTURE PLAY SOCCER$')
    ),
    # Mixed AM/PM + duration
    Case(
        "5 pm and lasts 30 minutes",
        "The meeting is at 5 pm and lasts 30 minutes.",
        compile_pat(r'^MEETING (?:BE )?5 PM LAST 30-MINUTE$')
    ),
    # Mixed WH + subordinator (statement)
    Case(
        "Tell me when you arrive (statement)",
        "Tell me when you arrive.",
        compile_pat(r'^TELL ME YOU ARRIVE$')
    ),
    # Safety / imperative-ish
    Case(
        "Make sure everyone wears a seatbelt",
        "Before you start the engine, make sure everyone is wearing a seatbelt.",
        compile_pat(r'^YOU START ENGINE MAKE SURE EVERYONE WEAR SEATBELT$')
    ),
]


# ---------- Runner ----------

def run_case(glosser: ASLGlosser, case: Case, show_tokens: bool = False) -> (bool, str):
    try:
        res = glosser.gloss(case.text)
    except Exception as e:
        return False, f"[{case.name}] RUNTIME ERROR: {e}"

    gloss = normalize_gloss(res.gloss)
    ok = bool(case.pattern.fullmatch(gloss))

    # Build message
    if ok:
        msg = f"[PASS] {case.name} -> {gloss}"
    else:
        msg = (f"[{'XFAIL' if case.xfail else 'FAIL'}] {case.name}\n" +
               pretty_diff(case.text, case.pattern.pattern, gloss))

    if show_tokens:
        msg += f"\n  tokens: {res.gloss_tokens}"

    # If marked xfail, do not fail the run
    return ok, msg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data",
                    help="Path to data dir containing lexicon.json and config.json (default: ./data)")
    ap.add_argument("--filter", default="", help="Run only tests whose name contains this substring (case-insensitive)")
    ap.add_argument("--show-tokens", action="store_true", help="Show gloss_tokens for each case")
    args = ap.parse_args()

    # Initialize glosser once
    glosser = ASLGlosser(args.data)

    selected = [c for c in RAW_CASES if args.filter.lower() in c.name.lower()]
    if not selected:
        print("No tests selected (check --filter?). Exiting.")
        sys.exit(2)

    print(f"Running {len(selected)} test(s) using data dir: {args.data}\n")

    passes = 0
    xpasses = 0
    fails = 0
    xfails = 0

    for case in selected:
        ok, msg = run_case(glosser, case, show_tokens=args.show_tokens)
        if ok and not case.xfail:
            passes += 1
        elif ok and case.xfail:
            xpasses += 1  # unexpected pass
        elif not ok and case.xfail:
            xfails += 1
        else:
            fails += 1
        print(msg)

    total = len(selected)
    print("\nSummary:")
    print(f"  Total: {total}")
    print(f"  PASS : {passes}")
    print(f"  FAIL : {fails}")
    print(f"  XFAIL: {xfails}   (known limitations)")
    print(f"  XPASS: {xpasses}  (unexpected passes)")

    # Exit non-zero if there are unexpected failures
    if fails > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
