"""Simulate a two-way conversation through the full pipeline.

Turn 1 (speaker side, voice -> sign): a spoken WAV is transcribed by Whisper,
glossed, and converted to SiGML for the signer's avatar.

Turn 2 (signer side, sign -> voice): real sign videos are run through MediaPipe
and the recognizer, and the recognized signs are assembled into an English
sentence the speaker would hear.

Usage:
  python tools/sim_conversation.py <turn1.wav> <reply_video> [<reply_video> ...]
"""

import math
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))
from extract_landmarks import extract_video  # noqa: E402

API = "http://localhost:5000"


def to_payload(arr):
    out = []
    for f in arr:
        out.append([[None if math.isnan(v) else float(v) for v in lm] for lm in f])
    return out


def turn1_voice_to_sign(wav_path):
    print("=" * 64)
    print("TURN 1  speaker speaks  ->  signer sees avatar")
    print("=" * 64)
    with open(wav_path, "rb") as f:
        r = requests.post(f"{API}/audio-to-text",
                          files={"audio": ("turn1.wav", f, "audio/wav")},
                          data={"task": "transcribe"}, timeout=120)
    r.raise_for_status()
    text = r.json().get("text", "").strip()
    print(f"  voice transcribed : {text!r}")

    g = requests.post(f"{API}/text-to-gloss", json={"text": text}, timeout=60).json()
    gloss = g["gloss"]
    print(f"  ASL gloss         : {gloss}")

    s = requests.post(f"{API}/gloss-to-sigml", json={"gloss": gloss}, timeout=60).json()
    kinds = list(zip(s["tokens"], s["token_kinds"]))
    signed = [t for t, k in kinds if k == "sign"]
    spelled = s["fingerspelled"]
    print(f"  rendered as signs : {signed}")
    print(f"  fingerspelled     : {spelled}")
    print(f"  avatar SiGML bytes: {len(s['sigml'])}")
    return text


def turn2_sign_to_voice(videos):
    print("=" * 64)
    print("TURN 2  signer signs on video  ->  speaker hears voice")
    print("=" * 64)
    recognized = []
    for v in videos:
        arr = extract_video(v)
        payload = {"landmarks": to_payload(arr), "return_all_probs": False}
        r = requests.post(f"{API}/sign-to-text", json=payload, timeout=120)
        r.raise_for_status()
        d = r.json()
        top = d.get("top_5", [])[:3]
        sign = d.get("sign")
        recognized.append(sign)
        shown = ", ".join(f"{p['sign']}({p['confidence']:.2f})" for p in top)
        print(f"  {os.path.basename(v):16} -> {sign:12} [top: {shown}]")

    print(f"\n  recognized signs  : {recognized}")
    r = requests.post(f"{API}/sign-to-sentence", json={"signs": recognized}, timeout=60)
    if r.status_code == 200:
        print(f"  spoken to speaker : {r.json().get('sentence')!r}")
    else:
        err = r.json().get("error", r.status_code)
        print(f"  sentence stage    : unavailable ({err})")
        print(f"  (speaker would hear the raw signs: {' '.join(recognized)})")


def main():
    if len(sys.argv) < 3:
        print("usage: python tools/sim_conversation.py <turn1.wav> <reply_video> ...")
        sys.exit(1)
    wav, videos = sys.argv[1], sys.argv[2:]
    turn1_voice_to_sign(wav)
    print()
    turn2_sign_to_voice(videos)


if __name__ == "__main__":
    main()
