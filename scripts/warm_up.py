"""Bring every deployed Speak2Sign service to a warm state and report how long each took.

Run from the backend directory with the project virtual environment:
    .venv/Scripts/python.exe scripts/warm_up.py              # warm once
    .venv/Scripts/python.exe scripts/warm_up.py --hold 45    # warm, then keep warm for 45 minutes

Exits non-zero, naming the service, when any service fails to answer.
"""

import argparse
import os
import sys
import time
import urllib.request

BACKEND_URL = os.environ.get("S2S_BACKEND_URL", "https://s2s-backend-b718.onrender.com")
FRONTEND_URL = os.environ.get("S2S_FRONTEND_URL", "https://s2s-app-alpha.vercel.app")
MODAL_APP = "wlasl-i3d"
MODAL_CLASS = "WLASL"
# A sleeping Render free instance took 72.5 s to answer /health after 17 minutes idle.
HTTP_TIMEOUT_S = 180
# Half of the WLASL class's scaledown_window (120 s in deploy/wlasl.py), so the GPU
# container never reaches its idle limit between two rounds.
HOLD_INTERVAL_S = 60


def timed(label, fn):
    start = time.monotonic()
    try:
        result = fn()
    except Exception as exc:
        print(f"FAIL {label}: {exc}", flush=True)
        sys.exit(1)
    print(f"ok   {label}: {result} in {time.monotonic() - start:.1f}s", flush=True)


def http_status(url):
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as resp:
        return resp.status


def warm_round(wlasl):
    timed(f"backend {BACKEND_URL}/health", lambda: http_status(f"{BACKEND_URL}/health"))
    timed(f"frontend {FRONTEND_URL}", lambda: http_status(FRONTEND_URL))
    timed(f"modal {MODAL_APP}.{MODAL_CLASS}.ping", lambda: wlasl.ping.remote())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hold", type=float, default=0, metavar="MINUTES",
                        help="keep every service warm for this many minutes after warming")
    args = parser.parse_args()

    import modal
    wlasl = modal.Cls.from_name(MODAL_APP, MODAL_CLASS)()
    warm_round(wlasl)
    deadline = time.monotonic() + args.hold * 60
    while time.monotonic() + HOLD_INTERVAL_S <= deadline:
        time.sleep(HOLD_INTERVAL_S)
        warm_round(wlasl)


if __name__ == "__main__":
    main()
