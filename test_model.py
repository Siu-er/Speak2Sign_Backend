"""Test ASL recognition model precision.

Two modes:
  python test_model.py video <path> [--expected hello]
  python test_model.py webcam --expected hello [--duration 2.0]

Extracts MediaPipe Holistic landmarks (543 per frame), POSTs to
/sign-to-text endpoint, prints top-5 predictions vs expected label.
"""

import argparse
import json
import math
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import requests

BACKEND_URL = "http://localhost:5000/sign-to-text"

FACE_LANDMARKS = 468
LEFT_HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
RIGHT_HAND_LANDMARKS = 21
TOTAL = FACE_LANDMARKS + LEFT_HAND_LANDMARKS + POSE_LANDMARKS + RIGHT_HAND_LANDMARKS  # 543


def extract_frame_landmarks(results) -> np.ndarray:
    """Extract 543 landmarks in canonical order. NaN for missing."""
    frame = np.full((TOTAL, 3), np.nan, dtype=np.float32)

    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark[:FACE_LANDMARKS]):
            frame[i] = [lm.x, lm.y, lm.z]

    lh_off = FACE_LANDMARKS
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:LEFT_HAND_LANDMARKS]):
            frame[lh_off + i] = [lm.x, lm.y, lm.z]

    pose_off = FACE_LANDMARKS + LEFT_HAND_LANDMARKS
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:POSE_LANDMARKS]):
            frame[pose_off + i] = [lm.x, lm.y, lm.z]

    rh_off = FACE_LANDMARKS + LEFT_HAND_LANDMARKS + POSE_LANDMARKS
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:RIGHT_HAND_LANDMARKS]):
            frame[rh_off + i] = [lm.x, lm.y, lm.z]

    return frame


def landmarks_to_json(frames: list) -> list:
    """Convert list of (543, 3) arrays to JSON-safe nested list with null for NaN."""
    out = []
    for frame in frames:
        frame_data = []
        for i in range(TOTAL):
            x, y, z = float(frame[i, 0]), float(frame[i, 1]), float(frame[i, 2])
            frame_data.append([
                None if math.isnan(x) else x,
                None if math.isnan(y) else y,
                None if math.isnan(z) else z,
            ])
        out.append(frame_data)
    return out


def predict(frames: list, expected: str = None):
    """POST landmarks to backend, print result."""
    if not frames:
        print("ERROR: no frames captured")
        return

    print(f"\nFrames captured: {len(frames)}")

    # Count detection rates
    hand_detected = 0
    for f in frames:
        lh = not np.all(np.isnan(f[FACE_LANDMARKS:FACE_LANDMARKS + LEFT_HAND_LANDMARKS]))
        rh = not np.all(np.isnan(f[FACE_LANDMARKS + LEFT_HAND_LANDMARKS + POSE_LANDMARKS:]))
        if lh or rh:
            hand_detected += 1
    print(f"Frames with hands detected: {hand_detected}/{len(frames)} ({100*hand_detected/len(frames):.0f}%)")

    payload = {"landmarks": landmarks_to_json(frames)}
    payload_size = len(json.dumps(payload))
    print(f"Payload size: {payload_size/1024:.1f} KB")

    t0 = time.time()
    resp = requests.post(BACKEND_URL, json=payload, timeout=30)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"ERROR {resp.status_code}: {resp.text}")
        return

    result = resp.json()
    print(f"Latency: {elapsed*1000:.0f}ms")
    print(f"\nTop 5 predictions:")
    for i, p in enumerate(result["top_5"]):
        marker = "  <- expected" if expected and p["sign"] == expected else ""
        rank = "*" if i == 0 else " "
        print(f"  {rank} {p['sign']:20s} {p['confidence']*100:5.2f}%{marker}")

    if expected:
        ranks = [p["sign"] for p in result["top_5"]]
        if expected in ranks:
            print(f"\nExpected '{expected}' rank: {ranks.index(expected) + 1}/5")
        else:
            print(f"\nExpected '{expected}' NOT in top 5")


def from_video(video_path: str, expected: str = None):
    """Extract landmarks from every frame of video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video: {video_path}, fps: {fps:.1f}")

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)
        frames.append(extract_frame_landmarks(results))

    cap.release()
    holistic.close()
    predict(frames, expected)


def from_webcam(duration: float, expected: str = None):
    """Capture from webcam for N seconds with countdown."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("\nGet ready...")
    for i in [3, 2, 1]:
        print(f"  {i}")
        time.sleep(1)
    print(f"GO! (sign for {duration}s)")

    frames = []
    t_start = time.time()
    last_print = t_start
    while time.time() - t_start < duration:
        ok, img = cap.read()
        if not ok:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)
        frames.append(extract_frame_landmarks(results))

        # Visual feedback
        cv2.imshow("Capturing... (press q to abort)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Tick every 0.5s
        if time.time() - last_print > 0.5:
            remaining = duration - (time.time() - t_start)
            print(f"  ...{remaining:.1f}s")
            last_print = time.time()

    print("DONE")
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    predict(frames, expected)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    pv = sub.add_parser("video")
    pv.add_argument("path")
    pv.add_argument("--expected", default=None)

    pw = sub.add_parser("webcam")
    pw.add_argument("--duration", type=float, default=2.0)
    pw.add_argument("--expected", default=None)

    args = parser.parse_args()

    if args.mode == "video":
        from_video(args.path, args.expected)
    elif args.mode == "webcam":
        from_webcam(args.duration, args.expected)


if __name__ == "__main__":
    main()
