"""Extract MediaPipe Holistic landmarks from a video into the canonical
543-landmark layout the recognizer expects.

Layout (must match asl_sign_recognizer and the live frontend extractor):
  face(468) + left_hand(21) + pose(33) + right_hand(21) = 543, each x,y,z.
Missing landmarks are NaN, matching the runtime payload.

Usable standalone (smoke test on a clip) or imported by prepare_dataset.
"""

import math
import sys

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

FACE, LH, POSE, RH = 468, 21, 33, 21
TOTAL = FACE + LH + POSE + RH  # 543
LH_OFF = FACE
POSE_OFF = FACE + LH
RH_OFF = FACE + LH + POSE


def _frame_from_results(results) -> np.ndarray:
    frame = np.full((TOTAL, 3), np.nan, dtype=np.float32)
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark[:FACE]):
            frame[i] = [lm.x, lm.y, lm.z]
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:LH]):
            frame[LH_OFF + i] = [lm.x, lm.y, lm.z]
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:POSE]):
            frame[POSE_OFF + i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:RH]):
            frame[RH_OFF + i] = [lm.x, lm.y, lm.z]
    return frame


def extract_video(video_path: str, max_frames: int = 0) -> np.ndarray:
    """Return an array of shape (num_frames, 543, 3) for one video."""
    if mp is None:
        raise RuntimeError("mediapipe is not installed in this environment")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1, refine_face_landmarks=False
    )
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            frames.append(_frame_from_results(results))
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        holistic.close()
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python extract_landmarks.py <video> [<video> ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        arr = extract_video(path)
        hands = int(np.sum(~np.isnan(arr[:, LH_OFF:LH_OFF + LH, 0]).all(axis=1)))
        print(f"{path}: frames={arr.shape[0]} shape={arr.shape} "
              f"hand_frames={hands} nan_ratio={np.isnan(arr).mean():.2f}")
