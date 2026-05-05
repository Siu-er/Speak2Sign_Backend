"""Offline continuous recognition simulator.

Replays a video through MediaPipe + the recognition engine state machine
(reimplemented in Python here, mirroring the TS engine logic) and prints
the sequence of emitted signs.

Usage:
  python test_continuous.py video <path>
  python test_continuous.py videos test_videos/  # batch all *.mp4 in dir
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import requests

BACKEND_URL = "http://localhost:5000/sign-to-text"

FACE = 468
LH = 21
POSE = 33
RH = 21
TOTAL = FACE + LH + POSE + RH  # 543

LH_OFF = FACE
RH_OFF = FACE + LH + POSE


@dataclass
class Config:
    buffer_size: int = 48
    # Sync test runs blocking inference; use larger interval to keep test fast
    # In live browser, engine drops ticks while inference in flight (effectively async)
    inference_interval_frames: int = 15
    ema_alpha: float = 0.4
    confidence_threshold: float = 0.5
    stability_ticks: int = 3
    early_break_confidence: float = 0.85
    hand_presence_ratio: float = 0.5
    idle_hands_gone_ms: int = 500
    watching_hands_present_ms: int = 200
    cooldown_ms: int = 500
    rearm_low_confidence: float = 0.3
    rearm_low_confidence_ms: int = 150
    target_fps: int = 30  # for timestamp simulation


@dataclass
class Emission:
    sign: str
    confidence: float
    class_index: int
    timestamp: float


class ContinuousRecognitionEngine:
    """Python mirror of the TS RecognitionEngine for offline testing."""

    def __init__(self, config: Config, inference_fn: Callable, label_map: Dict[int, str]):
        self.config = config
        self.inference_fn = inference_fn
        self.label_map = label_map
        self.num_classes = len(label_map)

        self.buffer: deque = deque(maxlen=config.buffer_size)
        self.hand_flags: deque = deque(maxlen=config.buffer_size)

        self.ema: Optional[np.ndarray] = None
        self.state = "IDLE"
        self.frames_since_inference = 0

        self.hands_present_since: Optional[float] = None
        self.hands_absent_since: Optional[float] = None

        self.current_argmax = -1
        self.stable_ticks = 0

        self.last_emit_idx: Optional[int] = None
        self.last_emit_below_rearm_since: Optional[float] = None
        self.cooldown_start: float = 0

        self.emissions: List[Emission] = []
        self.last_ts: float = 0
        self.debug_log: List[Dict] = []

    def _ema_update(self, probs: np.ndarray):
        if self.ema is None:
            self.ema = probs.copy()
        else:
            a = self.config.ema_alpha
            self.ema = a * probs + (1 - a) * self.ema

    def _ema_argmax(self) -> Tuple[int, float]:
        if self.ema is None:
            return -1, 0.0
        idx = int(np.argmax(self.ema))
        return idx, float(self.ema[idx])

    def _hand_presence_ratio(self) -> float:
        if not self.hand_flags:
            return 0
        return sum(self.hand_flags) / len(self.hand_flags)

    def _set_state(self, new_state: str):
        if new_state != self.state:
            # print(f"  STATE: {self.state} -> {new_state}")
            self.state = new_state

    def _to_idle(self):
        self._set_state("IDLE")
        self.buffer.clear()
        self.hand_flags.clear()
        self.ema = None
        self.current_argmax = -1
        self.stable_ticks = 0
        self.last_emit_idx = None

    def push_frame(self, frame: np.ndarray, has_hands: bool, ts: float):
        self.last_ts = ts
        self.buffer.append(frame.copy())
        self.hand_flags.append(1 if has_hands else 0)
        self.frames_since_inference += 1

        # Hand presence timers
        if has_hands:
            self.hands_absent_since = None
            if self.hands_present_since is None:
                self.hands_present_since = ts
        else:
            self.hands_present_since = None
            if self.hands_absent_since is None:
                self.hands_absent_since = ts

        self._run_state_machine(ts)

        if self._should_run_inference():
            self.frames_since_inference = 0
            self._run_inference(ts)

    def _run_state_machine(self, ts: float):
        # Universal: hands gone too long
        if self.state != "IDLE" and self.hands_absent_since is not None:
            if ts - self.hands_absent_since >= self.config.idle_hands_gone_ms / 1000:
                self._to_idle()
                return

        if self.state == "IDLE":
            if self.hands_present_since is not None:
                if ts - self.hands_present_since >= self.config.watching_hands_present_ms / 1000:
                    self.buffer.clear()
                    self.hand_flags.clear()
                    self.ema = None
                    self.current_argmax = -1
                    self.stable_ticks = 0
                    self._set_state("WATCHING")

        elif self.state == "WATCHING":
            if (len(self.buffer) >= self.config.buffer_size
                    and self._hand_presence_ratio() >= self.config.hand_presence_ratio):
                self._set_state("INFERRING")

        elif self.state == "EMITTING":
            self.cooldown_start = ts
            self._set_state("COOLDOWN")

        elif self.state == "COOLDOWN":
            # Cooldown is a minimum quiet period; transition out via re-arm
            # signal in _run_inference, not auto-timer.
            pass

    def _should_run_inference(self) -> bool:
        if self.state not in ("INFERRING", "COOLDOWN"):
            return False
        if len(self.buffer) < self.config.buffer_size:
            return False
        if self._hand_presence_ratio() < self.config.hand_presence_ratio:
            return False
        return self.frames_since_inference >= self.config.inference_interval_frames

    def _run_inference(self, ts: float):
        frames = list(self.buffer)
        try:
            probs = self.inference_fn(frames)
        except Exception as e:
            print(f"  INFERENCE ERROR: {e}")
            return

        self._ema_update(probs)
        idx, prob = self._ema_argmax()

        if idx == self.current_argmax:
            self.stable_ticks += 1
        else:
            self.current_argmax = idx
            self.stable_ticks = 1

        # Top-3 for debug
        top3 = np.argsort(self.ema)[-3:][::-1]
        debug_top = [(self.label_map[int(i)], float(self.ema[i])) for i in top3]
        self.debug_log.append({
            "ts": ts,
            "state": self.state,
            "argmax": self.label_map.get(idx, "?"),
            "prob": prob,
            "stable": self.stable_ticks,
            "top3": debug_top,
        })

        if self.state == "INFERRING":
            if (prob >= self.config.confidence_threshold
                    and self.stable_ticks >= self.config.stability_ticks):
                self._emit(idx, prob, ts)
        elif self.state == "COOLDOWN" and self.last_emit_idx is not None:
            cooldown_elapsed = ts - self.cooldown_start >= self.config.cooldown_ms / 1000
            same = idx == self.last_emit_idx
            if not same and prob >= self.config.early_break_confidence:
                self._emit(idx, prob, ts)
                return
            if cooldown_elapsed:
                last_prob = float(self.ema[self.last_emit_idx])
                if last_prob < self.config.rearm_low_confidence:
                    if self.last_emit_below_rearm_since is None:
                        self.last_emit_below_rearm_since = ts
                    elif ts - self.last_emit_below_rearm_since >= self.config.rearm_low_confidence_ms / 1000:
                        self._set_state("WATCHING")
                else:
                    self.last_emit_below_rearm_since = None

    def _emit(self, idx: int, conf: float, ts: float):
        sign = self.label_map.get(idx, f"class_{idx}")
        emission = Emission(sign=sign, confidence=conf, class_index=idx, timestamp=ts)
        self.emissions.append(emission)
        print(f"  >>> EMIT: {sign} ({conf*100:.1f}%) at t={ts:.2f}s")
        self.last_emit_idx = idx
        self.last_emit_below_rearm_since = None
        self.cooldown_start = ts
        self.stable_ticks = 0
        self._set_state("COOLDOWN")


def extract_frame(results) -> Tuple[np.ndarray, bool]:
    frame = np.full((TOTAL, 3), np.nan, dtype=np.float32)
    has_hands = False
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark[:FACE]):
            frame[i] = [lm.x, lm.y, lm.z]
    if results.left_hand_landmarks:
        has_hands = True
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:LH]):
            frame[LH_OFF + i] = [lm.x, lm.y, lm.z]
    if results.pose_landmarks:
        off = FACE + LH
        for i, lm in enumerate(results.pose_landmarks.landmark[:POSE]):
            frame[off + i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        has_hands = True
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:RH]):
            frame[RH_OFF + i] = [lm.x, lm.y, lm.z]
    return frame, has_hands


def landmarks_to_payload(frames: List[np.ndarray]) -> dict:
    out = []
    for f in frames:
        fd = []
        for i in range(TOTAL):
            x, y, z = float(f[i, 0]), float(f[i, 1]), float(f[i, 2])
            fd.append([
                None if math.isnan(x) else x,
                None if math.isnan(y) else y,
                None if math.isnan(z) else z,
            ])
        out.append(fd)
    return {"landmarks": out, "return_all_probs": True}


def make_inference_fn() -> Callable:
    def fn(frames: List[np.ndarray]) -> np.ndarray:
        payload = landmarks_to_payload(frames)
        resp = requests.post(BACKEND_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "all_probs" not in data:
            raise RuntimeError("Backend did not return all_probs")
        return np.array(data["all_probs"], dtype=np.float32)
    return fn


def fetch_label_map() -> Dict[int, str]:
    resp = requests.get("http://localhost:5000/sign-labels", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return {int(k): v for k, v in data["labels"].items()}


def process_video(video_path: str, label_map: Dict[int, str], config: Config) -> List[Emission]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"\nVideo: {video_path} ({total_frames} frames, {fps:.1f}fps, {duration:.1f}s)")

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    inference_fn = make_inference_fn()
    engine = ContinuousRecognitionEngine(config, inference_fn, label_map)

    frame_idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        ts = frame_idx / fps
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        frame, has_hands = extract_frame(results)
        engine.push_frame(frame, has_hands, ts)
        frame_idx += 1

    cap.release()
    holistic.close()

    print(f"\n--- {len(engine.emissions)} emissions ---")
    for em in engine.emissions:
        print(f"  t={em.timestamp:.2f}s  {em.sign}  ({em.confidence*100:.1f}%)")
    return engine.emissions


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    pv = sub.add_parser("video")
    pv.add_argument("path")

    pb = sub.add_parser("videos")
    pb.add_argument("dir")

    args = parser.parse_args()

    config = Config()
    print(f"Fetching label map...")
    label_map = fetch_label_map()
    print(f"Got {len(label_map)} labels")

    if args.mode == "video":
        process_video(args.path, label_map, config)
    elif args.mode == "videos":
        videos = sorted(f for f in os.listdir(args.dir) if f.endswith((".mp4", ".webm")))
        for v in videos:
            expected = os.path.splitext(v)[0]
            print(f"\n{'='*60}\n[{expected}]")
            process_video(os.path.join(args.dir, v), label_map, config)


if __name__ == "__main__":
    main()
