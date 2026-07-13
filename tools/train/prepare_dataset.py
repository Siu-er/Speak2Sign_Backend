"""Turn a directory of labelled sign videos into a landmark dataset.

Input layout (one subdirectory per sign, video clips inside):
  <root>/HELLO/clip1.mp4
  <root>/HELLO/clip2.mp4
  <root>/COFFEE/clip1.mp4
  ...

Output: <out>/<LABEL>/<n>.npy   (each an array of shape (frames, 543, 3))
        <out>/labels.json       (sorted label list -> class index)

Works on any labelled video set: the bundled test_videos (smoke test), or a
downloaded corpus such as ASL Citizen or WLASL once arranged into this layout.

Run from the backend root:
  python tools/train/prepare_dataset.py <video_root> <out_dir> [--max-frames N]
"""

import argparse
import json
import os

import numpy as np

from extract_landmarks import extract_video

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_root")
    ap.add_argument("out_dir")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    labels = sorted(
        d for d in os.listdir(args.video_root)
        if os.path.isdir(os.path.join(args.video_root, d))
    )
    if not labels:
        raise SystemExit(f"No label subdirectories under {args.video_root}")

    os.makedirs(args.out_dir, exist_ok=True)
    label_index = {label: i for i, label in enumerate(labels)}
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_index, f, indent=2)

    total = 0
    for label in labels:
        src = os.path.join(args.video_root, label)
        dst = os.path.join(args.out_dir, label)
        os.makedirs(dst, exist_ok=True)
        clips = [c for c in os.listdir(src) if c.lower().endswith(VIDEO_EXTS)]
        for n, clip in enumerate(clips):
            try:
                arr = extract_video(os.path.join(src, clip), max_frames=args.max_frames)
            except Exception as e:
                print(f"  skip {label}/{clip}: {e}")
                continue
            np.save(os.path.join(dst, f"{n}.npy"), arr)
            total += 1
        print(f"{label}: {len(clips)} clips")
    print(f"Prepared {total} clips across {len(labels)} labels -> {args.out_dir}")


if __name__ == "__main__":
    main()
