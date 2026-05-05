"""Batch test all videos in test_videos/ folder.
Filename (without .mp4) is the expected sign label.
"""
import os
import subprocess
import sys

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "test_videos")
PYTHON = sys.executable

videos = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
results = []

for fname in videos:
    expected = os.path.splitext(fname)[0]
    path = os.path.join(VIDEO_DIR, fname)
    print(f"\n{'='*60}\n[{expected}] {fname}\n{'='*60}")

    proc = subprocess.run(
        [PYTHON, "test_model.py", "video", path, "--expected", expected],
        capture_output=True, text=True
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-500:])

    # Parse top-1 from output
    top1 = "?"
    rank = "?"
    for line in proc.stdout.splitlines():
        if line.strip().startswith("*"):
            parts = line.strip().split()
            if len(parts) >= 2:
                top1 = parts[1]
        if "rank:" in line:
            rank = line.split("rank:")[1].strip()
        if "NOT in top 5" in line:
            rank = "miss"
    results.append((expected, top1, rank))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Expected':<15} {'Top-1':<15} {'Rank':<10}")
print("-"*40)
hits_top1 = 0
hits_top5 = 0
for exp, top1, rank in results:
    match = "OK" if exp == top1 else ""
    print(f"{exp:<15} {top1:<15} {rank:<10} {match}")
    if exp == top1:
        hits_top1 += 1
    if rank not in ("?", "miss"):
        hits_top5 += 1
print("-"*40)
print(f"Top-1 accuracy: {hits_top1}/{len(results)} ({100*hits_top1/len(results):.0f}%)")
print(f"Top-5 accuracy: {hits_top5}/{len(results)} ({100*hits_top5/len(results):.0f}%)")
