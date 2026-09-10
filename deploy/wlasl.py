"""Word-level ASL recognition on continuous video, via Modal (serverless GPU).

Pipeline: decode video -> Moryossef BIO-tagging sign segmenter over MediaPipe
Holistic pose -> drop short windows and widen short nuclei -> classify each
window with WLASL-1000 I3D, restricted to the supported conversational vocabulary
-> confidence-gate and collapse repeats -> ordered gloss sequence. The gloss
sequence is handed to an LLM downstream to produce fluent English.

Recognition is closed to SUPPORTED_VOCAB: signs outside the WLASL-1000 label set
(for example the pronoun "I", which the model does not contain) cannot be
recognized and are recovered by the LLM from sentence context.
"""

import modal

WEIGHTS_GDRIVE = "1jALimVOB69ifYkeT0Pe297S1z4U3jC48"

# Conversational signs verified present in the WLASL-1000 label set. Recognition
# is restricted to this set so co-articulated transition frames cannot resolve to
# arbitrary out-of-domain glosses; the downstream LLM restores dropped pronouns,
# articles and tense.
SUPPORTED_VOCAB = [
    "learn", "sign", "can", "talk", "with", "deaf", "people", "hello", "how",
    "you", "name", "meet", "help", "want", "understand", "yes", "no", "good",
    "please", "sorry", "friend", "work", "nice", "fine", "what", "who", "love",
    "know", "need", "go", "eat", "drink", "happy", "school", "home", "family",
    "teacher", "student", "book", "read", "write", "language", "world", "time",
    "day", "today", "tomorrow", "week", "year", "many", "more", "different",
]

# Window post-processing (frames at 25fps). The segmenter separates distinct
# signs well; merging across gaps chains neighbours and dilutes both below the
# gate, so windows are kept as segmented and filtered only by length and
# classifier confidence.
MIN_LEN = 6       # drop windows shorter than this - transition, not a sign nucleus
MIN_NUCLEUS = 24  # widen windows shorter than this to a full sign nucleus
GATE = 0.40       # keep a window's gloss only above this closed-set probability

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "unzip")
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0", "opencv-contrib-python==4.11.0.86",
        "numpy==1.26.4", "gdown", "scipy",
    )
    .pip_install("pose-format[mediapipe]")
    .pip_install("git+https://github.com/sign-language-processing/segmentation")
    .run_commands("git clone https://github.com/dxli94/WLASL /wlasl")
)

app = modal.App("wlasl-i3d")
vol = modal.Volume.from_name("wlasl-weights", create_if_missing=True)


@app.function(image=image, volumes={"/weights": vol}, timeout=1800)
def prepare():
    import glob
    import os
    import subprocess
    if not os.path.exists("/weights/ready.txt"):
        subprocess.run(["gdown", WEIGHTS_GDRIVE, "-O", "/weights/w.zip"], check=True)
        subprocess.run(["unzip", "-o", "/weights/w.zip", "-d", "/weights/"], check=True)
        open("/weights/ready.txt", "w").write("ok")
        vol.commit()
    pts = glob.glob("/weights/**/*.pt", recursive=True) + glob.glob("/weights/**/*.pth", recursive=True)
    print("PT FILES:", pts)


@app.cls(image=image, gpu="a10g", cpu=8.0, volumes={"/weights": vol}, scaledown_window=300, timeout=900, min_containers=1)
class WLASL:
    @modal.enter()
    def load(self):
        import glob
        import sys

        import torch
        sys.path.insert(0, "/wlasl/code/I3D")
        from pytorch_i3d import InceptionI3d

        self.torch = torch
        self.device = torch.device("cuda")
        # asl1000 outputs 1000 logits; the class list is frequency-ordered, so the
        # first 1000 glosses are exactly this model's label set. Truncate to match.
        self.glosses = [
            line.rstrip("\n").split("\t")[-1]
            for line in open("/wlasl/code/I3D/preprocess/wlasl_class_list.txt")
            if line.strip()
        ][:1000]
        # asl1000 covers our target vocab (all indices <= 999) at ~47% top-1,
        # markedly better than the full asl2000 model (~32%).
        i3d = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(1000)
        pts = [p for p in glob.glob("/weights/**/*.pt", recursive=True) if "1000" in p] or \
              glob.glob("/weights/**/*.pt", recursive=True)
        state = torch.load(sorted(pts)[0], map_location="cpu")
        i3d.load_state_dict(state)
        i3d.to(self.device).eval()
        self.model = i3d

        # Pose extraction (MediaPipe Holistic) is CPU-bound and dominates latency;
        # run it across a worker pool sized to the container's cores.
        import os
        self.pose_workers = max(1, (os.cpu_count() or 8) - 2)
        # Warm the segmentation model weights so the first request does not pay the
        # one-time load.
        from sign_language_segmentation.bin import load_model, resolve_model_path
        load_model(model_dir=resolve_model_path())
        print(f"WLASL I3D loaded ({len(self.glosses)} glosses), "
              f"pose_workers={self.pose_workers}, weights={sorted(pts)[0]}")

    @modal.method()
    def ping(self):
        """No-op that routes to the warm container, resetting its idle timer so
        it stays loaded (avoids a cold start on the next real request)."""
        return "warm"

    def _classify(self, frames, vocab=None):
        """frames: list of HxWx3 BGR uint8. Returns top-5 (gloss, prob).
        vocab: optional list of allowed glosses - softmax is taken over only
        those logits (closed-set re-ranking for a known conversational phrasebook)."""
        import cv2
        import numpy as np
        import torch
        if len(frames) < 9:  # I3D needs enough temporal depth
            frames = frames + [frames[-1]] * (9 - len(frames))
        proc = []
        for img in frames:
            h, w = img.shape[:2]
            sc = 256.0 / min(h, w)
            img = cv2.resize(img, (int(round(w * sc)), int(round(h * sc))))
            img = (img / 255.0) * 2 - 1
            H, W = img.shape[:2]
            y, x = (H - 224) // 2, (W - 224) // 2
            proc.append(img[y:y + 224, x:x + 224])
        arr = np.asarray(proc, dtype=np.float32)              # T,224,224,3
        t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)  # 1,3,T,224,224
        with torch.no_grad():
            logits = self.model(t)                            # 1, 2000, T'
            logits = logits.mean(dim=2)[0]
            if vocab:
                idx = [self.glosses.index(g) for g in vocab if g in self.glosses]
                sub = torch.softmax(logits[idx], 0)
                order = torch.argsort(sub, descending=True)
                return [(self.glosses[idx[i]], round(float(sub[i]), 3)) for i in order.tolist()]
            probs = torch.softmax(logits, 0)
        top = torch.topk(probs, 5)
        return [(self.glosses[i], round(float(probs[i]), 3)) for i in top.indices.tolist()]

    def _segment(self, frames, fps):
        """Sign-boundary detection: Moryossef BIO-tagging segmenter over
        MediaPipe Holistic pose (sign-language-processing/segmentation). Pose is
        extracted in-process across a pool of Holistic workers (the dominant cost),
        reusing the already-decoded frames rather than re-running the CLI decoder."""
        import cv2
        from pose_format.utils.holistic import load_holistic
        from sign_language_segmentation.bin import segment_pose
        h, w = frames[0].shape[:2]
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        # static_image_mode=True makes each frame independent: the Holistic workers
        # are then thread-safe and the result is deterministic. Interleaving frames
        # across workers in tracking mode corrupts MediaPipe's per-frame tracking
        # state, which is both non-reproducible and less accurate.
        pose = load_holistic(
            rgb, fps=fps, width=w, height=h, pose_workers=self.pose_workers,
            additional_holistic_config={"static_image_mode": True},
        )
        _eaf, tiers = segment_pose(pose)
        signs = tiers.get("SIGN", []) if isinstance(tiers, dict) else []
        out = []
        for s in signs:
            a = s["start"] if isinstance(s, dict) else s.start
            b = s["end"] if isinstance(s, dict) else s.end
            out.append((int(a), int(b)))
        return out

    @modal.method()
    def classify_clip(self, video_bytes):
        """Whole-clip classification (no segmentation) - measures classifier ceiling."""
        import subprocess
        import tempfile

        import cv2
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(video_bytes)
            raw = f.name
        path = raw + ".mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw, "-vf", "fps=25", "-c:v", "libx264",
             "-preset", "ultrafast", "-an", path],
            check=True, capture_output=True,
        )
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        return {"top5": self._classify(frames), "num_frames": len(frames)}

    def _clean_windows(self, segs, n_frames):
        """Clip to bounds and drop windows too short to be a sign nucleus."""
        segs = sorted((max(0, a), min(n_frames, b)) for a, b in segs if b > a)
        return [(a, b) for a, b in segs if b - a >= MIN_LEN]

    @modal.method()
    def recognize(self, video_bytes, vocab="__supported__"):
        import subprocess
        import tempfile

        import cv2
        if vocab == "__supported__":
            vocab = SUPPORTED_VOCAB
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(video_bytes)
            raw = f.name
        path = raw + ".mp4"
        # Normalize to constant 25fps so pose frame indices and decoded frames align.
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw, "-vf", "fps=25", "-c:v", "libx264",
             "-preset", "ultrafast", "-an", path],
            check=True, capture_output=True,
        )
        # WLASL I3D was trained on cv2 BGR frames - keep BGR for the classifier.
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()

        windows = self._clean_windows(self._segment(frames, 25), len(frames))
        gloss, detail = [], []
        for (a, b) in windows:
            # A short window is usually a boundary the segmenter cut through the
            # middle of a sign; widen it toward its centre so the classifier sees
            # the whole sign nucleus.
            if b - a < MIN_NUCLEUS:
                c = (a + b) // 2
                a2 = max(0, c - MIN_NUCLEUS // 2)
                b2 = min(len(frames), c + MIN_NUCLEUS // 2)
            else:
                a2, b2 = a, b
            top = self._classify(frames[a2:b2], vocab=vocab)
            g, p = top[0]
            kept = p >= GATE
            if kept and (not gloss or gloss[-1] != g):
                gloss.append(g)
            detail.append({"seg": [a, b], "kept": kept, "top5": top[:5]})
        return {
            "gloss": gloss,
            "num_windows": len(windows),
            "detail": detail,
            "num_frames": len(frames),
        }


@app.local_entrypoint()
def main(video: str):
    with open(video, "rb") as f:
        data = f.read()
    import json
    print("RESULT:", json.dumps(WLASL().recognize.remote(data), indent=1))




@app.local_entrypoint()
def ceiling(folder: str):
    """Classify each isolated *.mp4 in a folder to measure classifier accuracy."""
    import json
    import os
    w = WLASL()
    files = sorted(f for f in os.listdir(folder) if f.endswith(".mp4"))
    out = {}
    for fn in files:
        with open(os.path.join(folder, fn), "rb") as f:
            out[fn] = w.classify_clip.remote(f.read())["top5"]
    print("CEILING:", json.dumps(out, indent=1))
