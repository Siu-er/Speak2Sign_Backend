"""SHuBERT ASL-video-to-English translation on Modal (serverless GPU).

Runs the TTIC SHuBERT Space's inference code inside a Modal container:
MediaPipe holistic -> hand/face crops -> DINOv2 embeddings + pose -> ByT5 decoder.
Models load once per warm container (@enter) and are reused across calls, so
only extraction + generation run per request. Weights come from the public
ShesterG/SHuBERT repo into a persisted Volume.
"""

import modal

SPACE_REPO = "ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText"
WEIGHTS_REPO = "ShesterG/SHuBERT"
M = "/weights/models"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands("pip install pip==23.1.2")
    .pip_install("numpy==1.24.3", "torch==2.2.0", "torchvision", "torchaudio")
    .pip_install(
        "transformers==4.30.2", "tokenizers==0.13.3", "huggingface-hub==0.28.1",
        "accelerate==1.0.1", "datasets==3.1.0", "decord==0.6.0",
        "opencv-contrib-python==4.11.0.86", "scipy==1.10.1", "einops==0.8.0",
        "mediapipe==0.10.11", "pandas==2.0.3", "matplotlib==3.7.5",
        "jiwer==3.1.0", "evaluate==0.4.3", "cffi", "omegaconf",
    )
    .pip_install("fairseq==0.12.2")
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/{SPACE_REPO} /app"
    )
)

app = modal.App("shubert-asl")
weights = modal.Volume.from_name("shubert-weights", create_if_missing=True)


@app.function(image=image, volumes={"/weights": weights}, timeout=3600)
def prepare_weights():
    import os
    from huggingface_hub import snapshot_download
    if not os.path.exists(f"{M}/byt5_base/config.json"):
        snapshot_download(
            WEIGHTS_REPO,
            allow_patterns="models/*",
            ignore_patterns=["*optimizer.pt", "*rng_state*", "*scheduler.pt", "*trainer_state*"],
            local_dir="/weights",
        )
        weights.commit()
    print("weights ready:", os.listdir(M))


def _patch_dinov2():
    """Download DINOv2 via torch.hub then swap in the Space's compat-patched files."""
    import os, sys, shutil, torch
    try:
        torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", pretrained=False)
    except Exception as e:
        print("dinov2 preload note:", e)
    layers = "/root/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers"
    if os.path.isdir(layers):
        for fn in ("attention.py", "block.py"):
            if os.path.exists(f"/app/{fn}"):
                shutil.copy(f"/app/{fn}", f"{layers}/{fn}")
        for m in [k for k in list(sys.modules) if "dinov2" in k]:
            del sys.modules[m]


@app.cls(image=image, gpu="a10g", cpu=8.0, volumes={"/weights": weights}, scaledown_window=300, timeout=900)
class SHuBERT:
    @modal.enter()
    def load(self):
        import os, sys, torch
        os.environ["TORCH_HOME"] = "/root/.cache/torch"
        os.environ["HF_HOME"] = "/root/.cache/huggingface"
        os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
        os.makedirs("/tmp/mpl", exist_ok=True)
        sys.path.insert(0, "/app")
        os.chdir("/app")

        _patch_dinov2()

        from dinov2_features import DINOEmbedder
        from inference import SignLanguageByT5ForConditionalGeneration
        from transformers import ByT5Tokenizer
        from kpe_mediapipe import HolisticDetector

        self.torch = torch
        self.device = torch.device("cuda")
        # Heavy models loaded once, reused across calls.
        self.dino_hand = DINOEmbedder(f"{M}/dinov2hand.pth")
        self.dino_face = DINOEmbedder(f"{M}/dinov2face.pth")
        self.model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
            f"{M}/checkpoint-11625", cache_dir="/tmp/cache"
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = ByT5Tokenizer.from_pretrained(f"{M}/byt5_base")
        # A pool of independent MediaPipe detectors so frames run in parallel
        # threads (the per-frame landmarking is the dominant cost).
        self.n_workers = 6
        self.detectors = [
            HolisticDetector(
                f"{M}/face_landmarker_v2_with_blendshapes.task",
                f"{M}/hand_landmarker.task",
            )
            for _ in range(self.n_workers)
        ]
        print("SHuBERT models + detector pool loaded and cached")

    @modal.method()
    def translate(self, video_bytes: bytes) -> str:
        import tempfile, decord, torch, numpy as np
        from concurrent.futures import ThreadPoolExecutor
        from crop_hands import HandExtractor
        from crop_face import FaceExtractor
        from body_features import process_pose_landmarks

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            path = f.name

        vr = decord.VideoReader(path)
        # SHuBERT trains at ~12fps; MediaPipe over every frame is the cost, so
        # downsample to that rate - faster and in-distribution.
        fps = vr.get_avg_fps() or 24.0
        stride = max(1, round(fps / 12.0))
        frames = vr.get_batch(list(range(0, len(vr), stride))).asnumpy()

        # Landmark every frame in parallel across the cached detector pool; the
        # MediaPipe graphs release the GIL during C++ inference, so threads run
        # concurrently. Merge back into a frame-indexed dict (same shape the
        # Space's video_holistic returns).
        splits = np.array_split(np.arange(len(frames)), self.n_workers)

        def _work(w):
            det = self.detectors[w]
            out = {}
            for j in splits[w]:
                _, lm = det.detect_frame_landmarks(frames[int(j)])
                out[int(j)] = lm
            return out

        landmarks = {}
        with ThreadPoolExecutor(self.n_workers) as ex:
            for part in ex.map(_work, range(self.n_workers)):
                landmarks.update(part)

        hand_ex = HandExtractor()
        left_frames, right_frames = hand_ex.extract_hand_frames(frames, landmarks)
        left_emb = self.dino_hand.extract_embeddings_from_frames(left_frames)
        right_emb = self.dino_hand.extract_embeddings_from_frames(right_frames)

        face_ex = FaceExtractor()
        face_frames = face_ex.extract_face_frames(frames, landmarks)
        face_emb = self.dino_face.extract_embeddings_from_frames(face_frames)

        pose_emb = process_pose_landmarks(landmarks)

        d = self.device
        face_t = torch.tensor(face_emb, dtype=torch.float32).unsqueeze(0).to(d)
        left_t = torch.tensor(left_emb, dtype=torch.float32).unsqueeze(0).to(d)
        right_t = torch.tensor(right_emb, dtype=torch.float32).unsqueeze(0).to(d)
        pose_t = torch.tensor(pose_emb, dtype=torch.float32).unsqueeze(0).to(d)

        with torch.no_grad():
            generated = self.model.generate(
                face_features=face_t,
                left_hand_features=left_t,
                right_hand_features=right_t,
                pose_features=pose_t,
                max_length=2048,
                num_beams=5,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


@app.local_entrypoint()
def main(video: str):
    with open(video, "rb") as f:
        data = f.read()
    print("TRANSLATION:", SHuBERT().translate.remote(data))
