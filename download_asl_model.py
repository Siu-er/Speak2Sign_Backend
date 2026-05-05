"""Download the Kaggle GISLR 1st place ASL recognition model from HuggingFace."""

import os
import urllib.request
import sys

MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "models")
REPO_BASE = "https://huggingface.co/sign/kaggle-asl-signs-1st-place/resolve/main"

FILES = {
    "model.tflite": f"{REPO_BASE}/model.tflite",
    "sign_to_prediction_index_map.json": f"{REPO_BASE}/sign_to_prediction_index_map.json",
}


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        print(f"  Already exists: {dest_path} ({size / 1024:.1f} KB)")
        return False

    print(f"  Downloading: {url}")
    print(f"  Saving to:   {dest_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print()
    return True


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Model directory: {MODEL_DIR}\n")

    for filename, url in FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        print(f"[{filename}]")
        try:
            downloaded = download_file(url, dest)
            if downloaded:
                size = os.path.getsize(dest)
                print(f"  Done: {size / 1024:.1f} KB")
        except Exception as e:
            print(f"  ERROR: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            return 1
        print()

    print("All model files ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
