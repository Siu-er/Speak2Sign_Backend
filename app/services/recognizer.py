"""ASL video recognition via the WLASL segment-and-classify pipeline on Modal,
plus retention of the most recent clips for debugging."""

import json
import os
import re
import secrets
import shutil

from app import config


def get_wlasl():
    import modal
    return modal.Cls.from_name(config.WLASL_MODAL_APP, config.WLASL_MODAL_CLASS)()


def clip_slug(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "none"


def retain_debug_clip(src_path, ext, labels, meta=None):
    os.makedirs(config.DEBUG_CLIPS_DIR, exist_ok=True)
    label = "-".join(labels) if labels else "none"
    token = secrets.token_hex(4)
    dest = os.path.join(config.DEBUG_CLIPS_DIR, f"clip_{label}_{token}{ext}")
    shutil.copyfile(src_path, dest)
    # Persist the recognition result beside the clip so a run's gloss and
    # per-window scores can be reviewed without re-running (which is not
    # bit-reproducible across pose extraction).
    if meta is not None:
        with open(dest.rsplit(ext, 1)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    entries = [
        os.path.join(config.DEBUG_CLIPS_DIR, f)
        for f in os.listdir(config.DEBUG_CLIPS_DIR)
        if not f.endswith(".json")
    ]
    for old in sorted(entries, key=os.path.getmtime, reverse=True)[config.DEBUG_CLIPS_KEEP:]:
        for p in (old, old.rsplit(".", 1)[0] + ".json"):
            try:
                os.remove(p)
            except OSError:
                pass
