"""Train an isolated-sign classifier on a prepared landmark dataset and export
a TFLite model plus a recognizer-compatible label map.

Input: the output of prepare_dataset.py (per-label .npy clips + labels.json).
Each clip (frames, 543, 3) is resampled to a fixed length, NaNs zeroed, and fed
to a small masked GRU classifier. Exports:
  <out>/model.tflite
  <out>/sign_to_prediction_index_map.json   (same schema asl_sign_recognizer reads)

Input contract of the exported model: float32 tensor (1, SEQ_LEN, 1629) where
1629 = 543 landmarks * 3 coords. To swap a model trained here into the live
recognizer, either feed it this shape or adapt asl_sign_recognizer.predict to
this signature (see README).

Run from the backend root:
  python tools/train/train_classifier.py <prepared_dir> <out_dir> [--epochs N]
"""

import argparse
import json
import os

import numpy as np

FACE_LH_POSE_RH = 543
FEAT = FACE_LH_POSE_RH * 3  # 1629
SEQ_LEN = 32


def resample(arr, seq_len=SEQ_LEN):
    """(frames, 543, 3) -> (seq_len, 1629), NaN zeroed, linearly resampled."""
    flat = np.nan_to_num(arr.reshape(arr.shape[0], -1), nan=0.0)
    n = flat.shape[0]
    if n == seq_len:
        out = flat
    else:
        idx = np.linspace(0, n - 1, seq_len)
        lo = np.floor(idx).astype(int)
        hi = np.ceil(idx).astype(int)
        frac = (idx - lo)[:, None]
        out = flat[lo] * (1 - frac) + flat[hi] * frac
    return out.astype(np.float32)


def load(prepared_dir):
    labels = json.load(open(os.path.join(prepared_dir, "labels.json")))
    X, y = [], []
    for label, idx in labels.items():
        d = os.path.join(prepared_dir, label)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".npy"):
                X.append(resample(np.load(os.path.join(d, fn))))
                y.append(idx)
    return np.stack(X), np.array(y, dtype=np.int64), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    import tensorflow as tf

    X, y, labels = load(args.prepared_dir)
    num_classes = len(labels)
    print(f"dataset: X={X.shape} y={y.shape} classes={num_classes}")

    # Temporal CNN: converts cleanly to pure TFLite builtins, so the exported
    # model loads on the same tflite runtime as the current recognizer (a GRU
    # path needs SELECT_TF_OPS and does not convert reliably under Keras 3).
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN, FEAT)),
        tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=args.epochs, batch_size=16, verbose=2)

    os.makedirs(args.out_dir, exist_ok=True)
    # Keras 3 converts to TFLite via an exported SavedModel; from_keras_model
    # crashes in the MLIR pass on this stack.
    saved = os.path.join(args.out_dir, "saved_model")
    model.export(saved)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved)
    tflite_model = converter.convert()
    with open(os.path.join(args.out_dir, "model.tflite"), "wb") as f:
        f.write(tflite_model)
    with open(os.path.join(args.out_dir, "sign_to_prediction_index_map.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Exported model.tflite + label map ({num_classes} classes) -> {args.out_dir}")


if __name__ == "__main__":
    main()
