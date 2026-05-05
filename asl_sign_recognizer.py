"""ASL sign recognition using the Kaggle GISLR 1st place TFLite model.

Model: sign/kaggle-asl-signs-1st-place (HuggingFace)
Input: MediaPipe Holistic landmarks (N frames x 543 landmarks x 3 coords)
Output: Predicted sign label from 250-sign vocabulary
"""

import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

# Canonical landmark ordering (must match model training):
# face: 468 landmarks, left_hand: 21, pose: 33, right_hand: 21 = 543 total
LANDMARKS_PER_FRAME = 543
COORDS_PER_LANDMARK = 3


class ASLSignRecognizer:

    def __init__(self, model_dir):
        model_path = os.path.join(model_dir, "model.tflite")
        label_path = os.path.join(model_dir, "sign_to_prediction_index_map.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label map not found: {label_path}")

        try:
            import tensorflow.lite as tflite
        except ImportError:
            import tflite_runtime.interpreter as tflite

        logger.info("Loading ASL sign recognition model from %s", model_path)
        start = time.time()

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.runner = self.interpreter.get_signature_runner("serving_default")

        with open(label_path, "r") as f:
            sign_to_idx = json.load(f)
        self.idx_to_sign = {int(v): k for k, v in sign_to_idx.items()}
        self.num_signs = len(self.idx_to_sign)

        elapsed = time.time() - start
        logger.info(
            "ASL sign recognizer loaded in %.2fs (%d signs)",
            elapsed, self.num_signs
        )

    def predict(self, landmarks, top_k=5, return_all_probs=False):
        """Run inference on a sequence of landmark frames.

        Args:
            landmarks: numpy array of shape (num_frames, 543, 3), float32.
                       Landmark ordering: face(468) + left_hand(21) + pose(33) + right_hand(21).
                       Values normalized to [0,1] by MediaPipe.
            top_k: number of top predictions to return.
            return_all_probs: if True, include 'all_probs' key with full softmax (250 floats).

        Returns:
            dict with 'sign', 'confidence', 'top_5', and optionally 'all_probs' and 'class_indices'.
        """
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks, dtype=np.float32)

        if landmarks.dtype != np.float32:
            landmarks = landmarks.astype(np.float32)

        if landmarks.ndim == 2:
            landmarks = landmarks.reshape(-1, LANDMARKS_PER_FRAME, COORDS_PER_LANDMARK)

        num_frames = landmarks.shape[0]
        if landmarks.shape != (num_frames, LANDMARKS_PER_FRAME, COORDS_PER_LANDMARK):
            raise ValueError(
                f"Expected shape (N, {LANDMARKS_PER_FRAME}, {COORDS_PER_LANDMARK}), "
                f"got {landmarks.shape}"
            )

        # Keep NaN values -- the model's internal preprocessing uses NaN
        # to distinguish "not detected" from "at position (0,0,0)"

        start = time.time()
        result = self.runner(inputs=landmarks)
        logits = result["outputs"]
        elapsed = time.time() - start

        # Convert logits to probabilities via softmax
        exp_logits = np.exp(logits - np.max(logits))
        outputs = exp_logits / exp_logits.sum()

        top_indices = np.argsort(outputs)[-top_k:][::-1]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                "sign": self.idx_to_sign.get(int(idx), f"unknown_{idx}"),
                "class_index": int(idx),
                "confidence": float(outputs[idx]),
            })

        best = top_predictions[0]
        logger.info(
            "Sign prediction: '%s' (%.3f) in %.0fms (%d frames)",
            best["sign"], best["confidence"], elapsed * 1000, num_frames
        )

        response = {
            "sign": best["sign"],
            "confidence": best["confidence"],
            "top_5": top_predictions,
        }
        if return_all_probs:
            response["all_probs"] = outputs.astype(float).tolist()
        return response

    def get_label_map(self):
        """Return the index -> sign label mapping (for client to resolve names)."""
        return dict(self.idx_to_sign)
