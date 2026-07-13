# Recognition retraining pipeline

Trains an isolated-sign classifier on landmark sequences and exports a TFLite
model plus a label map in the schema `asl_sign_recognizer` reads. The current
live model is the Kaggle GISLR 250-class set, whose vocabulary is toddler-domain
ASL; this pipeline is how that gets replaced with everyday adult vocabulary.

## Stages

1. `extract_landmarks.py` - one video to a `(frames, 543, 3)` landmark array in
   the canonical face+left_hand+pose+right_hand order. Runnable standalone.
2. `prepare_dataset.py` - a directory of labelled video clips to per-clip `.npy`
   landmark files plus `labels.json`.
3. `train_classifier.py` - a temporal CNN over fixed-length landmark sequences,
   exported to `model.tflite` + `sign_to_prediction_index_map.json`.

Input layout for stage 2 (one subdirectory per sign):

```
<root>/HELLO/clip1.mp4
<root>/COFFEE/clip1.mp4
...
```

## Commands

```
python tools/train/prepare_dataset.py <video_root> <prepared_dir>
python tools/train/train_classifier.py <prepared_dir> <out_dir> --epochs 60
```

The pipeline is verified end to end on the bundled `test_videos` as a smoke
test (8 labels, one clip each): it produces a TFLite model with input
`(1, 32, 1629)` and a normalized softmax output that loads and runs on the same
tflite runtime as the live recognizer. That run is a mechanical check only, not
a usable model; real accuracy needs a real corpus and many clips per sign.

## Datasets for everyday adult vocabulary

Arrange any of these into the stage-2 layout. Licensing governs whether a model
trained on them can ship; verify before production use.

| Corpus | Vocab | Notes | License reality |
| --- | --- | --- | --- |
| ASL Citizen (Microsoft) | 2,731 | Consented studio capture, balanced, ISLR-ready | Microsoft Research license, non-commercial research only |
| Sem-Lex | 3,149 | Largest ASL ISLR set, aligns to ASL-LEX, pose files | Code Apache-2.0; data via terms-of-use form, confirm commercial rights |
| WLASL | 2,000 | Web-scraped video, popular benchmark | C-UDA non-commercial; source videos third-party copyrighted |
| MS-ASL | 1,000 | YouTube-sourced, link rot common | C-UDA non-commercial |

Practical reading: none of the large ASL corpora cleanly authorize shipping a
commercial model except possibly Sem-Lex pending its terms-of-use wording. For a
research or non-commercial build, ASL Citizen is the cleanest. A commercial
product needs either Sem-Lex clearance or licensed/own-captured data.

## Compute

Landmark extraction is CPU MediaPipe, roughly real-time per clip; tens of
thousands of clips is hours of preprocessing. Training the temporal model on a
few thousand signs needs a GPU. Neither the corpora nor the training run is
bundled here; both are offline steps.

## Integrating a trained model

The exported model takes `(1, 32, 1629)` (32 frames x 543 landmarks x 3). The
live `asl_sign_recognizer.predict` feeds the GISLR signature instead, so wiring
a model trained here in requires one of:

- adapt `asl_sign_recognizer` to resample incoming landmarks to 32 frames and
  call this model's signature, or
- match this pipeline's preprocessing to whatever architecture you train.

Keep the GISLR model in place until a replacement clears both an accuracy bar on
a held-out set and the licensing check above. Do not overwrite `data/models`
with a smoke-test artifact.
