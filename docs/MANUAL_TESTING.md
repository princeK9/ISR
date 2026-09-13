# Manual Testing — Live Webcam Demo

How to test `src/webcam_demo.py` once you have a webcam and, ideally, someone who can
produce the signs (or reference clips to copy).

**Read section 6 before judging the results.** Live accuracy is expected to be meaningfully
worse than the 0.5285 test accuracy, for reasons that are a property of the data rather than
a bug in the demo.

---

## 1. Setup

### 1.1 Dependencies

Both `opencv-python` and `mediapipe` are already in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Confirm they import and that the Tasks API is present:

```bash
python -c "import cv2, mediapipe as mp; print(cv2.__version__, mp.__version__); print(hasattr(mp.tasks.vision, 'HolisticLandmarker'))"
```

You want `True` on the last line.

> **Version note.** Most MediaPipe tutorials online use `mediapipe.solutions.holistic`. That
> legacy API was **removed in MediaPipe 1.x** — `mp.solutions` does not exist at all. This
> demo uses the current Tasks API (`mediapipe.tasks.vision.HolisticLandmarker`). Verified
> against mediapipe 1.0.1 and opencv-python 5.0.0.

### 1.2 MediaPipe model bundle

The Tasks API needs a `.task` model bundle (13.7 MB). The demo downloads it automatically to
`models/holistic_landmarker.task` on first run. To pre-fetch, or to fetch it manually on a
machine without internet at demo time:

```bash
python -c "from src.webcam_demo import ensure_model_bundle, HOLISTIC_MODEL_URL, DEFAULT_MODEL_PATH; ensure_model_bundle(DEFAULT_MODEL_PATH, HOLISTIC_MODEL_URL, True)"
```

Pass `--no-download` to fail loudly instead of downloading, and `--model-path` to point at a
bundle you obtained yourself. If the URL ever breaks, the bundle is published on the
MediaPipe "Holistic landmarker" task page in Google's official documentation.

### 1.3 Trained checkpoint

The demo defaults to `models/gru_best.pt` — the GRU, which is the model selected on
validation accuracy. `models/` is gitignored, so a fresh clone will not have it. Either
train it (`python -m src.train`, ~33 min on CPU) or copy the checkpoint across.

### 1.4 Label vocabulary

The demo resolves the 30 sign names in this order:

1. `data/processed/top30_signs.json` (written by `notebooks/eda.ipynb`)
2. `data/raw/train.csv`, re-deriving with `select_top_signs`
3. A built-in fallback list in `src/webcam_demo.py`

So it works from a fresh clone without the 53 GB dataset. The vocabulary size is checked
against the checkpoint's classifier width at load time — a mismatch raises rather than
silently mislabelling every prediction.

### 1.5 Camera permissions on Windows

1. **Settings → Privacy & security → Camera**
2. Turn on **Camera access**
3. Turn on **Let apps access your camera**
4. Turn on **Let desktop apps access your camera** — this is the one that matters for
   Python, and it is a separate toggle further down the page that is easy to miss.

Close Teams, Zoom, OBS, and anything else holding the camera; on Windows only one process
can open it at a time.

---

## 2. Running it

```bash
# Default webcam
python -m src.webcam_demo

# A different camera
python -m src.webcam_demo --source 1

# A pre-recorded clip instead of a camera
python -m src.webcam_demo --source path/to/clip.mp4

# Process a clip with no window and save the annotated result
python -m src.webcam_demo --source clip.mp4 --headless --save out.mp4
```

Press **`q`** to quit (or `Ctrl-C` in the terminal).

### Useful flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--source` | `0` | Camera index, or a video file path |
| `--checkpoint` | `models/gru_best.pt` | Trained model to load |
| `--model` | `gru` | `gru` or `transformer` |
| `--window` | `64` | Sliding window length; matches training `max_len` |
| `--predict-every` | `5` | Classify every N frames, to keep CPU load sane |
| `--smooth` | `3` | Average probabilities over the last N predictions; `1` disables |
| `--min-confidence` | `0.0` | Hide predictions below this confidence |
| `--allow-no-hands` | off | Predict even with no hand detected (see §6.3) |
| `--save` | — | Write the annotated video |
| `--headless` | off | No window; for file processing |
| `--no-mirror` | off | Disable the display mirror |
| `--max-frames` | `0` | Stop after N frames |

### What you should see

- A window with the camera feed, mirrored by default so it behaves like a mirror.
- Coloured dots on the body and hands: blue pose, green left hand, red right hand.
- A top bar reading `buffering 12/64` until the window fills, then a predicted sign and
  confidence — green ≥60%, amber ≥35%, red below.
- A bottom status line: `L:Y R:- buf 64/64 22 fps q=quit`.

---

## 3. The 30 supported signs

The model is a **closed-set 30-way classifier**. It has no "unknown" class, so any sign
outside this list will still be labelled as one of these 30 — confidently, sometimes.
Testing only makes sense with signs from this list.

```
awake     bird      brown     bye       cat
cow       doll      donkey    drink     duck
fireman   first     hear      icecream  lips
listen    look      make      mouse     napkin
nuts      pen       pretend   shhh      sleepy
think     toothbrush uncle    wake      who
```

This is the exact list in `data/processed/top30_signs.json` — the 30 most frequent signs in
the dataset, alphabetically sorted, which is the ordering the label indices follow.

### Suggested easier starting signs

Based on measured per-class test F1, start with the ones the model handles best:

- **`brown`** (F1 0.78), **`uncle`** (0.77), **`cow`** (0.75), **`lips`** (0.70),
  **`bird`** (0.70)

And expect trouble with the weakest, which are worth testing last:

- **`pretend`** (F1 0.19), **`awake`** (0.21), **`who`** (0.31), **`pen`** (0.32),
  **`listen`** (0.32)

---

## 4. Finding reference clips

I have not linked to specific videos, because I cannot verify individual page URLs are live
and correct for each of the 30 words, and a wrong reference is worse than none.

Suggested sources — public ASL dictionaries where you can look up each word by name:

- **Lifeprint / ASL University** (Dr. Bill Vicars) — widely used, has an alphabetical
  dictionary index.
- **HandSpeak** — searchable ASL dictionary with video for each entry.
- **Signing Savvy** — searchable, with multiple angles for many signs.
- **The ASL App**, or any reputable ASL learning app with a dictionary section.

Search each of the 30 words above by name. **Check that the source is ASL specifically** —
BSL, Auslan, and ISL are different languages with different signs, and a BSL clip will not
match what this model was trained on.

> **Caveat on reference clips.** Signs have regional and individual variation. The dataset
> was collected through the PopSign ASL learning game, so the model learned whatever
> variants those particular 21 participants produced. A perfectly correct sign from a
> different regional variant may still be misclassified.

---

## 5. What success looks like

### Working correctly

- Landmark dots track the hands and shoulders smoothly, without flicker or lag.
- `L:` / `R:` indicators light up when each hand is visible.
- The buffer fills to 64/64 within a few seconds and stays there.
- Performing a supported sign produces **that sign's label**, at least sometimes,
  with confidence above roughly 40%.
- The label changes when you change signs, rather than being stuck on one class.

### A reasonable pass bar

Given the caveats in §6, I would consider the demo working if:

- **The pipeline is provably correct**: dots track, buffer fills, predictions update, no
  crashes. This is the part that is fully verified already (§7).
- **Some fraction of the easier signs land correctly** — even 3 or 4 of the top-5 signs
  producing their own label some of the time demonstrates the pipeline end to end.
- Predictions are **not** constant. A label that never changes regardless of what you do
  means something is wrong upstream; see §8.

**Do not expect 52.9%.** See below.

---

## 6. Known limitations — expected, not bugs

### 6.1 Live accuracy will be lower than 0.5285, probably substantially

The 0.5285 test accuracy was measured on held-out **dataset** recordings: the same capture
setup, framing, and landmark extraction as training, differing only in the signer. A live
webcam changes several things at once — camera, distance, angle, lighting, background,
resolution, frame rate, and the MediaPipe version doing the extraction. Every one of those
is a distribution shift the model never saw.

This is a genuine limitation of the project, not a defect in the demo, and it should be
stated plainly rather than explained away. The model was never trained or validated on
webcam input.

### 6.2 The sliding window does not match training segmentation

Training sequences are **pre-segmented clips** containing one sign, median 24 frames, padded
out to 64 with the padding masked off. The live demo feeds a **continuous 64-frame window**
where every frame is real and unmasked.

So the live input resembles only the ~15% of training sequences that ran to 64 frames or
more. A window will also usually contain lead-in and lead-out motion around the sign rather
than the sign alone. There is no automatic segmentation here — that is a separate unsolved
problem the project does not address.

**Practical consequence:** hold each sign for about 2 seconds, pause between signs, and
watch for the label during and just after the sign.

### 6.3 The classifier always returns something

It is a 30-way softmax with no "none of these" option. With nobody in frame it will still
name a sign, and can do so at high confidence — during testing, an empty room produced
`icecream 84%`.

The demo therefore **withholds predictions when no hand was detected anywhere in the
window** and shows `no hands detected` instead. `--allow-no-hands` disables that guard, and
is mainly useful for observing the behaviour just described.

### 6.4 Confidence is not reliable

The model is overconfident on errors — measured on the test set, `pretend` was misclassified
as `fireman` at **99% confidence**. Treat confidence as a rough signal only. A high number
does not mean the prediction is right, and no calibration (temperature scaling, label
smoothing) has been applied.

### 6.5 Some confusions are systematic

The most frequent test-set errors were `awake`↔`wake` (44 in both directions),
`pretend`→`fireman` (19), and `doll`→`mouse` (14). These are signs sharing hand trajectory
and differing mainly in handshape. If you see those specific swaps live, that is the model
reproducing a known, documented weakness — worth recording, not surprising.

### 6.6 Normalization is translation-only

The pipeline centres coordinates on the shoulder midpoint but does **not** normalize scale.
Distance from the camera and body size are therefore uncontrolled variables. Framing
yourself at a similar distance to a typical dataset recording — upper body filling most of
the frame — is likely to work better, though this has not been measured.

### 6.7 Speed

On CPU, expect the MediaPipe extraction to dominate, not the classifier — the GRU itself
runs in roughly 3 ms. If the feed is choppy, raise `--predict-every` before anything else.

---

## 7. What is already verified

Testing done without a camera, so you are not starting from zero:

- **`src/test_webcam_pipeline.py` passes.** It feeds identical synthetic landmarks through
  both the training path (`ASLLandmarkDataset.process_sequence`, via a generated parquet
  file) and the live path (`window_to_features`, via mock MediaPipe results) and asserts the
  model inputs are **bit-identical** — for sequences both shorter and longer than `max_len`.
  This is the property that matters most: the live preprocessing is not an approximation of
  training, it is the same code.
- **Edge cases pass**: nothing detected at all, no pose/shoulders, a partially-filled window,
  and a single frame all produce finite features without raising.
- **The demo runs end to end** on a synthetic video file through real MediaPipe and the real
  checkpoint: buffering countdown, prediction, overlay rendering, and video writing all
  confirmed working, exit code 0.

**Not verified, because it needs hardware and a signer:** actual camera capture, real-world
landmark quality, and whether any sign is recognised correctly live.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `could not open video source 0` | Camera in use, or permissions | Close other camera apps; enable **desktop apps** camera access (§1.5); try `--source 1` |
| `Missing dependency` on startup | opencv/mediapipe not installed | `pip install -r requirements.txt` |
| `No checkpoint at models/gru_best.pt` | Model not trained or not copied | `python -m src.train`, or copy the checkpoint |
| `MediaPipe model bundle not found` | `--no-download` with no bundle | Drop `--no-download`, or pass `--model-path` |
| Vocabulary/class-count error | Checkpoint trained with a different vocabulary | Point `--signs-json` at the matching vocabulary |
| Stuck on `buffering N/64` | Window never fills — frames are not arriving | Check the feed is live; a video file shorter than 64 frames can never fill the window |
| `no hands detected` persists | Hands not visible to MediaPipe | Improve lighting, avoid backlighting; keep hands fully in frame; plain background; move closer |
| No dots on hands, dots on body | Hands out of frame or too small | Step back so upper body and both hands are visible |
| No dots at all | Camera producing black frames, or lens covered | Check the feed in another app first |
| Label never changes | Window full of near-identical frames, or hands never detected | Confirm `L:`/`R:` light up; move between signs; lower `--smooth` to `1` |
| Very choppy video | MediaPipe extraction is CPU-bound | Raise `--predict-every` to 10; add `--no-draw`; lower camera resolution |
| Label flickers rapidly | Predictions unstable frame to frame | Raise `--smooth` to 5; raise `--min-confidence` to ~0.4 |
| Crash mid-session | Camera disconnected or driver issue | The error is printed with its type; reconnect and rerun |

### Recording results

If you do get a signer, `--save out.mp4` writes the annotated feed so results can be
reviewed afterwards rather than judged live. Testing against saved clips with
`--source clip.mp4` is also more reproducible than live capture, since the same input can
be re-run after any change.
