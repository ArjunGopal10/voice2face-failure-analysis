# Troubleshooting

- **`ModuleNotFoundError: vf`**: run `pip install -e .` from the repository root.
- **`ModuleNotFoundError: arc2face`**: clone <https://github.com/foivospar/Arc2Face> next to the scripts and pass `--arc-root`.
- **Weights not found / antelopev2 download fails**: follow the Arc2Face README for the current model locations and put the
  files under `Arc2Face/models` (`arc2face/`, `encoder/`, `antelopev2/`). Update `generator.download_weights` if a repo id changed.
- **SpeechBrain import error**: `speechbrain.inference` (new) and `speechbrain.pretrained` (old) are both handled; update SpeechBrain if neither works.
- **No face detected in reference frames**: speakers without a usable frame are excluded and listed in `work/excluded_speakers.csv`. Use a later frame (`-ss` in `01_make_subset.py`) or a lower detection threshold.
- **`too few train/val samples`**: use more speakers or clips per speaker; the projector needs a few dozen training samples at least.
- **Out of GPU memory**: lower `--seeds` or keep the default fp16; close other notebooks.
- **Subgroup test says "not analysable"**: fewer than two groups have at least 5 speakers.
- **`too few verified validation samples to calibrate thresholds`**: the projector is too weak on this subset. Train on more speakers; do not lower the bar to force a result.
