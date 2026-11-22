# Voice-to-Face Failure Analysis

A pipeline and analysis toolkit for studying **why diffusion-based voice-to-face generation fails**, not for building a
better generator. A face can look realistic and still be the wrong identity. This project generates faces from speech with
a frozen identity-conditioned diffusion model (Arc2Face), then diagnoses each failure as **acoustic**, **alignment**,
**generative drift**, or **distributional**, with speaker-level confidence intervals.

> **Status.** The analysis code (`src/vf/analysis.py`) is unit-tested on synthetic data (`pytest`). The GPU stages
> (2, 3, 4, 6) were written against the Arc2Face, insightface and SpeechBrain READMEs but **have not yet been run
> end to end**. Expect small fixes on the first run, and delete this note once you have run it. No results are included
> in this repository. Never report numbers that this code has not produced.

## Pipeline

```
VoxCeleb2 clip --> face frame + 16 kHz wav
wav --> ECAPA-TDNN (192-d) --> projector g_phi --> 512-d ArcFace-space identity vector z
z + noise seed --> Arc2Face (frozen SD1.5 + identity UNet) --> face image (M seeds)
generated faces --> recogniser A (identity space) and recogniser B (independent evaluator)
embeddings --> metrics, thresholds, failure categories, subgroup tests, trust check
```

| Stage | Script | Output (in `work/`) |
|---|---|---|
| 1 | `scripts/01_make_subset.py` | `pairs.csv`, face frames, wavs (a subset of VoxCeleb2, default <= 3 GB) |
| 2 | `scripts/02_extract_features.py` | `samples.csv`, `features.npz` (reference faces, speech and perturbed-speech embeddings) |
| 3 | `scripts/03_train_projector.py` | `projector.pt`, `projected.npz`, `train_log.json` |
| 4 | `scripts/04_generate_and_embed.py` | `gen_A.npy`, `gen_B.npy`, `undetected.npy`, `faces/`, `generation_config.json` |
| 5 | `scripts/05_analyze.py` | `results_A/`, `results_B/` (`results.json`, `per_sample.csv`, `results_tables.txt`) |
| 6 | `scripts/06_demo_app.py` | Gradio demo with live failure diagnostics |

## Quick start

### 0. Check the analysis code (CPU, no data needed)
```bash
pip install -e ".[dev]"
pytest -q
python -m vf.analysis --demo          # SYNTHETIC smoke test; its numbers mean nothing
```

### 1. Get the data
1. Request VoxCeleb2 from the official page: <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html>.
   The full dev set is hundreds of GB. Download only part of it, or use the test split.
2. Download `vox2_meta.csv` (speaker sex) from <https://www.openslr.org/49/>.
3. Install `ffmpeg` (`apt-get install ffmpeg`).

### 2. Run (GPU runtime, e.g. Colab T4)
```bash
pip install -r requirements-gpu.txt && pip install -e .
git clone https://github.com/foivospar/Arc2Face       # Arc2Face code (weights are downloaded by stage 4 --download)
# edit the three paths at the top of run_all.sh, then:
bash run_all.sh
```
Or run stage by stage (every stage reads the previous stage's files, and stage 4 is resumable).

Rough size of a pilot: 150 to 300 speakers, 2 clips each, 5 seeds. Only validation and test samples are generated, so
about 800 images for 80 speakers. Expect tens of minutes to a couple of hours on a T4 (an estimate, not a measurement).
Keep at least 5 speakers per subgroup, or the subgroup test reports "not analysable".

### 3. Read the results
`work/results_A/results_tables.txt` has the identity and retrieval table, the failure profile, the failed-versus-verified
diagnostics, the subgroup disparity, the thresholds and the trust check. `work/results_B` repeats identity and retrieval with
an independent recogniser (ignore its alignment rows, since the projected vector lives in recogniser A's space).

## What the analysis computes (`src/vf/analysis.py`)

- Identity similarity (cosine, generated vs reference face), acceptance rate at a fixed false-acceptance rate set on
  **impostor pairs in the validation split**, Recall@1/5/10 with the chance level K/N.
- Acoustic sensitivity (speaker-embedding change under noise, clipping, telephone band), alignment error, identity drift
  (mean pairwise cosine distance between seeds).
- Thresholds for the three diagnostics: a quantile of the **verified validation samples**. The test split never touches them.
- Failure categorisation in pipeline order (acoustic, alignment, generative drift, distributional, otherwise unattributed).
- Failure profile: share of all samples per class, with speaker-level bootstrap intervals.
- Subgroup disparity with a bootstrap interval, a permutation null, Kruskal-Wallis, and Holm-corrected pairwise tests.
- Label-free trust check: refuse an output when acoustic stability or drift is out of range.

Intervals resample **speakers**, not utterances, because utterances of one speaker are not independent.

## Not implemented (report as "not run" until you do them)
Integrated Gradients and SHAP attributions, attention maps, counterfactual pitch and rate edits through the generator,
LPIPS diversity, FID, a face-quality score. The analysis accepts an optional `quality` array if you compute one.

## Design choices you should state in the paper
Number of seeds per sample (default 5; the paper's protocol uses 10), sampler and guidance scale (25 steps, 3.0), clip
length (3 s), the perturbation grid (`src/vf/audio.py`), the quantile used for thresholds (0.9) and the false-acceptance
rate (0.01), and that ECAPA-TDNN was pretrained on VoxCeleb, so check for speaker overlap with your test speakers.

## Repository layout
```
src/vf/        analysis.py audio.py speech.py faces.py projector.py generator.py
scripts/       01 ... 06 (one per stage)
tests/         CPU smoke tests on synthetic data
docs/          PROTOCOL.md  TROUBLESHOOTING.md  ETHICS_AND_LICENSES.md
run_all.sh     end-to-end pilot
```

## Ethics
A face generated from a voice is **not** evidence of who a speaker is. Do not use this system for identification. See
`docs/ETHICS_AND_LICENSES.md`.
