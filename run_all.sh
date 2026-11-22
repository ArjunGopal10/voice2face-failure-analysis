#!/usr/bin/env bash
# End-to-end pilot. Edit the three paths, then: bash run_all.sh
set -euo pipefail
VOX2_MP4=/path/to/vox2/dev/mp4        # official VoxCeleb2 video folder (idXXXXX/videoID/clip.mp4)
META=/path/to/vox2_meta.csv           # https://www.openslr.org/49/
WORK=work

python scripts/01_make_subset.py --root "$VOX2_MP4" --max-gb 3 --extract-dir "$WORK/pairs" --out "$WORK/pairs.csv"
# 01 writes face_path/wav_path columns when --extract-dir is used
python scripts/02_extract_features.py --pairs "$WORK/pairs.csv" --meta "$META" --work "$WORK"
python scripts/03_train_projector.py --work "$WORK"
python scripts/04_generate_and_embed.py --work "$WORK" --download --seeds 5
python scripts/05_analyze.py --work "$WORK"
# python scripts/06_demo_app.py --work "$WORK"
