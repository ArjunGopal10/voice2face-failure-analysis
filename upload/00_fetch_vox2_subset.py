#!/usr/bin/env python3
"""Fetch a small, speaker-balanced VoxCeleb2 subset WITHOUT downloading the full archive.

A .zip keeps its index (central directory) at the end of the file, so with HTTP range
requests we can list every member and pull only the ones we want. The VoxCeleb2 *test*
archive is ~8.9 GB in total, but a pilot needs well under 1 GB of it.

    pip install remotezip tqdm
    export HF_TOKEN=hf_xxx          # only if the mirror asks you to log in
    python scripts/00_fetch_vox2_subset.py --out data/vox2_subset --speakers 118 \
        --videos-per-speaker 12 --max-gb 2.5

Output: <out>/mp4/<speaker>/<video>/<clip>.mp4 and <out>/manifest.csv
Then point scripts/01_make_subset.py at <out>/mp4 (it extracts the face frame + 16 kHz wav).

Why the test split: its speakers are disjoint from the VoxCeleb2 dev set, which is what
speechbrain/spkrec-ecapa-voxceleb was trained on, so the speech encoder has never seen them.
Only 118 speakers exist, so use many clips per speaker and keep splits speaker-disjoint.

Licence: VoxCeleb is for research use under the VGG / Oxford terms. The Hugging Face repo is a
third-party mirror; register on the official page and respect its terms.
"""
import argparse, csv, os, random, re, sys
from collections import defaultdict
from pathlib import Path

DEFAULT_URL = ("https://huggingface.co/datasets/ProgramComputer/voxceleb/"
               "resolve/main/vox2/vox2_test_mp4.zip")
SPK = re.compile(r"^id\d{5}$")


def parse(name):
    parts = Path(name).parts
    for i, p in enumerate(parts):
        if SPK.match(p) and i + 2 < len(parts):
            return p, parts[i + 1], parts[-1]        # speaker, video id, clip file
    return None


def choose(infos, n_spk, n_vid, rng):
    tree = defaultdict(lambda: defaultdict(list))
    for zi in infos:
        if zi.filename.lower().endswith(".mp4") and (k := parse(zi.filename)):
            tree[k[0]][k[1]].append(zi)
    speakers = sorted(s for s in tree if len(tree[s]) >= 2)
    rng.shuffle(speakers)
    picked = []
    for s in speakers[:n_spk]:
        vids = sorted(tree[s]); rng.shuffle(vids)
        for v in vids[:n_vid]:                         # one clip per distinct video
            picked.append(rng.choice(tree[s][v]))
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default="data/vox2_subset")
    ap.add_argument("--speakers", type=int, default=118)
    ap.add_argument("--videos-per-speaker", type=int, default=12)
    ap.add_argument("--max-gb", type=float, default=2.5, help="hard cap on downloaded bytes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="list the plan, download nothing")
    a = ap.parse_args()

    try:
        from remotezip import RemoteZip
        from tqdm import tqdm
    except ImportError:
        sys.exit("pip install remotezip tqdm")

    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    with RemoteZip(a.url, headers=headers) as zf:
        picked = choose(zf.infolist(), a.speakers, a.videos_per_speaker, rng)
        cap, total, keep = int(a.max_gb * 1024 ** 3), 0, []
        for zi in picked:
            if total + zi.file_size > cap:
                break
            keep.append(zi); total += zi.file_size
        print(f"plan: {len(keep)} clips, {len({parse(z.filename)[0] for z in keep})} speakers, "
              f"{total / 1e6:.0f} MB (cap {a.max_gb} GB)")
        if a.dry_run:
            return
        rows = []
        for zi in tqdm(keep, unit="clip"):
            dest = out / zi.filename
            if not (dest.exists() and dest.stat().st_size == zi.file_size):   # resumable
                zf.extract(zi.filename, path=out)
            spk, vid, clip = parse(zi.filename)
            rows.append([spk, vid, clip, str(dest), zi.file_size])
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["speaker", "video", "clip", "path", "bytes"]); w.writerows(rows)
    print(f"done -> {out}/manifest.csv")


if __name__ == "__main__":
    main()
