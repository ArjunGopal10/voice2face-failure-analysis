#!/usr/bin/env python3
"""
Build a <= N GB face-voice subset manifest (CSV) from a local VoxCeleb2 copy.

Expected layout (official VoxCeleb2):
    <root>/id00012/21Uxsk56VDQ/00001.mp4
    <audio-root>/id00012/21Uxsk56VDQ/00001.m4a   (optional; otherwise audio is
                                                   taken from the mp4 itself)

Clips are picked round-robin across speakers (1 clip per speaker per round,
each from a different video when possible) so the subset covers as many
identities as the size budget allows. Splits are speaker-disjoint.

Usage:
    python scripts/01_make_subset.py --root /data/vox2/dev/mp4 --max-gb 3
    python scripts/01_make_subset.py --root ... --audio-root /data/vox2/dev/aac \
        --copy-to ./vox2_subset
    python scripts/01_make_subset.py --root ... --extract-dir ./vox2_pairs   # needs ffmpeg
"""
import argparse, csv, random, shutil, subprocess
from pathlib import Path


def list_speaker_clips(spk_dir: Path):
    """Return {video_id: [clip paths]} for one speaker."""
    vids = {}
    for v in sorted(p for p in spk_dir.iterdir() if p.is_dir()):
        clips = sorted(v.glob("*.mp4"))
        if clips:
            vids[v.name] = clips
    return vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="VoxCeleb2 mp4 root (contains idXXXXX folders)")
    ap.add_argument("--audio-root", default=None, help="optional separate audio (m4a) root")
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--max-clips-per-speaker", type=int, default=5)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="voxceleb2_pairs_3gb.csv")
    ap.add_argument("--copy-to", default=None, help="copy selected files into this folder")
    ap.add_argument("--extract-dir", default=None,
                    help="extract one face frame (jpg) + 16 kHz wav per clip (needs ffmpeg)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    root = Path(a.root)
    aroot = Path(a.audio_root) if a.audio_root else None
    budget = int(a.max_gb * 1024 ** 3)

    speakers = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("id"))
    rng.shuffle(speakers)
    if a.max_speakers:
        speakers = speakers[: a.max_speakers]

    # per-speaker shuffled clip queues (one clip per video first)
    queues = {}
    for s in speakers:
        vids = list_speaker_clips(s)
        vid_ids = list(vids); rng.shuffle(vid_ids)
        q = []
        for v in vid_ids:
            q.append((v, rng.choice(vids[v])))
        queues[s.name] = q[: a.max_clips_per_speaker]

    rows, used = [], 0
    rnd = 0
    while used < budget:
        progressed = False
        for s in speakers:
            q = queues[s.name]
            if rnd >= len(q):
                continue
            vid, clip = q[rnd]
            size = clip.stat().st_size
            apath = None
            if aroot:
                cand = aroot / s.name / vid / (clip.stem + ".m4a")
                if cand.exists():
                    apath = cand
                    size += cand.stat().st_size
            if used + size > budget:
                continue
            used += size
            progressed = True
            rows.append(dict(speaker_id=s.name, video_id=vid, clip_id=clip.stem,
                             video_path=str(clip.relative_to(root)),
                             audio_path=str(apath.relative_to(aroot)) if apath else str(clip.relative_to(root)),
                             size_bytes=size))
        if not progressed:
            break
        rnd += 1

    # speaker-disjoint split 80/10/10
    spk = sorted({r["speaker_id"] for r in rows}); rng.shuffle(spk)
    n = len(spk); cut1, cut2 = int(0.8 * n), int(0.9 * n)
    split = {s: "train" if i < cut1 else "val" if i < cut2 else "test" for i, s in enumerate(spk)}
    for r in rows:
        r["split"] = split[r["speaker_id"]]

    if a.copy_to:
        out = Path(a.copy_to)
        for r in rows:
            for base, rel in ((root, r["video_path"]), (aroot or root, r["audio_path"])):
                dst = out / ("video" if base == root else "audio") / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(base / rel, dst)

    if a.extract_dir:
        out = Path(a.extract_dir)
        for r in rows:
            src = root / r["video_path"]
            stem = f'{r["speaker_id"]}_{r["video_id"]}_{r["clip_id"]}'
            (out / "faces").mkdir(parents=True, exist_ok=True)
            (out / "wavs").mkdir(parents=True, exist_ok=True)
            face, wav = out / "faces" / f"{stem}.jpg", out / "wavs" / f"{stem}.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", str(src),
                            "-frames:v", "1", str(face)], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
                            "-ac", "1", "-ar", "16000", str(wav)], check=True)
            r["face_path"], r["wav_path"] = str(face), str(wav)

    fields = list(rows[0].keys()) if rows else []
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} pairs, {len(spk)} speakers, {used/1024**3:.2f} GB -> {a.out}")


if __name__ == "__main__":
    main()
