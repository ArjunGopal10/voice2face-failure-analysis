"""Stage 2: reference faces (recognisers A and B), speech embeddings, perturbed speech embeddings.
Input : pairs.csv from scripts/01_make_subset.py --extract-dir  (speaker_id, split, face_path, wav_path)
Output: <work>/samples.csv, <work>/features.npz, <work>/excluded_speakers.csv
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import argparse, os
import cv2, numpy as np, pandas as pd, torch
from vf.audio import load_wav, PERTURBATIONS
from vf.faces import build_apps, embed_largest
from vf.speech import SpeechEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True); ap.add_argument("--meta", default=None, help="vox2_meta.csv (speaker sex)")
ap.add_argument("--work", default="work"); ap.add_argument("--arc-root", default="Arc2Face")
ap.add_argument("--seconds", type=float, default=3.0)
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(a.pairs)
need = {"speaker_id", "split", "face_path", "wav_path"}
assert need <= set(df.columns), f"pairs.csv needs columns {need}"
df["group_sex"] = "unknown"
if a.meta:
    m = pd.read_csv(a.meta, sep=None, engine="python"); m.columns = [c.strip() for c in m.columns]
    spk, sex = m.iloc[:, 0].astype(str).str.strip(), m.iloc[:, 2].astype(str).str.strip()
    df["group_sex"] = df["speaker_id"].map(dict(zip(spk, sex))).fillna("unknown")

# reference face per speaker = highest detection score among that speaker's frames
appA, appB = build_apps(os.path.abspath(a.arc_root))
ref = {}
for r in df.itertuples():
    im = cv2.imread(r.face_path)
    if im is None: continue
    eA, s = embed_largest(appA, im)
    if eA is not None and s > ref.get(r.speaker_id, (None, -1, None))[1]:
        eB, _ = embed_largest(appB, im)
        if eB is not None: ref[r.speaker_id] = (eA, s, eB)
excluded = sorted(set(df.speaker_id) - set(ref))
pd.Series(excluded, name="speaker_id").to_csv(f"{a.work}/excluded_speakers.csv", index=False)
print(f"{len(excluded)} speakers excluded (no usable reference face)")
df = df[df.speaker_id.isin(ref)].reset_index(drop=True)
eA = np.stack([ref[s][0] for s in df.speaker_id]); eB = np.stack([ref[s][2] for s in df.speaker_id])

enc = SpeechEncoder(dev)
names = list(PERTURBATIONS)
V, VP = [], []
for p in df.wav_path:
    w = load_wav(p, a.seconds)
    V.append(enc(w)); VP.append(np.stack([enc(PERTURBATIONS[n](w)) for n in names]))
df.to_csv(f"{a.work}/samples.csv", index=False)
np.savez(f"{a.work}/features.npz", e_ref_A=eA, e_ref_B=eB, v=np.stack(V), v_pert=np.stack(VP), pert_names=np.array(names))
print("samples:", len(df), "| speakers:", df.speaker_id.nunique(), "\n", df.groupby("split").speaker_id.nunique())
