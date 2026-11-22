"""Stage 4: generate M faces per VAL/TEST sample with Arc2Face, embed them with recognisers A and B.
Resumable: progress is saved every 10 samples. Train samples are not generated.
Output: <work>/gen_A.npy, gen_B.npy, undetected.npy, done.npy, generation_config.json, faces/*.jpg"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import argparse, json, os
import numpy as np, pandas as pd
from vf.faces import build_apps, embed_largest
from vf.generator import Arc2FaceGenerator, download_weights

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="work"); ap.add_argument("--arc-root", default="Arc2Face")
ap.add_argument("--seeds", type=int, default=5, help="M faces per sample (paper default 10; state your value)")
ap.add_argument("--steps", type=int, default=25); ap.add_argument("--guidance", type=float, default=3.0)
ap.add_argument("--seed0", type=int, default=1000); ap.add_argument("--save-faces", type=int, default=30)
ap.add_argument("--download", action="store_true", help="download Arc2Face weights first")
a = ap.parse_args()
arc = os.path.abspath(a.arc_root)
if a.download: download_weights(arc)
df = pd.read_csv(f"{a.work}/samples.csv"); Z = np.load(f"{a.work}/projected.npz")["z"]
N, M = len(df), a.seeds
gen = Arc2FaceGenerator(arc); appA, appB = build_apps(arc)
p = lambda n: f"{a.work}/{n}.npy"
GA = np.load(p("gen_A")) if os.path.exists(p("gen_A")) else np.zeros((N, M, 512), np.float32)
GB = np.load(p("gen_B")) if os.path.exists(p("gen_B")) else np.zeros_like(GA)
UN = np.load(p("undetected")) if os.path.exists(p("undetected")) else np.zeros((N, M), bool)
DONE = np.load(p("done")) if os.path.exists(p("done")) else np.zeros(N, bool)
os.makedirs(f"{a.work}/faces", exist_ok=True)
json.dump(dict(seeds=M, steps=a.steps, guidance=a.guidance, seed0=a.seed0, base="stable-diffusion-v1-5",
               generator="Arc2Face", scheduler="DPMSolverMultistep"), open(f"{a.work}/generation_config.json", "w"))
todo = [i for i in np.where(df.split.isin(["val", "test"]))[0] if not DONE[i]]
print(len(todo), "samples to generate")
def save():
    for n, x in (("gen_A", GA), ("gen_B", GB), ("undetected", UN), ("done", DONE)): np.save(p(n), x)
for c, i in enumerate(todo):
    imgs = gen.generate(Z[i], [a.seed0 + j for j in range(M)], a.steps, a.guidance)
    for j, im in enumerate(imgs):
        bgr = np.array(im)[:, :, ::-1]
        ea, _ = embed_largest(appA, bgr); eb, _ = embed_largest(appB, bgr)
        if ea is None or eb is None: UN[i, j] = True           # no detectable face = no identity (zero vector)
        if ea is not None: GA[i, j] = ea
        if eb is not None: GB[i, j] = eb
        if c < a.save_faces: im.save(f"{a.work}/faces/{df.speaker_id[i]}_{i}_{j}.jpg")
    DONE[i] = True
    if c % 10 == 0: save(); print(f"{c + 1}/{len(todo)}")
save(); print("faces with no detectable face:", int(UN.sum()), "of", int(DONE.sum() * M))
