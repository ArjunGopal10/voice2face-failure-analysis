"""Stage 5: build the analysis arrays and run the failure analysis (recogniser A = main tables,
recogniser B = independent check of identity and retrieval only).
Output: <work>/results_A/, <work>/results_B/ (results.json, per_sample.csv, results_tables.txt)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import argparse
import numpy as np, pandas as pd
from vf.analysis import analyse, load

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="work"); ap.add_argument("--q", type=float, default=0.9)
ap.add_argument("--far", type=float, default=0.01); ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
w = a.work
df = pd.read_csv(f"{w}/samples.csv"); f = np.load(f"{w}/features.npz"); pr = np.load(f"{w}/projected.npz")
keep = np.where(df.split.isin(["val", "test"]) & np.load(f"{w}/done.npy"))[0]
def pack(gen, eref):
    return dict(speaker_id=df.speaker_id.values[keep].astype(str), split=df.split.values[keep].astype(str),
                e_ref=eref[keep], z_hat=pr["z"][keep], gen_emb=gen[keep], v=f["v"][keep],
                v_pert=f["v_pert"][keep], z_pert=pr["z_pert"][keep], group_sex=df.group_sex.values[keep].astype(str))
np.savez(f"{w}/pilot_A.npz", **pack(np.load(f"{w}/gen_A.npy"), f["e_ref_A"]))
np.savez(f"{w}/pilot_B.npz", **pack(np.load(f"{w}/gen_B.npy"), f["e_ref_B"]))
print("==== Recogniser A (identity space; all tables) ====")
analyse(load(f"{w}/pilot_A.npz"), f"{w}/results_A", a.q, a.far, a.boot)
print("\n==== Recogniser B (independent evaluator; read ONLY identity/retrieval rows) ====")
print("(alignment-error rows are not meaningful for B: z_hat lives in space A)")
analyse(load(f"{w}/pilot_B.npz"), f"{w}/results_B", a.q, a.far, a.boot)
