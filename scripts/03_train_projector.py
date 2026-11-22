"""Stage 3: train the projection network on TRAIN speakers (early stop on VAL alignment loss).
Output: <work>/projector.pt, <work>/projected.npz (z, z_pert), <work>/train_log.json"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import argparse, json
import numpy as np, pandas as pd, torch
from vf.projector import train_projector

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="work"); ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--patience", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(f"{a.work}/samples.csv"); f = np.load(f"{a.work}/features.npz")
g, log = train_projector(f["v"], f["e_ref_A"], df.speaker_id.values, df.split.values, dev,
                         a.epochs, a.patience, out=f"{a.work}/projector.pt", seed=a.seed)
with torch.no_grad():
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    Z = g(t(f["v"])).cpu().numpy()
    ZP = np.stack([g(t(f["v_pert"][:, k])).cpu().numpy() for k in range(f["v_pert"].shape[1])], 1)
np.savez(f"{a.work}/projected.npz", z=Z, z_pert=ZP)
json.dump(log, open(f"{a.work}/train_log.json", "w"))
print("epochs", log["epochs_run"], "| best val alignment loss", round(log["best_val_alignment_loss"], 4))
