"""Projection network g_phi: speaker embedding -> ArcFace identity vector (unit norm).
Trained on TRAIN speakers only; early stopping on the VALIDATION alignment loss."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, d_in=192, d_out=512, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def info_nce_same_speaker(z, e, same, temperature=0.07):
    """InfoNCE where every sample of the same speaker is a positive (no false negatives)."""
    logp = F.log_softmax(z @ e.T / temperature, dim=1)
    return -(logp * same).sum(1).div(same.sum(1)).mean()


def train_projector(V, E_ref, speaker_ids, split, device="cuda", epochs=300, patience=30,
                    lr=1e-3, weight_decay=1e-4, temperature=0.07, cos_weight=1.0,
                    out="projector.pt", seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = np.where(split == "train")[0]; va = np.where(split == "val")[0]
    if len(tr) < 20 or len(va) < 5:
        raise ValueError("too few train/val samples for a meaningful projector")
    g = Projector(V.shape[1], E_ref.shape[1]).to(device)
    opt = torch.optim.AdamW(g.parameters(), lr, weight_decay=weight_decay)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Vt, Et, Vv, Ev = t(V[tr]), t(E_ref[tr]), t(V[va]), t(E_ref[va])
    sp = np.unique(speaker_ids[tr], return_inverse=True)[1]
    same = torch.tensor(sp[:, None] == sp[None, :], dtype=torch.float32, device=device)
    best, bad, hist = 1e9, 0, []
    for ep in range(epochs):
        g.train(); z = g(Vt)
        loss = info_nce_same_speaker(z, Et, same, temperature) + cos_weight * (1 - (z * Et).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        g.eval()
        with torch.no_grad():
            lv = (1 - (g(Vv) * Ev).sum(-1)).mean().item()
        hist.append((ep, float(loss), lv))
        if lv < best - 1e-4:
            best, bad = lv, 0; torch.save(g.state_dict(), out)
        else:
            bad += 1
            if bad > patience: break
    g.load_state_dict(torch.load(out)); g.eval()
    return g, dict(epochs_run=len(hist), best_val_alignment_loss=best, history=hist)
