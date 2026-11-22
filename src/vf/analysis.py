#!/usr/bin/env python3
"""
Failure-analysis kit for diffusion-based voice-to-face generation.
Implements the measurement layers of the paper (Sections IV, VII, VIII):
identity similarity, verification, Recall@K, identity drift, acoustic
sensitivity, alignment error, subgroup disparity (speaker-level bootstrap +
permutation null + Kruskal-Wallis + Holm), validation-quantile thresholds,
failure categorisation (Fig. 2 / Algorithm 6), failure profile (Eq. 31),
and the label-free trust check (Algorithm 9).

It works on PRE-COMPUTED embeddings, so the GPU part (speech encoder, projector,
Arc2Face generation, face recogniser) is separate. See README_pipeline.md.

Input: one .npz with these arrays (N samples, M seeds, P perturbations):
  speaker_id  (N,)  str/int      split (N,) 'train'|'val'|'test'
  e_ref       (N,D) recogniser embedding of the speaker's reference face
  z_hat       (N,D) predicted identity vector (projector output)
  gen_emb     (N,M,D) recogniser embeddings of the M generated faces per sample
  v           (N,Dv) speaker embedding of the clean clip
  v_pert      (N,P,Dv) speaker embeddings of the P perturbed clips
  z_pert      (N,P,D) (optional) projector output for perturbed clips (trust check)
  quality     (N,) optional face-quality score in [0,1] (for the quality-identity gap)
  group_<name> (N,) optional metadata columns, e.g. group_sex, group_age
Usage:
  python -m vf.analysis --demo                 # synthetic smoke test ONLY
  python -m vf.analysis --npz data.npz --out results/
"""
import argparse, json, os
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- basics
def unit(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def cos(a, b):
    return np.sum(unit(a) * unit(b), axis=-1)


def per_speaker_mean(values, spk):
    """utterance -> speaker aggregation (paper Sec. V-B)."""
    u = np.unique(spk)
    return u, np.array([np.nanmean(values[spk == s]) for s in u])


def spk_bootstrap_ci(values, spk, fn=np.mean, R=2000, seed=0, alpha=0.05):
    """Speaker-level (cluster) bootstrap percentile CI (Eq. 27)."""
    rng = np.random.default_rng(seed)
    _, sv = per_speaker_mean(values.astype(float), spk)
    n = len(sv)
    reps = np.array([fn(sv[rng.integers(0, n, n)]) for _ in range(R)])
    return float(fn(sv)), float(np.percentile(reps, 100 * alpha / 2)), \
        float(np.percentile(reps, 100 * (1 - alpha / 2))), n


# ---------------------------------------------------------------- per-sample metrics
def sample_metrics(d):
    N = len(d["e_ref"])
    gen = d["gen_emb"]                                    # (N,M,D)
    S_seed = cos(gen, d["e_ref"][:, None, :])             # (N,M)
    S = S_seed.mean(1)                                    # Eq. 2
    E_align = 1 - cos(d["z_hat"], d["e_ref"])             # Eq. 19
    # identity drift: mean pairwise cosine distance between generated faces (Eq. 18)
    M = gen.shape[1]
    g = unit(gen)
    gram = np.einsum("nmd,nkd->nmk", g, g)
    iu = np.triu_indices(M, 1)
    drift = (1 - gram[:, iu[0], iu[1]]).mean(1) if M > 1 else np.zeros(N)
    # acoustic sensitivity: mean and worst case over perturbations (Eq. 16)
    sens = 1 - cos(d["v"][:, None, :], d["v_pert"])       # (N,P)
    return dict(S=S, E=E_align, drift=drift, A_mean=sens.mean(1), A_max=sens.max(1))


def verification_threshold(d, m, far=0.01):
    """tau_v from IMPOSTOR scores on the validation split at a fixed false-acceptance rate:
    similarity of a generated face to the reference faces of OTHER speakers."""
    idx = np.where(d["split"] == "val")[0]
    gm = unit(d["gen_emb"][idx].mean(1))
    ref = unit(d["e_ref"][idx])
    sim = gm @ ref.T
    spk = d["speaker_id"][idx]
    mask = spk[:, None] != spk[None, :]
    imp = sim[mask]
    return float(np.quantile(imp, 1 - far))


def recall_at_k(d, idx, K=(1, 5, 10)):
    """Rank of the true speaker's reference face in a gallery of the split's speakers."""
    spk = d["speaker_id"][idx]
    users = np.unique(spk)
    gallery = unit(np.stack([d["e_ref"][idx][spk == u][0] for u in users]))
    gm = unit(d["gen_emb"][idx].mean(1))
    sim = gm @ gallery.T
    true = np.searchsorted(users, spk)
    rank = (sim > sim[np.arange(len(idx)), true][:, None]).sum(1) + 1
    out = {}
    for k in K:
        hit = (rank <= k).astype(float)
        out[f"Recall@{k}"] = spk_bootstrap_ci(hit, spk)
        out[f"chance@{k}"] = min(1.0, k / len(users))
    out["gallery_size"] = len(users)
    return out


# ---------------------------------------------------------------- subgroup analysis
def holm(pvals):
    p = np.array(pvals); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


def subgroup_report(S, spk, labels, R=2000, seed=0, min_speakers=5):
    """Group means, disparity D (Eq. 17) with speaker bootstrap CI, permutation null,
    Kruskal-Wallis, Holm-corrected pairwise Mann-Whitney, effect sizes (Eq. 28)."""
    rng = np.random.default_rng(seed)
    users, sv = per_speaker_mean(S, spk)
    # one label per speaker (first occurrence)
    lab = np.array([labels[spk == u][0] for u in users])
    groups = [g for g in np.unique(lab) if (lab == g).sum() >= min_speakers]
    rep = {"groups": {}, "note": f"groups with <{min_speakers} speakers dropped"}
    if len(groups) < 2:
        rep["status"] = "not analysable (fewer than 2 groups with enough speakers)"
        return rep
    keep = np.isin(lab, groups); sv, lab = sv[keep], lab[keep]

    def D_of(vals, labs):
        mus = [vals[labs == g].mean() for g in groups]
        return max(mus) - min(mus)

    D = D_of(sv, lab)
    boots = []
    for _ in range(R):
        ii = rng.integers(0, len(sv), len(sv))
        try:
            boots.append(D_of(sv[ii], lab[ii]))
        except ValueError:
            pass
    null = np.array([D_of(sv, rng.permutation(lab)) for _ in range(R)])
    rep["D"] = float(D)
    rep["D_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    rep["D_perm_p"] = float((np.sum(null >= D) + 1) / (R + 1))
    rep["D_null_95th"] = float(np.percentile(null, 95))
    for g in groups:
        x = sv[lab == g]
        rep["groups"][str(g)] = dict(mean=float(x.mean()), speakers=int(len(x)))
    rep["kruskal_p"] = float(stats.kruskal(*[sv[lab == g] for g in groups]).pvalue)
    pairs, pv, eff = [], [], []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = sv[lab == groups[i]], sv[lab == groups[j]]
            pv.append(stats.mannwhitneyu(a, b).pvalue)
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            eff.append((a.mean() - b.mean()) / sp if sp > 0 else np.nan)
            pairs.append(f"{groups[i]} vs {groups[j]}")
    adj = holm(pv)
    rep["pairwise"] = [dict(pair=p, p_holm=float(a), cohens_d=float(e)) for p, a, e in zip(pairs, adj, eff)]
    rep["lowest_group"] = str(min(rep["groups"], key=lambda g: rep["groups"][g]["mean"]))
    return rep


# ---------------------------------------------------------------- categorisation
def thresholds(m, d, q=0.9):
    """tau_a, tau_e, tau_d = quantile q among VERIFIED validation samples (Eq. 20)."""
    val = d["split"] == "val"
    ver = val & m["V"]
    if ver.sum() < 10:
        raise RuntimeError("too few verified validation samples to calibrate thresholds")
    return dict(tau_a=float(np.quantile(m["A_mean"][ver], q)),
                tau_e=float(np.quantile(m["E"][ver], q)),
                tau_d=float(np.quantile(m["drift"][ver], q)))


def categorise(m, th, dist_flag, in_low_group):
    """Algorithm 6, sequential in pipeline order; unattributed if none fires."""
    out = np.full(len(m["S"]), "verified", dtype=object)
    fail = ~m["V"]
    for i in np.where(fail)[0]:
        if m["A_mean"][i] > th["tau_a"]:
            out[i] = "acoustic"
        elif m["E"][i] > th["tau_e"]:
            out[i] = "alignment"
        elif m["drift"][i] > th["tau_d"]:
            out[i] = "generative_drift"
        elif dist_flag and in_low_group[i]:
            out[i] = "distributional"
        else:
            out[i] = "unattributed"
    return out


TYPES = ["acoustic", "alignment", "generative_drift", "distributional", "unattributed"]


def profile(cat, spk, R=2000):
    """pi_k = share of ALL samples whose failure is of type k (Eq. 31), speaker-bootstrap CIs."""
    res = {}
    for t in TYPES + ["verified"]:
        res[t] = spk_bootstrap_ci((cat == t).astype(float), spk, R=R)
    return res


# ---------------------------------------------------------------- label-free trust (Alg. 9)
def trust_check(d, m, th, q=0.9):
    if "z_pert" not in d:
        return {"status": "z_pert not provided"}
    R = cos(d["z_hat"][:, None, :], d["z_pert"]).mean(1)
    val = d["split"] == "val"; test = d["split"] == "test"
    ver = val & m["V"]
    tau_r = float(np.quantile(R[ver], 1 - q))          # low stability => refuse
    T = (R >= tau_r) & (m["drift"] <= th["tau_d"])
    t = test
    return dict(tau_r=tau_r,
                pass_rate=float(T[t].mean()),
                precision_of_pass=float(m["V"][t & T].mean()) if (t & T).any() else None,
                verified_rate_of_refused=float(m["V"][t & ~T].mean()) if (t & ~T).any() else None)


# ---------------------------------------------------------------- driver
def analyse(d, out_dir, q=0.9, far=0.01, R=2000):
    os.makedirs(out_dir, exist_ok=True)
    m = sample_metrics(d)
    tau_v = verification_threshold(d, m, far)
    m["V"] = m["S"] >= tau_v
    th = thresholds(m, d, q)
    test = np.where(d["split"] == "test")[0]
    spk_t = d["speaker_id"][test]
    res = dict(n_test_samples=int(len(test)), n_test_speakers=int(len(np.unique(spk_t))),
               tau_v=tau_v, far=far, thresholds=th, quantile=q)
    res["S_id"] = spk_bootstrap_ci(m["S"][test], spk_t, R=R)
    res["acceptance_rate"] = spk_bootstrap_ci(m["V"][test].astype(float), spk_t, R=R)
    res["retrieval"] = recall_at_k(d, test)
    if "quality" in d:
        qn = (d["quality"] - d["quality"].min()) / (np.ptp(d["quality"]) + 1e-12)
        sn = (m["S"] - m["S"].min()) / (np.ptp(m["S"]) + 1e-12)
        res["quality_identity_gap"] = spk_bootstrap_ci((qn - sn)[test], spk_t, R=R)
    for k in ("E", "drift", "A_mean", "A_max"):
        res[f"mean_{k}"] = spk_bootstrap_ci(m[k][test], spk_t, R=R)
    # RQ tests: failed vs verified (test split)
    f, v = test[~m["V"][test]], test[m["V"][test]]
    for k, name in (("A_mean", "RQ1_acoustic"), ("E", "RQ2_alignment"), ("drift", "RQ3_drift")):
        if len(f) > 3 and len(v) > 3:
            res[name] = dict(mean_failed=float(m[k][f].mean()), mean_verified=float(m[k][v].mean()),
                             mannwhitney_p=float(stats.mannwhitneyu(m[k][f], m[k][v]).pvalue))
    # subgroups
    res["subgroups"] = {}
    lowflag = np.zeros(len(m["S"]), bool)
    dist_flag = False
    for key in [k for k in d if k.startswith("group_")]:
        rep = subgroup_report(m["S"][test], spk_t, d[key][test], R=R)
        res["subgroups"][key] = rep
        if rep.get("D_perm_p", 1) < 0.05:
            dist_flag = True
            lowflag[test[d[key][test] == rep["lowest_group"]]] = True
    cat = categorise(m, th, dist_flag, lowflag)
    res["failure_profile"] = {k: dict(zip(["share", "ci_lo", "ci_hi", "speakers"], v))
                              for k, v in profile(cat[test], spk_t, R=R).items()}
    res["trust_check"] = trust_check(d, m, th, q)
    res["scope"] = "pilot subset; one pipeline; numbers are NOT population estimates"
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=2, default=float)
    # flat per-sample table
    import csv
    with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["speaker_id", "split", "S_id", "verified", "E_align", "drift", "A_mean", "A_max", "category"])
        for i in range(len(cat)):
            w.writerow([d["speaker_id"][i], d["split"][i], f"{m['S'][i]:.4f}", int(m["V"][i]),
                        f"{m['E'][i]:.4f}", f"{m['drift'][i]:.4f}", f"{m['A_mean'][i]:.4f}",
                        f"{m['A_max'][i]:.4f}", cat[i]])
    # human-readable tables
    ci = lambda t, p=3: f"{t[0]:.{p}f} [{t[1]:.{p}f}, {t[2]:.{p}f}] (n={t[3]} speakers)"
    L = ["TABLE A. IDENTITY AND RETRIEVAL (test split, speaker-level 95% CI)",
         f"  ArcFace cosine S_id      : {ci(res['S_id'])}",
         f"  Acceptance rate (FAR={far}) : {ci(res['acceptance_rate'])}   tau_v={tau_v:.3f}"]
    for k in (1, 5, 10):
        L.append(f"  Recall@{k:<2} (chance {res['retrieval'][f'chance@{k}']:.3f}) : {ci(res['retrieval'][f'Recall@{k}'])}")
    L.append(f"  gallery size             : {res['retrieval']['gallery_size']}")
    L += ["", "TABLE B. FAILURE PROFILE (share of all test samples; Eq. 31)"]
    for t in TYPES + ["verified"]:
        p = res["failure_profile"][t]
        L.append(f"  {t:<17}: {p['share']*100:5.1f}%  [{p['ci_lo']*100:.1f}, {p['ci_hi']*100:.1f}]")
    L += ["", "TABLE C. DIAGNOSTICS, FAILED vs VERIFIED (test split)"]
    for name in ("RQ1_acoustic", "RQ2_alignment", "RQ3_drift"):
        if name in res:
            r = res[name]
            L.append(f"  {name:<14} failed={r['mean_failed']:.4f} verified={r['mean_verified']:.4f} p={r['mannwhitney_p']:.2g}")
    L += ["", "TABLE D. SUBGROUP DISPARITY (RQ4)"]
    for key, r in res["subgroups"].items():
        if "D" not in r:
            L.append(f"  {key}: {r.get('status')}"); continue
        gs = ", ".join(f"{g}={v['mean']:.3f} (n={v['speakers']})" for g, v in r["groups"].items())
        L.append(f"  {key}: D={r['D']:.3f} CI[{r['D_ci95'][0]:.3f},{r['D_ci95'][1]:.3f}] "
                 f"perm-p={r['D_perm_p']:.3f} KW-p={r['kruskal_p']:.3g} | {gs}")
    L += ["", f"Thresholds: tau_a={th['tau_a']:.3f} tau_e={th['tau_e']:.3f} tau_d={th['tau_d']:.3f} (q={q})",
          f"Trust check: {res['trust_check']}"]
    txt = "\n".join(L)
    open(os.path.join(out_dir, "results_tables.txt"), "w").write(txt)
    print(txt)
    return res


# ---------------------------------------------------------------- synthetic smoke test
def make_demo(path, seed=0, n_spk=120, per_spk=4, M=5, P=6, D=64, Dv=48):
    """SYNTHETIC data with known injected failure structure, only to verify the code.
    Not a result of any real model."""
    rng = np.random.default_rng(seed)
    rows = []
    splits = rng.permutation(["train"] * (n_spk * 6 // 10) + ["val"] * (n_spk * 2 // 10) + ["test"] * (n_spk - n_spk * 8 // 10))
    face = unit(rng.normal(size=(n_spk, D)))
    sex = rng.choice(["F", "M"], n_spk)
    voice_noise = np.where(rng.random(n_spk) < 0.2, 1.5, 0.3)        # 20% fragile voices
    bad_align = rng.random(n_spk) < 0.25                              # 25% mis-projected
    gen_noise = np.where(rng.random(n_spk) < 0.15, 1.0, 0.15)         # 15% drifty generations
    out = {k: [] for k in "speaker_id split e_ref z_hat gen_emb v v_pert z_pert group_sex quality".split()}
    for s in range(n_spk):
        for _ in range(per_spk):
            e = face[s] + 0.05 * rng.normal(size=D)
            z = unit(face[s] + (1.2 * rng.normal(size=D) if bad_align[s] else 0.25 * rng.normal(size=D)) + (0.2 if sex[s] == "F" else 0) * 0)
            v = unit(rng.normal(size=Dv) * 0 + face[s][:Dv] + 0.1 * rng.normal(size=Dv))
            vp = unit(v[None] + voice_noise[s] * 0.5 * rng.normal(size=(P, Dv)))
            zp = unit(z[None] + voice_noise[s] * 0.5 * rng.normal(size=(P, D)))
            g = unit(z[None] + gen_noise[s] * rng.normal(size=(M, D)))
            out["speaker_id"].append(f"id{s:05d}"); out["split"].append(splits[s % len(splits)])
            out["e_ref"].append(e); out["z_hat"].append(z); out["gen_emb"].append(g)
            out["v"].append(v); out["v_pert"].append(vp); out["z_pert"].append(zp)
            out["group_sex"].append(sex[s]); out["quality"].append(float(rng.uniform(0.7, 1.0)))
    np.savez(path, **{k: np.array(v) for k, v in out.items()})


def load(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["speaker_id"] = d["speaker_id"].astype(str); d["split"] = d["split"].astype(str)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz"); ap.add_argument("--out", default="results")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--q", type=float, default=0.9); ap.add_argument("--far", type=float, default=0.01)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if a.demo:
        make_demo("demo_synthetic.npz"); a.npz = "demo_synthetic.npz"
        print("### SYNTHETIC SMOKE TEST - not real results ###")
    analyse(load(a.npz), a.out, a.q, a.far, a.boot)
