# Mapping the code to the paper

| Paper item | Where in the code |
|---|---|
| Identity similarity, acceptance rate, Recall@K | `analysis.sample_metrics`, `verification_threshold`, `recall_at_k` |
| Acoustic sensitivity (perturbation grid) | `audio.PERTURBATIONS`, `analysis.sample_metrics` (`A_mean`, `A_max`) |
| Alignment error | `analysis.sample_metrics` (`E`) |
| Identity drift over seeds | `analysis.sample_metrics` (`drift`) |
| Threshold calibration on verified validation samples | `analysis.thresholds` |
| Failure taxonomy and categorisation | `analysis.categorise` |
| Failure profile with speaker-level CIs | `analysis.profile`, `spk_bootstrap_ci` |
| Subgroup disparity (bootstrap, permutation, Kruskal-Wallis, Holm) | `analysis.subgroup_report`, `holm` |
| Label-free trust check | `analysis.trust_check` |
| Projection network and contrastive training | `projector.py` |
| Frozen generator | `generator.py` |
| Two recognisers (identity space and independent evaluator) | `faces.py` |

## Input contract of the analysis (`.npz`)
`speaker_id`, `split`, `e_ref`, `z_hat`, `gen_emb[N,M,D]`, `v`, `v_pert[N,P,Dv]`; optional `z_pert`, `quality`, and any
`group_<name>` column (for example `group_sex`). Any pipeline that produces these arrays can use the analysis.

## Honest-reporting rules
- Thresholds come from the validation split only. Never tune them on test data.
- Report speaker counts next to every figure. Report the gallery size next to every Recall@K.
- Faces with no detectable face are counted as having no identity (zero vector). Report how many there were.
- If an analysis was not run, say so. Do not replace it with an estimate.
