"""CPU-only smoke tests for the analysis code, using SYNTHETIC data with injected failures.
They check that the code runs and is internally consistent. They say nothing about any real model."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from vf import analysis as A


def _run(tmp_path):
    npz = tmp_path / "demo.npz"
    A.make_demo(str(npz), seed=1)
    return A.analyse(A.load(str(npz)), str(tmp_path / "res"), R=200)


def test_outputs_exist_and_profile_sums_to_one(tmp_path):
    res = _run(tmp_path)
    for f in ("results.json", "per_sample.csv", "results_tables.txt"):
        assert (tmp_path / "res" / f).exists()
    shares = [v["share"] for v in res["failure_profile"].values()]
    assert abs(sum(shares) - 1.0) < 1e-6          # every test sample is verified or in exactly one failure class


def test_thresholds_finite_and_recall_monotone(tmp_path):
    res = _run(tmp_path)
    assert all(np.isfinite(list(res["thresholds"].values())))
    r = res["retrieval"]
    assert r["Recall@1"][0] <= r["Recall@5"][0] <= r["Recall@10"][0] + 1e-9


def test_injected_failures_are_detected(tmp_path):
    res = _run(tmp_path)
    p = res["failure_profile"]
    assert p["verified"]["share"] < 1.0           # synthetic data contains unreliable speakers
    assert p["acoustic"]["share"] + p["alignment"]["share"] + p["generative_drift"]["share"] > 0


def test_cosine_and_unit_helpers():
    x = np.array([[3.0, 4.0]])
    assert abs(np.linalg.norm(A.unit(x)) - 1.0) < 1e-9
    assert abs(A.cos(x, x).item() - 1.0) < 1e-9
