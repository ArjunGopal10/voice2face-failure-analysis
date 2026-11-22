"""Stage 6: Gradio demo. Voice clip in -> faces + failure diagnostics (needs stages 2-5 done).
Thresholds come from <work>/results_A/results.json (calibrated on the validation split)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import argparse, json, os
import gradio as gr, numpy as np, torch
from vf.audio import load_wav, PERTURBATIONS
from vf.faces import build_apps, embed_largest
from vf.generator import Arc2FaceGenerator
from vf.projector import Projector
from vf.speech import SpeechEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="work"); ap.add_argument("--arc-root", default="Arc2Face")
ap.add_argument("--steps", type=int, default=25); ap.add_argument("--guidance", type=float, default=3.0)
ap.add_argument("--seed0", type=int, default=1000)
a = ap.parse_args()
dev = "cuda"
th = json.load(open(f"{a.work}/results_A/results.json"))["thresholds"]
enc = SpeechEncoder(dev); sd = torch.load(f"{a.work}/projector.pt", map_location=dev)
g = Projector(sd["net.0.weight"].shape[1], sd["net.2.weight"].shape[0]).to(dev); g.load_state_dict(sd); g.eval()
gen = Arc2FaceGenerator(os.path.abspath(a.arc_root)); appA, _ = build_apps(os.path.abspath(a.arc_root))
cosv = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

def run(audio, n):
    if audio is None: return [], "Record or upload a voice clip first."
    w = load_wav(audio); v = enc(w)
    sens = float(np.mean([1 - cosv(v, enc(f(w))) for f in PERTURBATIONS.values()]))
    with torch.no_grad(): z = g(torch.tensor(v[None], dtype=torch.float32, device=dev))[0].cpu().numpy()
    imgs = gen.generate(z, [a.seed0 + j for j in range(int(n))], a.steps, a.guidance)
    es = [e for e in (embed_largest(appA, np.array(im)[:, :, ::-1])[0] for im in imgs) if e is not None]
    drift = float(np.mean([1 - cosv(x, y) for i, x in enumerate(es) for y in es[i + 1:]])) if len(es) > 1 else float("nan")
    flags = []
    if sens > th["tau_a"]: flags.append("acoustically unstable (short, noisy or unusual recording)")
    if drift > th["tau_d"]: flags.append("high identity drift across seeds")
    verdict = "TRUSTED: diagnostics inside the validation-calibrated limits." if not flags else "NOT TRUSTED: " + "; ".join(flags) + "."
    return imgs, (f"Acoustic sensitivity {sens:.3f} (limit {th['tau_a']:.3f})\nIdentity drift {drift:.3f} (limit {th['tau_d']:.3f})\n"
                  f"{verdict}\n\nA generated face is NOT evidence of who the speaker is.")

gr.Interface(run, [gr.Audio(sources=["microphone", "upload"], type="filepath"), gr.Slider(2, 6, value=4, step=1, label="Faces")],
             [gr.Gallery(label="Generated faces"), gr.Textbox(label="Failure diagnostics", lines=6)],
             title="Voice to face: generation with failure diagnostics").launch(share=True)
