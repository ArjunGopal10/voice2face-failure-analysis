"""Speaker encoder: pretrained ECAPA-TDNN (SpeechBrain). Output size 192."""
import torch


class SpeechEncoder:
    def __init__(self, device="cuda", savedir="pretrained/ecapa"):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # older SpeechBrain
            from speechbrain.pretrained import EncoderClassifier
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb", savedir=savedir, run_opts={"device": device})

    @torch.no_grad()
    def __call__(self, wav):
        return self.model.encode_batch(wav[None].to(self.device)).squeeze().cpu().numpy()
