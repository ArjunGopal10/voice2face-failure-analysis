"""Audio loading and the controlled perturbation grid (paper Table XI subset)."""
import torch, torchaudio

SR = 16000


def load_wav(path, seconds=3.0):
    """Mono, 16 kHz, truncated to a fixed length (paper: fixed 3 s clip)."""
    w, sr = torchaudio.load(path)
    w = w.mean(0)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w[: int(SR * seconds)]


def add_noise(w, snr_db, seed=0):
    """White noise at a target SNR. Seeded, so every run perturbs identically."""
    g = torch.Generator().manual_seed(seed)
    n = torch.randn(w.shape, generator=g)
    p = w.pow(2).mean().clamp_min(1e-10)
    pn = n.pow(2).mean()
    return w + n * torch.sqrt(p / (pn * 10 ** (snr_db / 10)))


def telephone(w):
    """Telephone band (300-3400 Hz)."""
    w = torchaudio.functional.highpass_biquad(w, SR, 300)
    return torchaudio.functional.lowpass_biquad(w, SR, 3400)


PERTURBATIONS = {
    "snr20": lambda w: add_noise(w, 20),
    "snr10": lambda w: add_noise(w, 10),
    "snr5": lambda w: add_noise(w, 5),
    "snr0": lambda w: add_noise(w, 0),
    "dur1s": lambda w: w[:SR],
    "dur2s": lambda w: w[: 2 * SR],
    "telephone": telephone,
}
