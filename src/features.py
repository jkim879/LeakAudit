from __future__ import annotations
import numpy as np
from scipy import signal

from . import config as cfg


def bandpass(x, sr, lo=cfg.BPF_LOW_HZ, hi=cfg.BPF_HIGH_HZ, order=5):

    nyq = sr / 2.0
    hi_ = min(hi, nyq * 0.99)
    if lo >= hi_:
        return x
    b, a = signal.butter(order, [lo / nyq, hi_ / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def pick_window(x, sr, sec=1.0, mode="minstd", seed=0):

    n = int(sec * sr)
    if mode == "full" or len(x) <= n:
        return x
    if mode == "first":
        return x[:n]
    if mode == "random":
        start = np.random.RandomState(seed).randint(0, len(x) - n)
        return x[start:start + n]
    # 'minstd': lowest-variance one-second window.
    step = max(1, n // 4)
    best_std, best_i = np.inf, 0
    for i in range(0, len(x) - n + 1, step):
        s = x[i:i + n].std()
        if s < best_std:
            best_std, best_i = s, i
    return x[best_i:best_i + n]


def _mel_db(x, sr, n_mels=cfg.N_MELS):
    """Mel-spectrogram in decibels (import librosa lazily)."""
    import librosa
    S = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=min(cfg.N_FFT, len(x)),
        hop_length=cfg.HOP, n_mels=n_mels, power=2.0)
    return librosa.power_to_db(S, ref=np.max)


def summary_features(x, sr, n_mels=cfg.N_MELS):

    D = _mel_db(x, sr, n_mels)
    mel_stats = np.concatenate([D.mean(axis=1), D.std(axis=1)])

    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    w = mag.sum() + 1e-12
    centroid = (mag * freqs).sum() / w
    spread = np.sqrt((((freqs - centroid) ** 2) * mag).sum() / w)
    p = mag / w
    entropy = -(p * np.log(p + 1e-12)).sum()
    rolloff = freqs[np.searchsorted(np.cumsum(mag), 0.85 * mag.sum())] \
        if mag.sum() > 0 else 0.0
    zcr = np.mean(np.abs(np.diff(np.sign(x)))) / 2.0
    extra = np.array([centroid, spread, entropy, rolloff, zcr,
                      x.std(), np.abs(x).mean()])
    return np.concatenate([mel_stats, extra]).astype(np.float32)


FEATURE_DIM = 2 * cfg.N_MELS + 7

DESCRIPTOR_NAMES = ["spec_centroid_hz", "spec_spread_hz", "spec_entropy",
                    "rolloff85_hz", "zero_crossing_rate", "rms", "mean_abs"]


def feature_names(n_mels=cfg.N_MELS):
    """Names of the 2 * n_mels + 7 summary features, in order."""
    return ([f"mel{i:02d}_mean_db" for i in range(n_mels)]
            + [f"mel{i:02d}_std_db" for i in range(n_mels)]
            + DESCRIPTOR_NAMES)


def effective_band(sr, lo=cfg.BPF_LOW_HZ, hi=cfg.BPF_HIGH_HZ):
    """(low, high) corner actually applied by :func:`bandpass` at rate ``sr``."""
    return float(lo), float(min(hi, sr / 2.0 * 0.99))
