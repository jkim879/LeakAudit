"""
Acoustic feature extraction reproducing the 2023 deployed pipeline.

The pipeline has three stages, each retained here so that the audit can vary
them one at a time:

1. Butterworth band-pass filtering (:func:`bandpass`).
2. One-second window selection (:func:`pick_window`), including the original
   *minimum-standard-deviation* heuristic that Experiment 08 scrutinises.
3. Mel-spectrogram summarisation to a fixed-length vector
   (:func:`summary_features`), plus a handful of spectral descriptors.

British English spelling is used throughout ('normalise', 'summarise').
"""
from __future__ import annotations
import numpy as np
from scipy import signal

from . import config as cfg


def bandpass(x, sr, lo=cfg.BPF_LOW_HZ, hi=cfg.BPF_HIGH_HZ, order=5):
    """Zero-phase Butterworth band-pass filter.

    The upper corner is clipped to just below the Nyquist frequency, because
    the Hong Kong recordings are sampled at 4096 Hz (Nyquist = 2048 Hz), so a
    nominal 5000 Hz corner is not attainable there.
    """
    nyq = sr / 2.0
    hi_ = min(hi, nyq * 0.99)
    if lo >= hi_:
        return x
    b, a = signal.butter(order, [lo / nyq, hi_ / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def pick_window(x, sr, sec=1.0, mode="minstd", seed=0):
    """Select a one-second analysis window from a longer clip.

    Parameters
    ----------
    mode : {'minstd', 'random', 'first', 'full'}
        * ``minstd`` -- the 2023 heuristic: slide a one-second window and keep
          the position with the lowest standard deviation. Experiment 08 tests
          whether this helps or is a selection bias.
        * ``random`` -- a random one-second window (control).
        * ``first``  -- the first second (control).
        * ``full``   -- return the whole clip unchanged (control).
    """
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
    """Summarise a waveform as a fixed-length feature vector.

    The vector concatenates the per-band mean and standard deviation of the
    mel-spectrogram with seven global spectral descriptors (centroid, spread,
    entropy, 85 % roll-off, zero-crossing rate, RMS and mean absolute
    amplitude). Length = 2 * n_mels + 7.
    """
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
