from __future__ import annotations
import os
from pathlib import Path


ROOT = Path(os.environ.get("LEAKAUDIT_ROOT", Path(__file__).resolve().parents[1]))


def set_root(root) -> None:
    """Re-point every derived path at a new repository root (call once)."""
    global ROOT
    ROOT = Path(root).resolve()
    _refresh()


def _refresh() -> None:
    global RAW, D1, D2, D3, CACHE, RESULTS, FIGURES
    RAW     = ROOT / "data_raw"
    D1      = RAW / "D1_aihub"        # Korean AI-Hub dataset 138 (CSV)
    D2      = RAW / "D2_hongkong"     # Hong Kong field recordings (Mendeley, CC BY 4.0)
    D3      = RAW / "D3_testbed"      # Aghashahi testbed (Mendeley, CC BY 4.0)
    CACHE   = ROOT / "data"           # derived feature caches (git-ignored)
    RESULTS = ROOT / "results"        # machine-readable results (CSV / JSON)
    FIGURES = ROOT / "figures"        # publication figures (PNG / PDF)
    for d in (CACHE, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


_refresh()

# --------------------------------------------------------------------------- #
# Run mode and reproducibility
# --------------------------------------------------------------------------- #
QUICK = os.environ.get("LEAKAUDIT_QUICK", "0") == "1"

# quick test mode
RANDOM_STATE = 0
N_SEEDS      = 2 if QUICK else 10    # random restarts for every experiment
N_BOOT       = 200 if QUICK else 2000  # bootstrap replicates
N_PERM       = 5 if QUICK else 50    # permutation replicates (model refits)
N_PERM_CHEAP = 100 if QUICK else 2000  # permutation replicates (no refit)
N_PERM_REFIT = 1 if QUICK else 5     # label-permutation replicates with D1 refits
N_SEEDS_AUX  = 2 if QUICK else 5     # seeds for auxiliary D1 analyses
D1_SUBSAMPLE = 3000 if QUICK else 30000  # max D1 training rows
D1_SUBSAMPLE_MITIGATION = 3000 if QUICK else 20000

CACHE_VERSION = "v2"


BPF_LOW_HZ  = 100
BPF_HIGH_HZ = 5000
BPF_ORDER   = 5            # zero-phase (filtfilt), so effective order 10

# Mel-spectrogram parameters, reproducing the 2023 deployed pipeline.
N_FFT   = 2048
HOP     = 512
N_MELS  = 96
WIN_SEC = 1.0              # analysis window (s)
D2_MAX_SEC = 10.0          # D2 audio read limit (s), as in v1


SR_HYDRO_HZ   = 8000
CLIP_DUR_SEC  = 30.0
CLIP_SAMPLES  = 240000          # 30 s at 8 kHz, as truncated by the provider
SR_CSV_HZ     = 51200           # accelerometer / dynamic-pressure .csv

# --------------------------------------------------------------------------- #
# Class taxonomy (AI-Hub dataset 138)
# --------------------------------------------------------------------------- #
CLASSES = ["out", "in", "noise", "other", "normal"]
CLASS_LABEL_EN = {
    "out":    "outdoor leak (utility side)",
    "in":     "indoor leak (customer side)",
    "noise":  "mechanical / electrical noise",
    "other":  "environmental noise",
    "normal": "no leak",
}

# Task definitions for D1, written once so that every notebook uses the same
# targets (R2-4: the 'alarm/normal' target is now defined explicitly).
D1_TASKS = {
    "T5_five_class":      "five classes: out / in / noise / other / normal",
    "TL_leak_vs_nonleak": "leak (out, in) vs non-leak (noise, other, normal)",
    "TR_out_vs_in":       "outdoor vs indoor, leak rows only",
    "TA_alarm_vs_normal": "alarm (out, in, noise, other) vs normal; the target "
                          "that the device field lrate reproduces",
}


D1_PROVIDER_DOC = {
    "lrate": ("누수확율 (leak rate): over a two-hour early-morning window each day the "
              "sensor checks for a leak 10 times; the share of checks whose response "
              "exceeds a set threshold is reported as a probability. Generated "
              "automatically by the leak-detection sensor. Integer 0-90."),
    "llevel": ("누수레벨 (leak level): mean magnitude of the above-threshold responses in "
               "the same 10 checks. Generated automatically by the sensor. Integer 0-5000."),
    "flagged_classes": ("out, in, noise, other: assigned by a field leak-survey expert to "
                        "sensors on which a leak had been DETECTED (flagged) by the sensor "
                        "(technical report, Table 3)."),
    "normal_class": ("normal: 'leak-detection records of service pipes on which a leak was "
                     "NOT detected' (technical report, Table 3), i.e. records the sensor "
                     "did not flag; not field-surveyed."),
    "provider_state_grouping": {"abnormal (이상 상태)": ["out", "in"],
                                "normal (정상 상태)": ["normal", "noise", "other"]},
    "acquisition_window": "daily, two hours in the early morning (10 checks)",
    "sensors": "Gwangju 7,676 + Goheung 3,441 installed sensors (about 11,000 sites)",
    "documented_size": "technical report 30,000 records; AI-Hub page 50,000; distributed copy 70,384",
    "spectral_columns_documented": ("'512 columns', listed as item 1 = 0 Hz ... item 512 = "
                                    "5,120 Hz at 10 Hz; 0-5,120 Hz with both ends is 513 positions"),
    "max_columns_documented": "20 = 10 detections x (max frequency, max level)",
    "inquiry": "Usol Co. (구종영), nine@usolkorea.com",
}

# --------------------------------------------------------------------------- #
# Full description of the pipeline (written to results/pipeline_spec.json)
# --------------------------------------------------------------------------- #
PIPELINE = {
    "D1_inputs": "513 raw spectral bins (0-5120 Hz inclusive, 10 Hz step); "
                 "no device field (lrate, llevel), no identifier, no date and "
                 "none of the 20 MAX columns is ever an input to an acoustic model",
    "D1_normalisation": "none for tree models; StandardScaler (fit on the "
                        "training fold only) for logistic regression and MLP",
    "D2_D3_audio": {
        "bandpass": f"Butterworth order {BPF_ORDER}, zero-phase, "
                    f"{BPF_LOW_HZ} Hz to min({BPF_HIGH_HZ} Hz, 0.99 x Nyquist)",
        "window": f"{WIN_SEC:.0f} s; 'minstd' = lowest-standard-deviation window "
                  "on a quarter-window hop (2023 rule); controls: random/first/full",
        "features": f"mel spectrogram (n_fft={N_FFT}, hop={HOP}, n_mels={N_MELS}, "
                    "power 2, dB re max); per-band mean and std (192) + 7 "
                    "spectral descriptors = 199 values",
        "D2_read": f"librosa.load(sr=None, mono=True, duration={D2_MAX_SEC:.0f} s)",
        "D3_read": "PCM_32 little-endian mono 8 kHz, first 240 000 samples",
    },
    "models": {
        "gbdt": "HistGradientBoostingClassifier(max_iter=150, early_stopping=False)",
        "rf": "RandomForestClassifier(n_estimators=300)",
        "logreg": "StandardScaler + LogisticRegression(max_iter=2000)",
        "mlp": "StandardScaler + MLPClassifier(hidden=(256,128), max_iter=300)",
        "tuning": "none: library defaults except the values listed; identical "
                  "settings under every protocol, so only the protocol varies",
    },
    "pretrained_checkpoints": {
        "WAV2VEC2_BASE": "torchaudio.pipelines.WAV2VEC2_BASE (last transformer layer)",
        "HUBERT_BASE": "torchaudio.pipelines.HUBERT_BASE (last transformer layer)",
        "AST": "MIT/ast-finetuned-audioset-10-10-0.4593 (last hidden state)",
        "pooling": "mean and std over time -> 1536 values; frozen; "
                   "logistic-regression probe",
    },
}

# --------------------------------------------------------------------------- #
# Figures: seaborn, greyscale, 600 dpi, PNG + PDF (revision 1)
# --------------------------------------------------------------------------- #
FIG_DPI = 600
GREYS = ["#f0f0f0", "#d0d0d0", "#a8a8a8", "#7a7a7a", "#4d4d4d", "#1f1f1f"]
"""Greyscale ramp used by every figure (light -> dark)."""
PALETTE = dict(                     # kept for backwards compatibility; greyscale now
    out=GREYS[5], inn=GREYS[3], noise=GREYS[2], other=GREYS[4],
    normal=GREYS[1], ink=GREYS[5], d1=GREYS[5], d2=GREYS[4], d3=GREYS[3],
)


def grey_palette(n):
    """n distinct greys from dark to light (seaborn 'Greys', excluding white)."""
    import seaborn as sns
    return sns.color_palette("Greys_r", n + 2)[:n]


def apply_mpl_style() -> None:
    """Shared seaborn style for every figure: greyscale, print-ready."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(context="paper", style="ticks", palette=grey_palette(6),
                  font_scale=1.0,
                  rc={"figure.dpi": 120, "savefig.dpi": FIG_DPI, "font.size": 9,
                      "axes.spines.top": False, "axes.spines.right": False,
                      "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                      "legend.frameon": False, "figure.facecolor": "white",
                      "image.cmap": "Greys", "pdf.fonttype": 42, "ps.fonttype": 42})
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=grey_palette(6))


def save_figure(fig, name) -> None:
    """Save a figure as PNG and PDF at 600 dpi under figures/ (one call, both formats)."""
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=FIG_DPI, bbox_inches="tight",
                    facecolor="white")
    print(f"saved figures/{name}.png and .pdf ({FIG_DPI} dpi, greyscale)")


def banner() -> None:
    """Print the run mode at the top of every notebook."""
    mode = "QUICK (smoke test - do NOT report these numbers)" if QUICK else "FULL"
    print(f"LeakAudit | root={ROOT} | mode={mode} | seeds={N_SEEDS} | "
          f"bootstrap={N_BOOT}")
