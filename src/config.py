"""
Project-wide configuration: paths, constants and plotting defaults.

All spelling follows British English (IWA house style): 'litre', 'metre',
'colour', 'behaviour', 'normalise', etc. Variable symbols are italicised in
the manuscript; here they are plain identifiers.

Edit ``ROOT`` to point at your local copy of the repository, then place the
three raw datasets under ``ROOT/data_raw`` as documented in the README.
"""
from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Repository root. Override from a notebook with:  cfg.set_root(Path("..."))
ROOT = Path(__file__).resolve().parents[1]

def _p(*parts) -> Path:
    return ROOT.joinpath(*parts)

def set_root(root) -> None:
    """Re-point every derived path at a new repository root (call once)."""
    global ROOT, RAW, D1, D2, D3, CACHE, RESULTS, FIGURES
    ROOT = Path(root).resolve()
    _refresh()

def _refresh() -> None:
    global RAW, D1, D2, D3, CACHE, RESULTS, FIGURES
    RAW     = ROOT / "data_raw"
    D1      = RAW / "D1_aihub"        # Korean AI-Hub dataset 138 (CSV)
    D2      = RAW / "D2_hongkong"     # Hong Kong field recordings (Mendeley, CC BY 4.0)
    D3      = RAW / "D3_testbed"      # Aghashahi testbed (Mendeley, CC BY 4.0)
    CACHE   = ROOT / "data"          # derived feature caches (git-ignored)
    RESULTS = ROOT / "results"       # machine-readable results (CSV / JSON)
    FIGURES = ROOT / "figures"       # publication figures (PNG)
    for d in (CACHE, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)

_refresh()

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 0
N_SEEDS      = 10          # number of random restarts for every experiment

# --------------------------------------------------------------------------- #
# Signal-processing constants (shared by D2 and D3 pipelines)
# --------------------------------------------------------------------------- #
# Butterworth band-pass corner frequencies (Hz). The upper corner is clipped
# to the Nyquist frequency at run time, because D2 recordings are sampled at
# only 4096 Hz (Nyquist = 2048 Hz).
BPF_LOW_HZ  = 100
BPF_HIGH_HZ = 5000

# Mel-spectrogram parameters, reproducing the 2023 deployed pipeline.
N_FFT   = 2048
HOP     = 512
N_MELS  = 96

# Aghashahi testbed formats, taken from the provider's own conversion script
# (Python code to convert RAW acoustic data to pandas DataFrame.py):
#   sf.read(path, channels=1, samplerate=8000, subtype='PCM_32', endian='LITTLE')
#   df.head(240000)
# So hydrophone .raw is 32-bit little-endian PCM, mono, 8 kHz, and only the
# first 240 000 samples (30 s) are the signal of record. The files on disk run
# to roughly 40 s; the provider truncates.
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

# --------------------------------------------------------------------------- #
# Plotting palette (colour-blind-aware, consistent across all figures)
# --------------------------------------------------------------------------- #
PALETTE = dict(
    out="#C1443C", inn="#2F6DB5", noise="#7A7A7A", other="#B08A3E",
    normal="#3E8E5A", ink="#1A1A1A",
    d1="#C1443C", d2="#2F6DB5", d3="#3E8E5A",
)

def apply_mpl_style() -> None:
    """Apply the shared Matplotlib style used for every figure."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.facecolor": "white",
    })
