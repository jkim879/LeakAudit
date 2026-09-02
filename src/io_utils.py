"""
Dataset discovery, path diagnostics and loaders for the three benchmarks.

D1  AI-Hub 138        Korean water-pipe leak detection (tabular CSV; 513 spectral
                      bins at 10 Hz spacing plus device fields and identifiers).
D2  Hong Kong         Field recordings on live buried mains (WAV; Mendeley,
                      CC BY 4.0). Metadata (logger, date, time, pipe material)
                      is parsed from file names and folder names.
D3  Aghashahi testbed Controlled laboratory rig (CSV + RAW; Mendeley, CC BY 4.0).
                      Fully factorial over topology, leak type, flow and noise.

Every loader is defensive: if a directory has been left zipped or nested one
level deep after extraction, :func:`resolve_root` descends automatically, and
:func:`diagnose` prints exactly what was found so problems are visible early.
"""
from __future__ import annotations
import os
import re
import glob
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------- #
# Generic path helpers
# --------------------------------------------------------------------------- #
def resolve_root(root, want_ext) -> Path:
    """Descend through single nested folders until files of interest appear.

    Extracting a Mendeley archive often yields ``D3_testbed/tbrnp6vrnj-1/...``;
    this walks down such single-child directories (up to five levels) so the
    caller need not know the exact nesting.
    """
    root = Path(root)
    if not root.exists():
        return root
    cur = root
    for _ in range(5):
        if any(p.suffix.lower() in want_ext for p in cur.rglob("*") if p.is_file()):
            return cur
        subdirs = [d for d in cur.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            cur = subdirs[0]
        else:
            break
    return cur


def diagnose(root, name, want_ext=None) -> None:
    """Print a concise inventory of a dataset directory for sanity checking."""
    root = Path(root)
    print("=" * 72)
    print(f"[{name}] path   = {root.resolve()}")
    print(f"[{name}] exists = {root.exists()}")
    if not root.exists():
        print("  !! path not found -- check ROOT or the folder name.")
        return
    files = [p for p in root.rglob("*") if p.is_file()]
    print(f"[{name}] files  = {len(files)}")
    if not files:
        print("  !! directory is empty -- is the archive still zipped?")
        return
    print(pd.Series([p.suffix.lower() or "(none)" for p in files])
          .value_counts().to_string())
    if want_ext:
        n = sum(p.suffix.lower() in want_ext for p in files)
        print(f"[{name}] target-extension files {sorted(want_ext)} = {n}")
    print(f"[{name}] top-level entries: {[d.name for d in root.iterdir()]}")
    for p in files[:5]:
        print("   ", p.relative_to(root))


# --------------------------------------------------------------------------- #
# D1 -- AI-Hub 138 (tabular)
# --------------------------------------------------------------------------- #
_D1_CLASS_KEYS = [("out-", "out"), ("in-", "in"), ("noise", "noise"),
                  ("other", "other"), ("normal", "normal")]
_D1_CLASS_KO = [("옥외", "out"), ("옥내", "in"), ("기계", "noise"),
                ("환경", "other"), ("정상", "normal")]


def _d1_class_of(path) -> str:
    base = os.path.basename(path)
    for key, val in _D1_CLASS_KEYS:
        if key in base:
            return val
    for key, val in _D1_CLASS_KO:
        if key in base:
            return val
    raise ValueError(f"class not resolved: {base}")


def _d1_split_of(path) -> str:
    return "train" if "training" in os.path.basename(path).lower() else "val"


def load_d1(force: bool = False):
    """Load and cache the AI-Hub tabular benchmark.

    Returns
    -------
    X : (N, 513) float32
        Raw frequency-bin spectra, 0-5120 Hz at 10 Hz spacing.
    X_max : (N, 20) float32
        Ten repeated night-time detections, each (max frequency, max level).
    meta : DataFrame
        Identifiers (site, sid), date, device fields (lrate, llevel),
        label and official train/val split.
    hz : (513,) int
        Centre frequency of each spectral bin.
    """
    fx, fm = cfg.CACHE / "d1_spec.npy", cfg.CACHE / "d1_meta.csv"
    if fx.exists() and not force:
        X = np.load(fx)
        X_max = np.load(cfg.CACHE / "d1_max.npy")
        meta = pd.read_csv(fm)
        # The CSV cache loads 'dt' as a string; restore it to datetime.
        meta["dt"] = pd.to_datetime(meta["dt"], errors="coerce")
        hz = np.load(cfg.CACHE / "d1_hz.npy")
        return X, X_max, meta, hz

    files = sorted(glob.glob(str(cfg.D1 / "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV under {cfg.D1}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df = df.assign(cls=_d1_class_of(path), split=_d1_split_of(path))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    freq_cols = [c for c in df.columns if c.upper().endswith("HZ")]
    max_cols = [c for c in df.columns if c.upper().startswith("MAX")]
    hz = np.array([int(re.sub(r"[^0-9]", "", c)) for c in freq_cols])
    order = np.argsort(hz)
    hz = hz[order]
    freq_cols = [freq_cols[i] for i in order]

    X = df[freq_cols].to_numpy(np.float32)
    X_max = df[max_cols].to_numpy(np.float32)
    meta = df[["site", "sid", "ldate", "lrate", "llevel",
               "leaktype", "cls", "split"]].copy()
    meta["dt"] = pd.to_datetime(meta.ldate.astype(str),
                                format="%Y%m%d", errors="coerce")

    np.save(fx, X)
    np.save(cfg.CACHE / "d1_max.npy", X_max)
    np.save(cfg.CACHE / "d1_hz.npy", hz)
    meta.to_csv(fm, index=False)
    return X, X_max, meta, hz


# --------------------------------------------------------------------------- #
# D2 -- Hong Kong field recordings
# --------------------------------------------------------------------------- #
def parse_d2_name(stem: str) -> dict:
    """Extract (logger, date, hhmm) from a Hong Kong file name.

    Three naming conventions occur in the public sample:
      * ``NNN_LOGGER-YYYYMMDD``        e.g. 003_05593-20180927
      * ``LOGGER_YYYYMMDD_HHMM``       e.g. 05574_20170703_1145
      * ``G.S.NN.HHMM``                e.g. 1.1.01.0330  (group.site.index.time)
    """
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)\.(\d{4})", stem)
    if m:
        return dict(logger=f"G{m.group(1)}S{m.group(2)}", date=None, hhmm=m.group(4))

    toks = stem.replace("-", "_").split("_")
    date = next((t for t in toks if re.fullmatch(r"(19|20)\d{6}", t)), None)
    hhmm = None
    if date is not None:
        after = toks[toks.index(date) + 1:]
        hhmm = next((t for t in after if re.fullmatch(r"\d{4}", t)), None)

    cand = [t for t in toks if re.fullmatch(r"\d{3,6}", t) and t not in (date, hhmm)]
    logger = None
    if cand:
        logger = max(cand, key=len)
        # If the first token is a 3-digit running index, prefer the other id.
        if len(cand) > 1 and re.fullmatch(r"\d{3}", toks[0]) and toks[0] in cand:
            rest = [c for c in cand if c != toks[0]]
            if rest:
                logger = max(rest, key=len)
    return dict(logger=logger, date=date, hhmm=hhmm)


def _d2_label_material(folder: str):
    """Map a folder name to (label, pipe material)."""
    f = folder.lower().replace("-", "").replace("_", "").replace(" ", "")
    table = {
        "leak": ("leak", None), "noleak": ("noleak", None),
        "leakmetal": ("leak", "metal"), "leaknonmetal": ("leak", "nonmetal"),
        "noleakmetal": ("noleak", "metal"), "noleaknonmetal": ("noleak", "nonmetal"),
    }
    if f in table:
        return table[f]
    if f.startswith("noleak"):
        return "noleak", ("nonmetal" if "nonmetal" in f else "metal" if "metal" in f else None)
    if f.startswith("leak"):
        return "leak", ("nonmetal" if "nonmetal" in f else "metal" if "metal" in f else None)
    return None, None


def scan_d2(root=None) -> pd.DataFrame:
    """Build a metadata table for the Hong Kong recordings (no audio loaded)."""
    root = resolve_root(root or cfg.D2, {".wav", ".xlsx"})
    rows = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in (".wav", ".xlsx") or p.stem.startswith("~$"):
            continue
        rel = p.relative_to(root).parts[:-1]
        sensor = rel[0] if rel else "?"
        folder = rel[-1] if rel else "?"
        label, material = _d2_label_material(folder)
        rows.append(dict(path=str(p), sensor=sensor, folder=folder,
                         label=label, material=material, ext=ext,
                         **parse_d2_name(p.stem)))
    cols = ["path", "sensor", "folder", "label", "material",
            "ext", "logger", "date", "hhmm"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# D3 -- Aghashahi testbed
# --------------------------------------------------------------------------- #
_D3_LEAK_CODE = {"LC": "longitudinal", "CC": "circumferential",
                 "GL": "gasket", "OL": "orifice", "NL": "noleak"}
_D3_FOLDER_LEAK = [("NL", "noleak"), ("LC", "longitudinalcrack"),
                   ("CC", "circumferentialcrack"), ("GL", "gasketleak"),
                   ("OL", "orificeleak")]


def parse_d3_name(stem: str) -> dict:
    """Parse a testbed file name such as ``LO_OL_ND_N_H2``.

    Fields: topology (LO/BR), leak type (LC/CC/GL/OL/NL), flow
    (ND / Transient / 0.18 / 0.47 litre s-1), background noise (N/NN)
    and sensor channel (A1/A2, H1/H2, P1/P2).
    """
    d = dict(topology=None, leaktype=None, flow=None, noise=None, sensor=None)
    for x in re.split(r"[_\s]+", stem):
        xu = x.upper()
        if xu in ("LO", "BR"):
            d["topology"] = xu
        elif xu in _D3_LEAK_CODE:
            d["leaktype"] = xu
        elif xu in ("ND", "TRANSIENT"):
            d["flow"] = xu
        elif re.fullmatch(r"0\.\d+", x):
            d["flow"] = x
        elif xu == "LPS":
            pass  # 'litres per second' unit token, already captured by the value
        elif xu in ("N", "NN"):
            d["noise"] = xu
        elif re.fullmatch(r"[AHP]\d", xu):
            d["sensor"] = xu
    return d


def _d3_modality(rel, sensor, ext) -> str:
    top = rel[0].lower() if rel else ""
    if "hydro" in top:
        return "hydrophone"
    if "acceler" in top:
        return "accel"
    if "pressure" in top:
        return "pressure"
    if ext == ".raw":
        return "hydrophone"
    return {"A": "accel", "P": "pressure", "H": "hydrophone"}.get(
        (sensor or "")[:1].upper(), "unknown")


def scan_d3(root=None) -> pd.DataFrame:
    """Build a metadata table for the testbed signals (no waveform loaded)."""
    root = resolve_root(root or cfg.D3, {".csv", ".raw"})
    rows = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in (".csv", ".raw"):
            continue
        if "convert" in p.stem.lower() or p.stem.startswith("~$"):
            continue
        rel = p.relative_to(root).parts[:-1]
        d = parse_d3_name(p.stem)
        d["path"], d["ext"] = str(p), ext
        d["folder"] = rel[-1] if rel else ""
        d["modality"] = _d3_modality(rel, d["sensor"], ext)
        d["is_bg"] = ("background" in p.stem.lower()) or \
                     any("background" in r.lower() for r in rel)
        # Fall back to the folder name when the leak type is absent from the stem.
        if d["leaktype"] is None and rel:
            f = rel[-1].lower().replace("-", "").replace(" ", "")
            for code, nm in _D3_FOLDER_LEAK:
                if nm in f:
                    d["leaktype"] = code
                    break
        d["label"] = None if d["leaktype"] is None else \
            ("noleak" if d["leaktype"] == "NL" else "leak")
        rows.append(d)
    cols = ["path", "ext", "folder", "modality", "topology", "leaktype",
            "flow", "noise", "sensor", "is_bg", "label"]
    return pd.DataFrame(rows, columns=cols)


def load_d3_raw(path, sr=None, n_samples=None):
    """Read one Aghashahi hydrophone ``.raw`` clip exactly as the provider does.

    The deposit ships *Python code to convert RAW acoustic data to pandas
    DataFrame.py*, which specifies

        soundfile.read(path, channels=1, samplerate=8000,
                       subtype='PCM_32', endian='LITTLE')
        df.head(240000)

    so the stream is **32-bit little-endian PCM, mono, 8 kHz**, and only the
    first 240 000 samples (30 s) are the signal of record. The files on disk are
    longer than that (about 40 s); the provider truncates.

    Reading these bytes as int16 splits every 32-bit word in two and yields a
    sequence with a zero in every other position, at twice the true length and
    the wrong amplitude scale. We therefore read int32 explicitly.
    """
    sr = cfg.SR_HYDRO_HZ if sr is None else sr
    n_samples = cfg.CLIP_SAMPLES if n_samples is None else n_samples
    try:
        import soundfile as sf
        x, _ = sf.read(str(path), channels=1, samplerate=sr, format="RAW",
                       subtype="PCM_32", endian="LITTLE", dtype="float32")
        x = np.asarray(x, dtype=np.float32).ravel()
    except Exception:
        # Equivalent numpy fallback when libsndfile cannot open a RAW stream.
        x = np.fromfile(path, dtype="<i4").astype(np.float32) / 2147483648.0
    if n_samples:
        x = x[:n_samples]
    return x, sr


def load_d3_signal(row, raw_dtype=None, sr_hydro=None, sr_csv=None):
    """Load a single testbed waveform, returning (samples, sampling_rate).

    ``raw_dtype`` is accepted for backwards compatibility and ignored: the raw
    format is fixed by the provider's specification (see :func:`load_d3_raw`).
    """
    sr_hydro = cfg.SR_HYDRO_HZ if sr_hydro is None else sr_hydro
    sr_csv = cfg.SR_CSV_HZ if sr_csv is None else sr_csv

    if row["ext"] == ".raw":
        return load_d3_raw(row["path"], sr=sr_hydro)

    df = pd.read_csv(row["path"])
    num = df.select_dtypes("number")
    x = num.iloc[:, -1].to_numpy(np.float32)
    return x, sr_csv
