from __future__ import annotations
import os
import re
import glob
import hashlib
import datetime as _dt
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------- #
# Generic path helpers
# --------------------------------------------------------------------------- #
def resolve_root(root, want_ext) -> Path:

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


def sorted_files(root, exts=None):

    root = Path(root)
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file()]
    if exts is not None:
        files = [p for p in files if p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix().casefold())


def diagnose(root, name, want_ext=None) -> None:
    """Print a concise inventory of a dataset directory for sanity checking."""
    root = Path(root)
    print("=" * 72)
    print(f"[{name}] path   = {root.resolve()}")
    print(f"[{name}] exists = {root.exists()}")
    if not root.exists():
        print("  !! path not found -- check ROOT or the folder name.")
        return
    files = sorted_files(root)
    print(f"[{name}] files  = {len(files)}")
    if not files:
        print("  !! directory is empty -- is the archive still zipped?")
        return
    print(pd.Series([p.suffix.lower() or "(none)" for p in files])
          .value_counts().to_string())
    if want_ext:
        n = sum(p.suffix.lower() in want_ext for p in files)
        print(f"[{name}] target-extension files {sorted(want_ext)} = {n}")
    print(f"[{name}] top-level entries: {sorted(d.name for d in root.iterdir())}")
    for p in files[:5]:
        print("   ", p.relative_to(root).as_posix())


# --------------------------------------------------------------------------- #
# Versioned caches that carry row identifiers
# --------------------------------------------------------------------------- #
def cache_path(name) -> Path:
    return cfg.CACHE / f"{name}_{cfg.CACHE_VERSION}.npz"


def save_cache(name, X, ids, **extra) -> Path:
    """Save features together with the identifier of every row."""
    fp = cache_path(name)
    np.savez_compressed(fp, X=np.asarray(X), ids=np.asarray(ids, dtype=str),
                        **{k: np.asarray(v) for k, v in extra.items()})
    return fp


def load_cache(name, expected_ids=None, by_unique=False):

    fp = cache_path(name)
    if not fp.exists():
        return None
    z = np.load(fp, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    if expected_ids is not None:
        exp = np.asarray(expected_ids, dtype=str)
        got = out["ids"].astype(str)
        if by_unique:
            got = pd.unique(pd.Series(got)).astype(str)
        if len(exp) != len(got) or not np.array_equal(exp, got):
            print(f"  cache {fp.name}: identifiers do not match the current scan "
                  f"-> rebuilding")
            return None
    return out


# --------------------------------------------------------------------------- #
# Checksums (data_manifest.txt)
# --------------------------------------------------------------------------- #
def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_data_manifest(out=None) -> pd.DataFrame:
    """Compute SHA-256 for every raw file and write ``data_manifest.txt``.

    The manifest identifies the exact copy of each dataset analysed. Commit the
    file to the repository; Table 2 of the manuscript is read from it.
    """
    out = Path(out) if out else cfg.ROOT / "data_manifest.txt"
    rows = []
    for ds, root in [("D1", cfg.D1), ("D2", cfg.D2), ("D3", cfg.D3)]:
        for p in sorted_files(root):
            rows.append(dict(dataset=ds,
                             path=p.relative_to(cfg.RAW).as_posix(),
                             bytes=p.stat().st_size,
                             sha256=sha256_file(p)))
    df = pd.DataFrame(rows, columns=["dataset", "path", "bytes", "sha256"])
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# LeakAudit data manifest: SHA-256 of every raw file analysed\n")
        f.write(f"# generated {stamp} by src.io_utils.write_data_manifest()\n")
        f.write("# columns: sha256  bytes  path (relative to data_raw/)\n")
        for ds in ["D1", "D2", "D3"]:
            sub = df[df.dataset == ds]
            f.write(f"\n# {ds}: {len(sub)} files, {int(sub.bytes.sum()):,} bytes\n")
            for _, r in sub.iterrows():
                f.write(f"{r.sha256}  {r.bytes:>12d}  {r.path}\n")
    return df


# --------------------------------------------------------------------------- #
# D1 -- AI-Hub 138 (tabular)
# --------------------------------------------------------------------------- #
_D1_CLASS_KEYS = [("out-", "out"), ("in-", "in"), ("noise", "noise"),
                  ("other", "other"), ("normal", "normal")]
_D1_CLASS_KO = [("옥외", "out"), ("옥내", "in"), ("기계", "noise"),
                ("환경", "other"), ("정상", "normal")]
D1_META_COLS = ["site", "sid", "ldate", "lrate", "llevel", "leaktype"]


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


def _d1_files():
    return sorted(glob.glob(str(cfg.D1 / "*.csv")))


def _read_csv_any(path, **kw):
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kw)


def _freq_cols(cols):
    return [c for c in cols if str(c).upper().endswith("HZ")]


def _max_cols(cols):
    return [c for c in cols if str(c).upper().startswith("MAX")]


def load_d1(force: bool = False):

    fx, fm = cfg.CACHE / "d1_spec.npy", cfg.CACHE / "d1_meta.csv"
    if fx.exists() and not force:
        X = np.load(fx)
        X_max = np.load(cfg.CACHE / "d1_max.npy")
        meta = pd.read_csv(fm)
        meta["dt"] = pd.to_datetime(meta["dt"], errors="coerce")
        hz = np.load(cfg.CACHE / "d1_hz.npy")
        return X, X_max, meta, hz

    files = _d1_files()
    if not files:
        raise FileNotFoundError(f"no CSV under {cfg.D1}")

    frames = []
    for path in files:
        df = _read_csv_any(path)
        extra = pd.DataFrame({"cls": _d1_class_of(path), "split": _d1_split_of(path)},
                             index=df.index)
        frames.append(pd.concat([df, extra], axis=1))   # no fragmentation warning
    df = pd.concat(frames, ignore_index=True)

    freq_cols = _freq_cols(df.columns)
    max_cols = _max_cols(df.columns)
    hz = np.array([int(re.sub(r"[^0-9]", "", c)) for c in freq_cols])
    order = np.argsort(hz)
    hz = hz[order]
    freq_cols = [freq_cols[i] for i in order]

    X = df[freq_cols].to_numpy(np.float32)
    X_max = df[max_cols].to_numpy(np.float32)
    meta = df[D1_META_COLS + ["cls", "split"]].copy()
    meta["dt"] = pd.to_datetime(meta.ldate.astype(str), format="%Y%m%d",
                                errors="coerce")

    np.save(fx, X)
    np.save(cfg.CACHE / "d1_max.npy", X_max)
    np.save(cfg.CACHE / "d1_hz.npy", hz)
    meta.to_csv(fm, index=False)
    return X, X_max, meta, hz


def d1_column_inventory():

    files = _d1_files()
    rows, cols_by_file = [], {}
    for path in files:
        cols = list(_read_csv_any(path, nrows=0).columns)
        cols_by_file[os.path.basename(path)] = cols
        fc, mc = _freq_cols(cols), _max_cols(cols)
        other = [c for c in cols if c not in fc and c not in mc]
        rows.append(dict(file=os.path.basename(path), n_columns=len(cols),
                         n_freq_bins=len(fc), n_max_cols=len(mc),
                         n_other=len(other), other_columns="|".join(map(str, other))))
    per_file = pd.DataFrame(rows)

    if files:
        cols = cols_by_file[os.path.basename(files[0])]
        same = all(v == cols for v in cols_by_file.values())
        fc = _freq_cols(cols)
        hz = np.sort([int(re.sub(r"[^0-9]", "", c)) for c in fc])
        source = "raw CSV headers"
        other = [c for c in cols if c not in fc and c not in _max_cols(cols)]
        n_max = len(_max_cols(cols))
        n_total = len(cols)
    else:
        hz = np.load(cfg.CACHE / "d1_hz.npy")
        same, other, source = None, D1_META_COLS, "cache (raw CSV files not found)"
        n_max = int(np.load(cfg.CACHE / "d1_max.npy", mmap_mode="r").shape[1])
        n_total = len(hz) + n_max + len(other)
    step = int(np.median(np.diff(hz))) if len(hz) > 1 else 0
    grid = int((hz.max() - hz.min()) / step + 1) if step else len(hz)
    summary = dict(
        source=source,
        identical_header_in_all_files=same,
        n_columns_total=int(n_total),
        n_spectral_bins=int(len(hz)), hz_min=int(hz.min()), hz_max=int(hz.max()),
        hz_step=step, n_positions_inclusive_grid=grid,
        spectral_bins_equal_inclusive_grid=bool(grid == len(hz)),
        n_max_columns=int(n_max),
        non_spectral_columns=[str(c) for c in other],
        n_non_spectral_columns=len(other),
        note=("0-5120 Hz at 10 Hz spacing, endpoints included, is 513 "
              "positions; the provider's '512' is the exclusive count, so the "
              "extra bin is an endpoint-counting convention, not an error."),
    )
    return per_file, summary


# --------------------------------------------------------------------------- #
# D2 -- Hong Kong field recordings
# --------------------------------------------------------------------------- #
D2_NAMING = {
    "hydro_GSNNHHMM": "G.S.NN.HHMM (hydrophone group.site.index.time)",
    "idx_logger_date": "NNN_LOGGER-YYYYMMDD (no time)",
    "logger_date_hhmm": "LOGGER_YYYYMMDD_HHMM",
    "mems_xlsx": "MEMS accelerometer spreadsheet",
    "other": "unrecognised",
}


def parse_d2_name(stem: str) -> dict:
    """Extract (logger, date, hhmm, naming convention) from a Hong Kong file name.

    Three naming conventions occur among the WAV files of the public sample:
      * ``NNN_LOGGER-YYYYMMDD``        e.g. 003_05593-20180927   (no time)
      * ``LOGGER_YYYYMMDD_HHMM``       e.g. 05574_20170703_1145
      * ``G.S.NN.HHMM``                e.g. 1.1.01.0330  (group.site.index.time)
    """
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)\.(\d{4})", stem)
    if m:
        return dict(logger=f"G{m.group(1)}S{m.group(2)}", date=None,
                    hhmm=m.group(4), naming="hydro_GSNNHHMM")

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
    if re.fullmatch(r"\d{3}_\d+-(19|20)\d{6}", stem):
        naming = "idx_logger_date"
    elif date is not None and hhmm is not None:
        naming = "logger_date_hhmm"
    else:
        naming = "other"
    return dict(logger=logger, date=date, hhmm=hhmm, naming=naming)


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


def scan_d2(root=None, audio_info: bool = False) -> pd.DataFrame:

    root = resolve_root(root or cfg.D2, {".wav", ".xlsx"})
    rows = []
    for p in sorted_files(root, {".wav", ".xlsx"}):
        if p.stem.startswith("~$"):
            continue
        rel = p.relative_to(root).parts[:-1]
        sensor = rel[0] if rel else "?"
        folder = rel[-1] if rel else "?"
        label, material = _d2_label_material(folder)
        info = parse_d2_name(p.stem)
        if p.suffix.lower() == ".xlsx":
            info.update(logger=None, date=None, hhmm=None, naming="mems_xlsx")
        rows.append(dict(path=str(p), file_id=p.relative_to(root).as_posix(),
                         sensor=sensor, folder=folder, label=label,
                         material=material, ext=p.suffix.lower(), **info))
    cols = ["path", "file_id", "sensor", "folder", "label", "material",
            "ext", "logger", "date", "hhmm", "naming"]
    df = pd.DataFrame(rows, columns=cols)
    df["timed"] = df.hhmm.notna()
    df["hour"] = np.nan
    t = df.timed
    df.loc[t, "hour"] = (df.loc[t, "hhmm"].str[:2].astype(int)
                         + df.loc[t, "hhmm"].str[2:].astype(int) / 60.0)
    if audio_info:
        import soundfile as sf
        for c in ["sr", "n_frames", "duration_s", "channels", "subtype"]:
            df[c] = None
        for i, r in df[df.ext == ".wav"].iterrows():
            try:
                inf = sf.info(r.path)
                df.loc[i, ["sr", "n_frames", "duration_s", "channels", "subtype"]] = \
                    [inf.samplerate, inf.frames, inf.duration, inf.channels, inf.subtype]
            except Exception as e:     # unreadable header: record, do not fail
                df.loc[i, "subtype"] = f"unreadable: {e}"[:60]
    return df


def d2_population(M2: pd.DataFrame) -> pd.DataFrame:

    wav = M2[(M2.ext == ".wav") & M2.label.isin(["leak", "noleak"])]
    pops = [
        ("all public files", M2, "file", "inventory only"),
        ("MEMS .xlsx", M2[M2.ext == ".xlsx"], "file",
         "accelerometer spreadsheets; not audio; excluded from every analysis"),
        ("WAV files", wav, "file", "pretrained-embedding analysis (Section 4.5)"),
        ("WAV, parseable time", wav[wav.timed], "file",
         "recording-time stump (Section 4.3), Figure 4"),
        ("WAV, no parseable time", wav[~wav.timed], "file",
         "excluded from the time analysis"),
        ("WAV, 03:30-04:30 window", wav[wav.timed & (wav.hour >= 3.5) & (wav.hour <= 4.5)],
         "file", "matched early-morning subset (R2-6)"),
    ]
    rows = []
    for name, d, unit, use in pops:
        vc = d.label.value_counts()
        n = len(d)
        rows.append(dict(population=name, unit=unit, n=n,
                         n_leak=int(vc.get("leak", 0)), n_noleak=int(vc.get("noleak", 0)),
                         majority_rate=float(vc.max() / n) if n else np.nan,
                         n_loggers=int(d.logger.nunique()) if n else 0,
                         sensors="|".join(sorted(d.sensor.unique())), used_for=use))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# D3 -- Aghashahi testbed
# --------------------------------------------------------------------------- #
_D3_LEAK_CODE = {"LC": "longitudinal", "CC": "circumferential",
                 "GL": "gasket", "OL": "orifice", "NL": "noleak"}
_D3_FOLDER_LEAK = [("NL", "noleak"), ("LC", "longitudinalcrack"),
                   ("CC", "circumferentialcrack"), ("GL", "gasketleak"),
                   ("OL", "orificeleak")]


def parse_d3_name(stem: str) -> dict:

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

    root = resolve_root(root or cfg.D3, {".csv", ".raw"})
    rows = []
    for p in sorted_files(root, {".csv", ".raw"}):
        if "convert" in p.stem.lower() or p.stem.startswith("~$"):
            continue
        rel = p.relative_to(root).parts[:-1]
        d = parse_d3_name(p.stem)
        d["path"], d["ext"] = str(p), p.suffix.lower()
        d["file_id"] = p.relative_to(root).as_posix()
        d["folder"] = rel[-1] if rel else ""
        d["modality"] = _d3_modality(rel, d["sensor"], d["ext"])
        d["is_bg"] = ("background" in p.stem.lower()) or \
                     any("background" in r.lower() for r in rel)
        if d["leaktype"] is None and rel:
            f = rel[-1].lower().replace("-", "").replace(" ", "")
            for code, nm in _D3_FOLDER_LEAK:
                if nm in f:
                    d["leaktype"] = code
                    break
        d["label"] = None if d["leaktype"] is None else \
            ("noleak" if d["leaktype"] == "NL" else "leak")
        rows.append(d)
    cols = ["path", "file_id", "ext", "folder", "modality", "topology",
            "leaktype", "flow", "noise", "sensor", "is_bg", "label"]
    df = pd.DataFrame(rows, columns=cols)
    s = lambda c: df[c].fillna("NA").astype(str)
    df["noise"] = df["noise"].fillna("NA")       # accel / pressure carry no noise token
    df["run_id"] = s("modality") + "_" + s("topology") + "_" + s("leaktype") + \
        "_" + s("flow") + "_" + s("noise")
    df["op_condition"] = s("flow") + "_" + s("noise")
    df["cond_topo_leak"] = s("topology") + "_" + s("leaktype")
    return df


def load_d3_raw(path, sr=None, n_samples=None, decoding="pcm32"):

    sr = cfg.SR_HYDRO_HZ if sr is None else sr
    if decoding == "int16_legacy":
        x = np.fromfile(path, dtype="<i2").astype(np.float32) / 32768.0
        return x, sr
    if decoding != "pcm32":
        raise ValueError(f"unknown decoding: {decoding}")
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


def raw_format_evidence(path, sr=None) -> dict:

    sr = cfg.SR_HYDRO_HZ if sr is None else sr
    path = Path(path)
    b = path.read_bytes()
    size = len(b)
    head = b[:4]
    printable = sum(1 for x in b[:4096] if 32 <= x < 127 or x in (9, 10, 13))

    def lag1(x):
        x = x.astype(np.float64)
        x = x - x.mean()
        d = (x * x).sum()
        return float((x[1:] * x[:-1]).sum() / d) if d > 0 else float("nan")

    n_probe = min(size // 4, 200_000) * 4
    i32le = np.frombuffer(b[:n_probe], dtype="<i4")
    i32be = np.frombuffer(b[:n_probe], dtype=">i4")
    i16le = np.frombuffer(b[:n_probe], dtype="<i2")
    low16_zero = float(np.mean((i32le & 0xFFFF) == 0))
    return dict(
        file=path.name, size_bytes=size,
        size_mod_2=size % 2, size_mod_4=size % 4,
        header_hex=head.hex(), has_riff_or_aiff_header=head in (b"RIFF", b"FORM", b"RIFX"),
        printable_fraction_first_4k=printable / min(size, 4096),
        samples_if_int16=size // 2, samples_if_int32=size // 4,
        seconds_if_int32_at_8k=(size // 4) / sr,
        seconds_if_int16_at_8k=(size // 2) / sr,
        spec_samples_30s=int(cfg.CLIP_DUR_SEC * sr),
        excess_samples_int32_vs_spec=size // 4 - int(cfg.CLIP_DUR_SEC * sr),
        frac_int32_words_with_zero_low16=low16_zero,
        frac_int16_even_positions_zero=float(np.mean(i16le[0::2] == 0)),
        lag1_autocorr_int32_le=lag1(i32le), lag1_autocorr_int32_be=lag1(i32be),
        lag1_autocorr_int16_le=lag1(i16le),
    )


def load_d3_signal(row, raw_dtype=None, sr_hydro=None, sr_csv=None,
                   decoding="pcm32"):
    """Load a single testbed waveform, returning (samples, sampling_rate).

    ``raw_dtype`` is accepted for backwards compatibility and ignored.
    """
    sr_hydro = cfg.SR_HYDRO_HZ if sr_hydro is None else sr_hydro
    sr_csv = cfg.SR_CSV_HZ if sr_csv is None else sr_csv

    if row["ext"] == ".raw":
        return load_d3_raw(row["path"], sr=sr_hydro, decoding=decoding)

    df = pd.read_csv(row["path"])
    num = df.select_dtypes("number")
    x = num.iloc[:, -1].to_numpy(np.float32)
    return x, sr_csv
