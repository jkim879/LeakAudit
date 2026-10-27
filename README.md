# LeakAudit

**A validity audit of public acoustic leak-detection benchmarks for water distribution networks.**

Reported accuracies in acoustic leak detection routinely exceed 90 %. This
repository asks a different question: *what are those numbers measuring?*
Across three heterogeneous public benchmarks we hold the learning problem
fixed and vary only the evaluation design, and examine how much of the
reported performance can be explained by evaluation design and data
collection rather than by leak acoustics. The weaknesses differ by benchmark
and arise through different mechanisms. We provide protocols matched to
deployment settings, a signal-free baseline as a diagnostic reporting item,
and a distribution-free abstention layer as a reporting device.

The work is prepared for the *Journal of Hydroinformatics* Special Issue
*AI/ML and Data-Driven Methods for Monitoring, Modelling and Decision Support
in Human-Impacted Water Systems* (IWA Publishing). Code comments follow
British English (IWA house style).

---

## Three benchmarks, three failure modes

| Dataset | Role | Failure mode exposed | Mechanism |
|---|---|---|---|
| **D1** AI-Hub 138 (Korea, 70,384 records, tabular spectra) | main evidence | site / sensor memorisation | sensor identity; device field `lrate` reproduces the sensor's own alarm |
| **D2** Hong Kong field recordings (160 public files: 140 WAV + 20 MEMS XLSX) | field data | recording-time and recording-format confounds | recording time alone reaches 84.7 % on 118 timed files |
| **D3** Aghashahi testbed (120 hydrophone recordings, RAW + CSV) | controlled testbed | recording-level leakage; unseen-topology shift | segment / recording / run / condition / topology split policy |

The three benchmarks differ deliberately in country, sensor, format and
scale. The failure-mode framing follows the data-leakage taxonomy of
Kapoor & Narayanan (2023) and the empirical-audit framing of
McGreivy & Hakim (2024).

---

## Headline results (real data, full run)

- **The signal-free baseline (SFB).** Scored out-of-fold under a
  group-independent protocol on the same rows as the acoustic results. On the
  Korean data the device field `lrate` reaches **0.595** (five-class; majority
  0.315) and **0.807** (leak vs non-leak) across held-out sites; on the Hong
  Kong sample a one-feature stump on *recording time* reaches **0.847**
  across held-out loggers (118 files, 48 leak / 70 no-leak, majority 0.593;
  logger-level 95 % interval 0.719–0.961); on the controlled testbed the
  metadata margin is **zero** across held-out runs (permutation p = 0.42; every
  operating condition holds the same 4 : 1 leak / no-leak ratio). The in-sample
  lookup (1.000 on D1 and D2) is reported separately as a memorisation
  ceiling. We propose the SFB as a diagnostic, not as a threshold an acoustic
  model must pass.
- **Sensor identity and the device field.** Sensor identity attains a
  *perfect* in-sample score (every one of 2,710 sensors is single-class) and
  does not transfer to held-out sensors. The spectrum identifies the sensor
  even within one class (**0.980**, 221 sensors), and with labels tied to
  sensor identity only, a naive split still reaches **0.759** (about half of
  the true-label excess). According to the provider documentation, `lrate` is
  generated automatically by the sensor; it equals 90 for every flagged record
  and is below 90 otherwise, so it reproduces the alarm / normal partition by
  construction and cannot separate leaks from noise.
- **Protocol versus model (D1, D3).** In a factorial variance attribution
  over 4 models × 4 protocols × 10 seeds (160 fits per task), the evaluation
  **protocol accounts for 79–84 %** of the sum of squares in accuracy on the
  three D1 tasks (five-class, leak vs non-leak, outdoor vs indoor; same rows,
  not independent) and **94 %** on D3, whereas the **model family accounts for
  2.5–2.6 %** on D1. On D2 (140 files) split-to-split variation dominates.
  These are descriptive shares for the levels tested.
- **D1 — protocol collapse.** Five-class accuracy falls **0.894 → 0.327**
  from a naive random split to a site-and-temporal split (new sites, later
  period), below the fold majority rate (0.341). The largest step is naive →
  sensor-independent (**35.1 pp**, site-level 95 % interval 23.5–47.7).
  Outdoor-vs-indoor falls **0.910 → 0.506** and is *below* the fold majority
  rate under every non-naive protocol.
- **D2 — listening to the clock.** The recording-time stump reaches **0.847**
  without opening the audio (leak files ≈ 03:50, no-leak files ≈ 10:11).
  Pretrained audio models (best: AST, 0.862) are indistinguishable from the
  clock on the same 118 files and folds. Within the 03:30–04:30 window
  (70 files) the 22 no-leak files come from 5 loggers, and 20 of them were
  recorded at 8,092 Hz while 32 of the 48 leak files were recorded at 4,096 Hz.
  The acoustic contribution therefore cannot be established on the public
  sample.
- **D3 — recording-level leakage and domain shift.** On identical
  out-of-fold segments (common majority rate 0.80, so raw differences equal
  differences in excess over baseline), accuracy is **0.987**
  (segment-random), **0.927** (recording-independent), **0.936**
  (run-independent and unseen flow × noise condition) and **0.802** (unseen
  pipe topology; balanced accuracy 0.51). Recording-level leakage is
  **6.1 pp** (run-level 95 % interval 2.9–9.6); an unseen operating condition
  costs **0.0 pp**; an unseen topology costs **13.3 pp** (6.3–20.8; balanced
  accuracy 34.9 pp), with two topologies only. A run-independent model trained
  on the same reduced size keeps balanced accuracy 0.765, so the topology drop
  is not explained by the smaller training set.
- **Mitigation and abstention (D1 outdoor vs indoor, held-out sites,
  majority 0.560).** Group-DRO and JTT, implemented as published, do not beat
  ERM (**−1.9 pp** and **−2.6 pp**). Split / Mondrian conformal prediction
  attains its target coverage when calibrated on the same sites and falls
  short when calibrated on other sites (0.786 at a 0.80 target); accuracy on
  decided cases stays near the majority rate. Per-cluster risk control lowers
  the selective error only at α = 0.05, where 0.5 % of cases are accepted.

Exact numbers are written to `results/` when the notebooks are run.

> **Note on D3 raw format.** The Aghashahi hydrophone `.raw` files are
> **32-bit little-endian PCM, mono, 8 kHz** (provider conversion script:
> `sf.read(..., samplerate=8000, subtype='PCM_32', endian='LITTLE')`), and the
> provider's converter keeps only the first 240,000 samples (30 s). Reading
> them as int16 splits each 32-bit word in two, halving every frequency and
> adding a mirror image near Nyquist (a 440 Hz test tone appears at 219.7 Hz
> and 3,780 Hz). Notebook `05b` verifies the format before any D3 result is
> trusted; notebook 06 compares both decodings on identical splits (int16
> overstates the recording-level step by about 2 pp; accuracy differences up
> to 2.6 pp).

---

## Repository layout

```
LeakAudit/
├── src/                      # reusable library (imported by every notebook)
│   ├── config.py             #   paths, constants, pipeline settings, plotting style
│   ├── io_utils.py           #   dataset discovery, loaders, caches, SHA-256 manifest
│   ├── features.py           #   2023 pipeline: band-pass, window selection, mel features
│   ├── evaluation.py         #   models, split protocols, metrics, conformal prediction
│   ├── sfb.py                #   signal-free baseline + bootstrap intervals
│   ├── diagnostics.py        #   identity probe, label permutation, incremental value
│   ├── mitigation.py         #   ERM / Group-DRO / JTT (PyTorch)
│   └── reporting.py          #   number formatting, result log
├── notebooks/                # one notebook per experiment, run in order
│   ├── 00_setup_and_data_check.ipynb
│   ├── 01_load_D1.ipynb
│   ├── 02_E1-E3_structural_audit.ipynb
│   ├── 03_E4_protocol_collapse.ipynb
│   ├── 04_E6_time_confound_D2.ipynb
│   ├── 05b_verify_D3_raw_format.ipynb   # run before 05 and 06
│   ├── 05_D3_load_and_E8_window_heuristic.ipynb
│   ├── 06_E5_recording_leakage_D3.ipynb
│   ├── 07_E7_foundation_models_D2.ipynb
│   ├── 08_E10-E11_mitigation_and_conformal.ipynb
│   ├── 08b_R2_identity_diagnostics.ipynb
│   ├── 08c_factorial_extension_D2_D3.ipynb
│   ├── 08d_supplementary_checks.ipynb
│   ├── 09_figures.ipynb
│   ├── 10_E0_signal_free_baseline.ipynb
│   └── 11_bootstrap_confidence_intervals.ipynb
├── data_raw/                 # place the three datasets here (git-ignored)
│   ├── D1_aihub/             #   10 CSV files
│   ├── D2_hongkong/          #   Hydrophones / MEMS Accelerometers / Noise Loggers
│   └── D3_testbed/           #   Accelerometer / Dynamic Pressure Sensor / Hydrophone
├── data/                     # derived feature caches (git-ignored, auto-created)
├── results/                  # machine-readable results (CSV / JSON)
├── figures/                  # publication figures (PNG / PDF, 600 dpi)
├── data_manifest.txt         # SHA-256 of every raw file
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting the data

None of the raw data is redistributed here.

**D1 — AI-Hub dataset 138 (수도관로 누수 감지 데이터).**
Register at <https://aihub.or.kr> and download dataset no. 138 (domestic
access). Place the ten CSV files directly under `data_raw/D1_aihub/`. The
public description quotes ~50,000 records over 512 frequency columns; the
distributed copy analysed here contains **70,384 records** with **513 spectral
bins** (0–5,120 Hz at 10 Hz, both endpoints included). The full column
inventory is printed by notebook 01. Per-file SHA-256 checksums identifying
the exact copy are listed in `data_manifest.txt`.

**D2 — Hong Kong acoustic dataset.** CC BY 4.0.
<https://doi.org/10.17632/hkn8mxcjyz.1> → *Download All*. Extract so that
`data_raw/D2_hongkong/` contains the `Hydrophones/`, `MEMS Accelerometers/`
and `Noise Loggers/` folders (a single extra nesting level is handled
automatically).

**D3 — Aghashahi testbed dataset.** CC BY 4.0.
<https://doi.org/10.17632/tbrnp6vrnj.1> → *Download All*. Extract so that
`data_raw/D3_testbed/` contains `Accelerometer/`, `Dynamic Pressure Sensor/`
and `Hydrophone/`.

Run **notebook 00** first — it prints a per-dataset inventory, flags any
missing or still-zipped folders and writes `data_manifest.txt`.

---

## Running

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in this order:

```
00 → 01 → 02 → 03 → 04 → 05b → 05 → 06 → 07 → 08 → 08b → 08c → 08d → 09 → 10 → 11
```

**Run 05b before 05 and 06.** It verifies that the D3 `.raw` files are being
decoded correctly; every D3 result depends on it. Feature caches store row
identifiers (`data/*_v2.npz`) and are rebuilt from `data_raw/` if missing.

Notebooks 03 and 08b do the bulk of the computation and resume from their own
result files; everything runs on CPU. Foundation-model embeddings in notebook
07 download pre-trained weights on first use. Notebook 08 needs PyTorch for
Group-DRO and JTT. Figures are saved as PNG and PDF at 600 dpi in greyscale.
For a quick test, set `LEAKAUDIT_QUICK=1` before starting the kernel (fewer
seeds and bootstrap replicates; not for reporting).

Set the repository root explicitly if you launch Jupyter from elsewhere:

```python
from src import config as cfg
cfg.set_root(r"/path/to/LeakAudit")
```

---

## Citing the datasets

```bibtex
@misc{aihub138,
  title  = {Water pipe leak detection dataset (dataset no.\ 138)},
  author = {{National Information Society Agency (Korea)}},
  year   = {2021}, howpublished = {AI-Hub}, note = {\url{https://aihub.or.kr}}
}
@misc{tijani2022hk,
  title     = {Acoustic Based Data Acquisition for Leak Detection of Water
               Distribution Networks},
  author    = {Tijani, Ibrahim A. and Tariq, Salman and Zayed, Tarek and Hu,
               Zhongyu and Bakhtawar, Beenish and Abdulmageed, Sherif and Liu,
               Rongsheng and Fares, Ali},
  year      = {2022}, publisher = {Mendeley Data},
  doi       = {10.17632/hkn8mxcjyz.1}, note = {CC BY 4.0}
}
@misc{aghashahi2022testbed,
  title     = {Dataset of Leak Simulations in Experimental Testbed Water
               Distribution System},
  author    = {Aghashahi, Mohsen and Sela, Lina and Banks, M. Katherine},
  year      = {2022}, publisher = {Mendeley Data},
  doi       = {10.17632/tbrnp6vrnj.1}, note = {CC BY 4.0}
}
```

---

## Authors

- **Jiseong Kim** — Graduate School of AI and Software, Sogang University, Seoul, Republic of Korea · <kjs1000@sogang.ac.kr> · ORCID 0009-0001-6720-0045
- **Munil Yang** (corresponding) — Institute for Industrial Policy Studies, Seoul, Republic of Korea · <miyang@ips.or.kr> · ORCID 0009-0004-1493-6691

---

## Licence

Code: MIT (see `LICENSE`). Data: as distributed by each provider (see above).
