# LeakAudit

**A forensic validity audit of public acoustic leak-detection benchmarks for water distribution networks.**

Reported accuracies in acoustic leak detection routinely exceed 90 %. This
repository asks a different question: *what are those numbers measuring?*
Across three heterogeneous public benchmarks we hold the learning problem
fixed and vary only the evaluation design, and show that most of the reported
performance is produced not by learning leak acoustics but by three distinct,
avoidable evaluation flaws — each arising through a different mechanism. We
provide deployment-realistic protocols, a signal-free baseline as a mandatory
reporting item, and a coverage-guaranteed abstention layer as the constructive
remedy.

The work is prepared for the *Journal of Hydroinformatics* Special Issue
*AI/ML and Data-Driven Methods for Monitoring, Modelling and Decision Support
in Human-Impacted Water Systems* (IWA Publishing). Code comments follow
British English (IWA house style).

---

## Three benchmarks, three failure modes

| Dataset | Role | Failure mode exposed | Mechanism |
|---|---|---|---|
| **D1** AI-Hub 138 (Korea, 70,384 rows, tabular spectra) | main evidence | site / sensor memorisation | sensor identity + `lrate` target proxy |
| **D2** Hong Kong field recordings (160 public files, WAV) | independent replication | time-of-day confound | recording time alone reaches 84.8 % |
| **D3** Aghashahi testbed (120 hydrophone recordings, RAW + CSV) | controlled control group | recording-level leakage | segment / condition split policy |

The three benchmarks differ deliberately in country, sensor, format and
scale. That the same class of validity risk recurs through unrelated
mechanisms across unrelated datasets is what makes it a reproducible concern
rather than a quirk of any one benchmark. The failure-mode framing follows the
data-leakage taxonomy of Kapoor & Narayanan (2023) and the empirical-audit
framing of McGreivy & Hakim (2024).

---

## Headline results (10 seeds, real data)

- **The signal-free baseline (SFB).** Each benchmark is partly solvable with
  the signal untouched. On the Korean data the device field `lrate` still
  reaches **0.60** (five-class) and **0.81** (leak vs non-leak) even across
  held-out sites; on the Hong Kong sample a one-feature stump on *recording
  time* reaches **0.85** without ever opening the audio; on the controlled
  testbed the metadata margin is **exactly zero**, the behaviour a valid
  benchmark should show. We propose SFB as a mandatory reporting item: an
  acoustic score is evidence of acoustic discrimination only once it clearly
  exceeds it.
- **Two distinct mechanisms, separated.** Sensor identity attains a *perfect*
  in-sample score (every one of 2,710 sensors is single-class) but falls to the
  majority-class level across sites — memorisation that does not transfer. The
  device field `lrate`, by contrast, stays high across held-out sites —
  a target-proxy shortcut that does transfer.
- **Protocol dominates model.** In a factorial variance attribution over
  4 models × 4 protocols × 10 seeds (480 fits per task), the evaluation
  **protocol accounts for 79–84 %** of the sum of squares in accuracy, whereas
  the **model family accounts for 2.5–2.6 %**. Eleven of the twelve task × model
  cells sit at or below the majority-class baseline once the evaluation
  reflects deployment.
- **D1 — protocol collapse.** Five-class accuracy falls **0.89 → 0.33** from a
  naive random split to a site-and-temporal split (at or below the
  majority-class baseline). Outdoor-vs-indoor falls **0.91 → 0.51** and is
  *below* baseline under every non-naive protocol.
- **D2 — listening to the clock.** A one-feature stump on *recording time*
  reaches **0.85** without opening the audio (leak clips ≈ 03:50, no-leak
  clips ≈ 10:11). The claim is not a clean night/day split but that the leak
  class carries no temporal diversity.
- **D3 — leakage decomposed.** In the site-free testbed, accuracy falls
  **0.9881 → 0.9052 → 0.5950** across segment-random, recording-independent and
  condition-independent splits. The steps isolate recording-level leakage
  (**+8.29 pp**) and an unseen-condition shift (**+31.02 pp**); the confidence
  intervals do not overlap, so both effects are real.
- **Mitigation fails; abstention is honest.** Two representative robustness
  methods (Group-DRO, JTT) do not beat ERM — the confound is in data
  collection, not the algorithm. Split / Mondrian conformal prediction attains
  its target coverage while reporting a high, honest abstention rate.

Exact numbers are written to `results/` when the notebooks are run.

> **Note on D3 raw format.** The Aghashahi hydrophone `.raw` files are
> **32-bit little-endian PCM, mono, 8 kHz**, and the provider's own converter
> keeps only the first 240,000 samples (30 s). Reading them as int16 splits
> each 32-bit word in two, halving every frequency and injecting a large
> artificial component near Nyquist. Notebook `05b` verifies this before any
> D3 result is trusted. An earlier int16 reading had *manufactured* a
> spuriously stronger result — a point we now use as evidence for including
> format verification in a reproducibility checklist.

---

## Repository layout

```
LeakAudit/
├── src/                      # tested, reusable library (imported by every notebook)
│   ├── config.py             #   paths, constants, plotting style
│   ├── sfb.py                #   signal-free baseline + bootstrap intervals
│   ├── io_utils.py           #   dataset discovery + loaders (D1/D2/D3)
│   ├── features.py           #   2023 pipeline: band-pass, window selection, mel features
│   └── evaluation.py         #   models, split protocols, metrics, conformal prediction
├── notebooks/                # one notebook per experiment, run in order
│   ├── 00_setup_and_data_check.ipynb
│   ├── 01_load_D1.ipynb
│   ├── 02_E1-E3_structural_audit.ipynb
│   ├── 03_E4_protocol_collapse.ipynb
│   ├── 04_E6_time_confound_D2.ipynb
│   ├── 05_D3_load_and_E8_window_heuristic.ipynb
│   ├── 05b_verify_D3_raw_format.ipynb   # BLOCKING check before trusting D3
│   ├── 06_E5_recording_leakage_D3.ipynb
│   ├── 07_E7_foundation_models_D2.ipynb
│   ├── 08_E10-E11_mitigation_and_conformal.ipynb
│   ├── 09_figures.ipynb
│   ├── 10_E0_signal_free_baseline.ipynb
│   └── 11_bootstrap_confidence_intervals.ipynb
├── data_raw/                 # place the three datasets here (git-ignored)
│   ├── D1_aihub/             #   10 CSV files
│   ├── D2_hongkong/          #   Hydrophones / MEMS Accelerometers / Noise Loggers
│   └── D3_testbed/           #   Accelerometer / Dynamic Pressure Sensor / Hydrophone
├── data/                     # derived feature caches (git-ignored, auto-created)
├── results/                  # machine-readable results (CSV / JSON)
├── figures/                  # publication figures (PNG)
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
distributed copy analysed here contains **70,384 rows over 513 columns**.
Per-file SHA-256 checksums identifying the exact copy are listed in
`data_manifest.txt`.

**D2 — Hong Kong acoustic dataset.** CC BY 4.0.
<https://doi.org/10.17632/hkn8mxcjyz.1> → *Download All*. Extract so that
`data_raw/D2_hongkong/` contains the `Hydrophones/`, `MEMS Accelerometers/`
and `Noise Loggers/` folders (a single extra nesting level is handled
automatically).

**D3 — Aghashahi testbed dataset.** CC BY 4.0.
<https://doi.org/10.17632/tbrnp6vrnj.1> → *Download All*. Extract so that
`data_raw/D3_testbed/` contains `Accelerometer/`, `Dynamic Pressure Sensor/`
and `Hydrophone/`.

Run **notebook 00** first — it prints a per-dataset inventory and flags any
missing or still-zipped folders.

---

## Running

```bash
pip install -r requirements.txt
jupyter lab            # then run notebooks/ in numerical order
```

**Run 05b before 05 and 06.** It verifies that the D3 `.raw` files are being
decoded correctly; every D3 result depends on it. If you have caches built
before this was fixed, delete `data/d3_*.npz` and re-run 05 and 06.

Running 00 → 11 in order always works. Notebook 03 (protocol collapse) and the
audio notebooks (04–07) do the bulk of the computation; everything runs on
CPU. Foundation-model embeddings in notebook 07 download pre-trained weights
on first use and fall back gracefully if unavailable. Result tables are
written to `results/`; figures to `figures/`.

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