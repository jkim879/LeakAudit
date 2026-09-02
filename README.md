# LeakAudit

**An empirical audit of evaluation validity in acoustic leak detection for water distribution networks.**

Reported accuracies in acoustic leak detection routinely exceed 90 %. This
repository asks a different question: *what are those numbers measuring?* Across
three public benchmarks we show that most of the reported performance is
produced not by learning leak acoustics but by three distinct, avoidable
evaluation flaws — and we provide deployment-realistic protocols and a
coverage-guaranteed abstention layer as the constructive remedy.

The work is prepared for the IWA special issue *AI/ML and Data-Driven Methods
for Monitoring, Modelling and Decision Support in Human-Impacted Water Systems*.
Code comments follow British English (IWA house style).

---

## Three benchmarks, three failure modes

| Dataset | Role | Failure mode exposed |
|---|---|---|
| **D1** AI-Hub 138 (Korea, 70 384 rows, tabular spectra) | main evidence | **F2** site / sensor memorisation |
| **D2** Hong Kong field recordings (160 files, WAV) | independent replication | **F4** time-of-day confound |
| **D3** Aghashahi testbed (280 signals, CSV + RAW) | controlled control group | **F1** recording-level leakage |

The three failure modes map onto the data-leakage taxonomy of
Kapoor & Narayanan (2023); the empirical-audit framing follows
McGreivy & Hakim (2024); the reporting-practice context is
Oberascher et al. (2026).

---

## Headline results (10 seeds, real data)

- **The signal-free baseline (SFB).** Each benchmark is partly solvable with the
  signal untouched. On the Hong Kong sample a one-feature stump on *recording
  time* reaches **0.85** without ever opening the audio; on the Korean dataset
  the device field `lrate` still reaches **0.60** (five-class) and **0.81**
  (leak vs non-leak) even across held-out sites. We propose SFB as a mandatory
  reporting item: an acoustic score is only evidence of acoustic discrimination
  once it clearly exceeds it.
- **Two distinct mechanisms, separated.** Sensor identity attains a *perfect*
  in-sample score (every one of 2 710 sensors is single-class) but falls to the
  majority-class level across sites — memorisation that does not transfer.
  The device field, by contrast, stays high everywhere — target leakage that
  does.
- **Protocol choice dominates model choice.** Across four learner families
  (gradient boosting, random forest, logistic regression, a neural network) the
  between-model spread is up to 27 accuracy points under a naive random split
  and about one point under the deployment protocol. Protocol accounts for
  roughly **32x** the variance that model choice does, and 11 of the 12
  task x model cells sit at or below the majority-class baseline once the
  evaluation reflects deployment.
- **D1 — a device field is the label.** The logger's `lrate` output reproduces
  the alarm/normal label with accuracy **1.000000**; every one of 2 710 sensors
  is single-class, so sensor identity alone predicts the label perfectly.
- **D1 — protocol collapse.** Five-class accuracy falls **0.89 → 0.33** from a
  naive random split to a site-and-time split (below the majority-class
  baseline). Outdoor-vs-indoor falls **0.91 → 0.51** and is *below* baseline
  under every non-naive protocol — that decision is not learnable from a
  single-site spectrum.
- **D2 — listening to the clock.** A one-feature stump on *recording time*
  reaches **0.85** accuracy without ever opening the audio (leak clips ≈ 03:50,
  no-leak clips ≈ 10:11).
- **D3 — leakage without sites.** In the site-free testbed, the split policy
  alone moves accuracy substantially. *These numbers are being regenerated
  after a raw-format correction (see below); treat the D3 figures as pending.*
- **Mitigation fails; abstention is honest.** Group-DRO and JTT do not beat ERM
  (the confound is in data collection, not the algorithm), whereas split /
  Mondrian conformal prediction attains its target coverage and reports a high,
  honest abstention rate.

Exact numbers are written to `results/` when the notebooks are run.

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
├── RESEARCH_IDEA_KR.md       # the research idea, corrections and positioning (Korean)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting the data

None of the raw data is redistributed here.

**D1 — AI-Hub dataset 138 (수도관로 누수 감지 데이터).**
Register at <https://aihub.or.kr> and download dataset no. 138. Place the ten
CSV files directly under `data_raw/D1_aihub/`.

**D2 — Hong Kong acoustic dataset.** CC BY 4.0.
<https://doi.org/10.17632/hkn8mxcjyz.1> → *Download All*. Extract so that
`data_raw/D2_hongkong/` contains the `Hydrophones/`, `MEMS Accelerometers/` and
`Noise Loggers/` folders (a single extra nesting level is handled automatically).

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

### Run order

Notebooks are numbered in dependency order, but not every one depends on the
previous. The binding constraints are:

| notebook | requires |
|---|---|
| 00, 01, 04, 05, 05b, 06, 07 | nothing beyond the raw data |

> **Run 05b before 05 and 06.** It verifies that the D3 `.raw` files are being
> decoded correctly; every D3 result depends on it.
>
> The Aghashahi hydrophone `.raw` files are **32-bit little-endian PCM, mono,
> 8 kHz**, and the provider's own converter keeps only the first 240 000
> samples (30 s). Reading them as int16 splits each 32-bit word in two, which
> halves every frequency and injects a large artificial component near
> Nyquist. If you have caches built before this was fixed, delete
> `data/d3_*.npz` and re-run 05 and 06.
| 02, 03, 08 | 01 (the D1 cache) |
| 09 (figures) | 01, 03 |
| 10 (signal-free baseline) | 01, 03, 05, 06, 07 |
| 11 (bootstrap intervals) | 03, 05, 06, 07, 08 |

Running 00 → 11 in order always works. Notebook 03 (protocol collapse) and the
audio notebooks (04–07) do the bulk of the computation; everything runs on CPU. Foundation-model embeddings in
notebook 07 download pre-trained weights on first use and fall back gracefully
if unavailable. Result tables are written to `results/`; figures to `figures/`.

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
  author    = {Tijani, Ibrahim A. and Zayed, Tarek and Tariq, Salman and Hu,
               Zhongyu and Abdulmageed, Sherif and Liu, Rongsheng and Fares,
               Ali and Bakhtawar, Beenish},
  year      = {2022}, publisher = {Mendeley Data},
  doi       = {10.17632/hkn8mxcjyz.1}, note = {CC BY 4.0}
}
@misc{aghashahi2022testbed,
  title     = {Dataset of Leak Simulations in Experimental Testbed Water
               Distribution System},
  author    = {Aghashahi, Mohammad and Sela, Lina and Banks, M. Katherine},
  year      = {2022}, publisher = {Mendeley Data},
  doi       = {10.17632/tbrnp6vrnj.1}, note = {CC BY 4.0}
}
```

## Licence

Code: MIT (see `LICENSE`). Data: as distributed by each provider (see above).
