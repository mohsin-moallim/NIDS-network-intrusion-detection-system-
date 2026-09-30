# Graticule design record

This file records the design decisions made or approved by the project owner (mohsin-moallim), each with the reason for it.
New entries are added as the build moves through its phases. Dates are ISO (YYYY-MM-DD).

## Decisions approved on 2026-10-01 (initial plan)

### Name and identity
- **Name: Graticule.** Tagline: *"A measuring bench for training, testing and comparing network-intrusion classifiers."*
  A graticule is the ruled grid on an oscilloscope screen or survey map that every trace is read against. The name
  frames the app as an instrument: every model is a measuring *channel* and every result is a *reading* shown with its
  sample size. That pushes the interface towards honest reporting (balanced accuracy and test-set class counts sit next
  to every score). It was chosen over two alternatives, *Inkling* (analyst's casebook) and *Pratique* (harbour clearance).
- **Palette.** Graphite dark mode (`#12171C`) and paper light mode (`#F3F1EA`), one enamel-green control colour, a brass
  secondary, and three reserved semantic colours: blue = normal, vermilion = attack, signal yellow/amber = alert.
  It deliberately avoids the black + neon cyan + purple look. Every text pair passes WCAG contrast (body ≥ 7:1,
  secondary ≥ 4.5:1, graphics ≥ 3:1) in both modes, and normal vs attack were checked under simulated colour blindness.
- **Never colour alone.** Normal is always "○ Normal", attack "◆ Attack" (or the class name), alerts "▲ Alert";
  model channels differ by dash pattern and marker as well as colour.
- **Fonts.** Instrument Sans (headings), Atkinson Hyperlegible Next (body) and Atkinson Hyperlegible Mono (numbers), all
  SIL OFL, self-hosted in `static/fonts/` so the UI and the PDF match offline. Atkinson Hyperlegible keeps look-alike
  characters (1/l/I, 0/O) distinct, which matters when reading feature names and exact values.
- **Icon.** A small oscilloscope screen with a flat trace and one sharp deflection, peak marked in vermilion.
- **Navigation.** A numbered stepper — 01 Sample · 02 Fit · 03 Measure · 04 Probe · 05 Assay · 06 Sweep · 07 Record — plus
  two utilities, Logbook and Bench. Testing a detector is an ordered procedure (no reading without a fit, no fit without a
  sample), so numbered stations teach the order, while any station can still be opened directly.

### Platform and tools
- **Python 3.12+ (tested on 3.13).** The brief said 3.10+, but the current stable numpy (2.5.3) and xgboost (3.4.1) need
  3.12 or newer; staying current was preferred over older pins.
- **Streamlit 1.64** for the UI. Compared with Dash, NiceGUI and Panel, it is the only one that covers every need with
  built-in features: auto-refreshing fragments for live panels, sidebar/top navigation, separate light and dark themes
  with custom fonts, uploads/downloads and headless testing (`AppTest`). Its rerun-on-every-interaction model is the
  known risk; the architecture below removes it and a test proves it.
- **Altair 6 + vl-convert** for charts: one chart definition renders on screen and as a PNG in the PDF without any
  browser. (Plotly's exporter needs Chrome; matplotlib would mean drawing every chart twice.)
- **fpdf2** for the PDF (pure Python, real tables, embedded fonts).
- **No SHAP dependency.** It would pull in numba/llvmlite, which lag behind numpy. XGBoost's built-in exact contributions
  plus a model-agnostic "reference swap" method cover explanations.
- **Persistence:** joblib for scikit-learn pipelines, XGBoost's native `.ubj` format (the only format XGBoost guarantees
  across versions), SQLite for run history (atomic writes, part of Python).
- **Package `xgboost`** (not the smaller `xgboost-cpu`), for recognisability.

### Architecture
- Core library `graticule/` holds all data and ML logic and never imports Streamlit (enforced by a test); the UI lives in
  `app.py` and `ui/`. This keeps the logic testable without the web framework.
- **No retraining on unrelated interaction.** Settings live in forms, training starts only from the Fit button, runs in a
  background thread that never touches the UI, and pages only read stored results. Finished runs are also kept in a
  process-wide registry so a browser refresh does not lose them.
- **Data folder:** in-app setting (saved in the git-ignored `local_settings.json`) wins over the `NIDS_DATA_DIR`
  environment variable; with neither, the app uses its synthetic generator. Only `*.csv` directly inside the folder are read.
- **No copies of the dataset anywhere in the project**, including inside saved models. Replay after a restart rebuilds the
  same held-out split from the data folder (recorded settings + seed, checked by a fingerprint). The "predicts identically"
  check uses generated probe vectors, not dataset rows.

### Data handling
- Duplicates are removed in stages before the split (within each file, across files, and again in the chosen feature
  space, because rows that differed only by `Destination Port` become identical once it is excluded); every count is shown.
- Rows with identical features but different labels are **kept and counted** by default; dropping them would flatter scores.
- The minimum class count (default 50) applies in **multi-class mode only**; a hard floor of 10 rows per class applies in both.
- `Destination Port` is opt-in (off by default) because it can act as a shortcut that inflates scores.
- Everything fitted from data (scalers, imputers, top-K ranking, SVM subset, calibration, early stopping) uses the training
  split only.

### Models (four families, five channels)
| Channel | Model | Why |
|---|---|---|
| CH1 | Random forest (150 trees grown 25 at a time, √features, min leaf 2, 50% bootstrap) | Strong, robust bagged baseline; growing in chunks gives progress and a cancel point |
| CH2 | XGBoost (hist, depth 8, learning rate 0.15, ≤300 rounds with early stopping on a training slice) | Usually the most accurate on tabular flow data; early stopping keeps multi-class time bounded on a laptop |
| CH3 | RBF SVM (C=10, gamma "scale"), capped at 20,000 training rows (adjustable 2k–50k), sigmoid-calibrated on a separate slice | The kernel family; fit time grows at least quadratically, so the cap is automatic and shown in the UI |
| CH4 | Neural net: MLP 128→64, adam, early stopping | The neural-network family, light enough to train on a CPU |
| CH5 | Logistic regression (lbfgs, C=1) | Fast linear baseline that shows how much the non-linear models add |
- Scale-sensitive channels (SVM, MLP, logistic regression) get a signed-log transform and standard scaling inside their
  pipelines, because flow features are extremely long-tailed and some contain negative placeholders.
- Balanced class weights are on by default (capped, then rescaled) and balanced accuracy is always shown.
- The combined verdict is an equal-weight average of channel probabilities plus an agreement count.

### Process
- Built phase by phase; each phase ends with the app running, `pytest` passing, this file updated and one commit.
- Commits use the owner's git identity. Nothing is pushed to any remote.

## Phase 0 — environment and skeleton (2026-10-01)
Measured on the target laptop (i5-8365U, 16 GB, Windows 11, Python 3.13):
- **Install:** all pins install from wheels (`--only-binary=:all:`), `pip check` is clean. `requirements-lock.txt`
  records the full resolved set.
- **Reading Wednesday (215 MB, 692,703 rows):** pyarrow engine 1.0 s with a ~1.5 GB working-set peak; C engine 6.3 s
  with ~0.5 GB. Decision: pyarrow first (fast), one file at a time with an immediate float32 cast so the peak stays
  per-file; C engine as the fallback.
- **The repeated `Fwd Header Length` column** keeps the *same name* under the pyarrow engine (only the C engine renames
  it to `.1`), so duplicates are removed by position after checking the two columns are identical (they are).
- **Bad values confirmed:** `Flow Bytes/s` has 289 +inf and 1,008 NaN, `Flow Packets/s` 1,297 +inf in Wednesday;
  eleven columns contain negatives (e.g. `-1` "not seen" markers in `Init_Win_bytes_*`, negative durations/IATs).
  This is why scale-sensitive channels use a *signed* log transform.
- **Duplicates:** 81,948 exact duplicate rows inside the Wednesday file alone.
- **Labels:** the Thursday web-attack labels hold a valid UTF-8 U+FFFD where the dash was; normalised to " - ".
- **Charts to PDF offline:** Altair → vl-convert PNG (0.7 s) → fpdf2 works with no browser and with the bundled
  fonts, including variable-font weights. The body font has no ○ ◆ ▲ glyphs, so the PDF draws those marks as shapes.
- **Navigation:** Streamlit's navigation menu is hidden and replaced by a custom stepper strip of page links; the brass
  index mark and completion ticks are small CSS on keyed containers.
