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
- Commits use the owner's git identity. Published by the owner to <https://github.com/mohsin-moallim/graticule>.

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

## Phases 1–2 — data core, 01 Sample, synthetic generator, feature sets (2026-10-01)
Built concurrently by two implementers, then integrated, reviewed through three lenses (spec conformance, correctness,
real data), with 17 verified findings fixed.
- **Reading.** pyarrow first, C engine as fallback; text that is not valid UTF-8 comes back from pyarrow as raw bytes,
  which moves the reader on to Windows-1252 and then Latin-1. Repeated header names are removed by position after an
  equality check. Files to be *scored* keep unlabelled rows; files used for training drop and count them.
- **Pipeline order.** Per file: read → normalise labels → bad-value strategy → within-file duplicates; then cross-file
  duplicates (files in capture order, Monday first, so the earliest copy is kept) → optional Web Attack merge →
  rare-aware sample. Exact duplicates are found with a 64-bit hash over the 77 features (−0.0 folded to +0.0, ±inf to NaN)
  plus the label.
- **Caching.** Reading and hashing are cached once per file regardless of options; the strategy-specific stage is cached
  separately and only holds row positions. All 8 files: first draw ~17 s, peak working set ~2.1 GB; drawing again with
  other options ~4 s. A "Release cached files" button frees the memory.
- **Identical flows with different labels** are kept and counted by default (options: majority label, or drop).
  On Wednesday there are 47 such groups (94 rows).
- **Sampling.** Each class first gets up to a floor of max(1,000, 2 % of the budget) rows — shrunk to budget ÷ classes
  when needed — and the rest of the budget is shared in proportion to class size. All 11 Heartbleed rows survive a
  200,000-row Wednesday sample.
- **Split rule.** Every class gets at least 2 training and 2 test rows; a class with fewer than 4 rows is reported by name.
- **Degenerate columns.** Constant columns are always left out of "all numeric"; from a group of identical columns only
  the first is kept. On the real files `SYN Flag Count` equals `Fwd PSH Flags`, so this rule keeps the curated set at 28.
- **Fingerprint.** A sample's fingerprint covers the request, the (file, row) pairs and labels, but not the folder path,
  so moving the dataset does not invalidate saved work.
- **Synthetic generator (original design).** It simulates individual packets from traffic profiles and computes all 77
  columns with one shared flow meter, so derived columns always agree (subflows = totals, variance = std², rates =
  totals ÷ duration). Profiles: normal traffic is a mix of web sessions (55 %), DNS (20 %), keep-alive (13 %) and bulk
  downloads (12 %); attacks are Flood (30 %), Sweep (25 %), Credential Guess (20 %), Web Injection (15 %) and
  Slow Drip (10 %). A small "blur" share of attacks borrows normal-looking timing and sizes, and some normal flows look like
  refused connections, so the classes separate clearly but not perfectly (binary balanced accuracy ≈ 0.975 at the default
  blur, without the port). Like the real files, zero-duration flows produce infinite rates in `Flow Bytes/s` and
  `Flow Packets/s` only. 40,000 flows take about 1 s.
- **Top-K ranking** uses a small XGBoost on at most 50,000 training rows. Multi-class ranking is the slow part (~20 s for
  6 classes), so it runs inside the background fit job.
- **UI.** Everything that starts work on 01 Sample sits in a form whose only action is *Draw sample*; options repeat the
  last draw while the Bench holds the defaults. Charts fit their container's width but keep their height, and the
  station stepper scrolls sideways on narrow screens instead of stacking.

## Phase 3 — channels, background fitting, 02 Fit (2026-10-01)
Built by two implementers (core and UI) against a shared interface, integrated, reviewed through three lenses; 17 verified
findings fixed.
- **Measured on this laptop** (Wednesday, 200,000-row budget → 145,262 training / 48,421 test rows, curated features):

  | Channel | Binary fit s | Binary bal. acc. | Multi-class fit s | Multi-class bal. acc. |
  |---|---|---|---|---|
  | CH1 Random forest | 9.9 | 0.9989 | 7.6 | 0.9964 |
  | CH2 XGBoost | 5.4 | 0.9995 | 8.2 | 0.9969 |
  | CH3 RBF SVM (20,000-row cap) | 2.1 | 0.9947 | 2.4 | 0.9922 |
  | CH4 Neural net (MLP) | 24.1 | 0.9967 | 43.3 | 0.9941 |
  | CH5 Logistic regression | 3.0 | 0.9723 | 12.4 | 0.9751 |

  Tree channels are far inside the 90 s / 180 s budget; a full five-channel binary fit in the app takes about 40 s.
- **Class weights.** Every channel receives capped, balanced *sample* weights (no `class_weight` anywhere), solved so
  no row exceeds the cap and the weights still sum to n. Without the cap an 11-row class such as Heartbleed would weigh
  thousands of times more than a normal flow.
- **CH3 (SVM).** A calibration slice (≤ 5,000 rows, at least 20 per class where possible) is set aside first; the SVM is fit
  on a rare-aware draw of at most the cap from the rest and then calibrated (sigmoid) on that slice in one pass. The
  cap is shown as a badge: "CH3 trained on 20,000 of 145,262 rows (SVM cap 20,000)".
- **CH2 (XGBoost)** trains on 90 % of the training rows and uses the other 10 % to stop early.
- **Trees get no imputer**: scikit-learn 1.9 forests and XGBoost handle missing values themselves.
- **Background fitting.** The fit runs in a background thread that never touches the UI, one job per process. Cancel is
  checked between forest chunks, XGBoost rounds, MLP epochs, logistic-regression iterations and scoring batches, so it
  answers within about 2 s (the SVM's own libsvm fit is the one step that cannot be interrupted).
  The MLP's per-epoch hook overrides a private scikit-learn method; it is pinned to scikit-learn 1.9.1 and covered by tests.
- **Warnings from the fit thread** are routed per thread, so notes on one channel never pick up warnings raised elsewhere.
- **Memory.** Finished runs live in a process-wide registry of the last three; the job object releases its copy.
- **Top-K** ranking runs inside the job on at most 50,000 training rows (12 real classes: ~19 s), and a test proves test
  rows cannot change it. Rows that become identical over the chosen columns are removed before the split and reported.
- **No retraining on unrelated interaction** is enforced by design (forms, a button callback, pages that only read stored
  results) and proved by `tests/ui/test_no_retrain.py`, which changes every widget on every station after a fit and checks
  that the fit counter and the stored run object are unchanged.
- **Honest readings.** The page warns when the loaded sample differs from the one the run was fitted on; scores are shown
  to four decimals and a score short of 1 never prints as 1.0000 (0.9999 is the ceiling below 1); the fit time
  includes building the matrices.

## Phases 4–5 — saved channel sets, run history, Logbook; evaluation and 03 Measure (2026-10-01)
Built concurrently by two implementers, integrated, reviewed through three lenses; 32 verified findings handled.
- **Saved channel sets** (`saved_models/<run id>/`, git-ignored): a manifest (features, mode, classes, metrics, data source,
  row counts, timestamp, library versions, the fit and sample settings), the models (scikit-learn pipelines with joblib;
  XGBoost in its own `.ubj` format), 512 *synthetic* probe flows drawn from per-feature training quantiles together with
  every channel's readings of them, and the quantiles. Every file — and the manifest itself — carries a SHA-256 checksum;
  a changed byte is refused.
- **"Predicts identically" is checked on every load**: each channel re-reads the probe flows and must reproduce its saved
  labels and probabilities exactly (`array_equal`, forests on one thread). The set is *verified* only when the library
  versions (Python, numpy, pandas, scikit-learn, scipy, xgboost, joblib) also match; otherwise the differences are listed.
  Measured after a full server restart: "Verified: all 4 channels reproduced their saved probe readings exactly."
- **Held-out rows after a restart** are rebuilt from the data folder with the recorded settings and seed, then checked row
  by row, label by label and value by value against digests in the manifest. On the Thursday web-attack run the rebuilt
  38,562 test rows matched, and 03 Measure showed the same readings as before the restart (CH2 XGBoost 0.7439).
- **CH3 (kernel SVM) is not written to saved sets.** A kernel SVM *is* a set of training rows (its support vectors,
  standardised), and the brief asks that no dataset rows be copied into the project. Its readings are kept in the manifest
  and the Logbook says so plainly; refit it at 02 Fit when needed. *(Superseded on 2026-10-01 by the owner's decision below: CH3 can be saved by explicit opt-in.)*
- **No dataset rows on disk** otherwise: probes are synthetic, quantiles are summary statistics, and the training reference
  sample used for explanations is never written.
- **Run history** lives in SQLite (`run_history/runs.sqlite3`, git-ignored), recorded from the fit job itself so even a fit
  that no page picks up is logged; it survives restarts and downloads as CSV.
- **One run type everywhere.** A loaded set becomes an ordinary run (`origin = "loaded"`), so every station works the same;
  when its held-out rows cannot be rebuilt the stations say why instead of failing.
- **03 Measure**: a leaderboard sorted by balanced accuracy with the gap to the best channel, a dot plot of every metric
  (accuracy, balanced accuracy, precision/recall/F1 macro and weighted, ROC-AUC, average precision), the held-out class mix,
  confusion matrices for every channel (row % or counts; each keeps its size and scrolls in narrow columns), ROC and
  precision-recall curves (all channels overlaid for binary, one-vs-rest per class for multi-class), built-in and on-demand
  permutation importance, timing (fit seconds, flows per second, single-flow latency), and cross-validation on demand
  (stratified folds over at most 50,000 training rows, with a time estimate and Cancel). Evaluations are computed once per
  run and kept; only the two explicit buttons ever fit anything, and they share one work slot with 02 Fit.
- **Test suite**: ~100 s by default (421 tests); whole-file real-data checks and the 200k benchmarks run with `-m realdata`
  or `-m slow`.

## Phases 6–8 — 04 Probe, 06 Sweep, 05 Assay, 07 Record (2026-10-01)
Built by four implementers in parallel, integrated, reviewed through three lenses; 22 verified findings fixed.
- **04 Probe.** A flow comes from the held-out rows (filter by true class, draw another, or a row number — with its source
  file and data row), from a class-typical median, or from an editable table (with the 1st–99th percentile range as a
  guide). Every channel gives a verdict and probability, plus a consensus with its agreement count ("4 of 4 channels
  agree"). Explanations: **exact XGBoost contributions** (log-odds; the bars, the remaining features and the bias add up to
  the raw score, which a test checks) and a model-agnostic **reference swap** for every channel (approximate: each feature
  is replaced by values from 32 background training rows in one batched call). No SHAP dependency.
- **06 Sweep.** A seeded, UI-free engine replays **only held-out rows with their true labels** for real-data runs (a
  synthetic stream for synthetic runs), with adjustable pace (flows per tick × tick interval), attack share and attack mix.
  Live accuracy equals the accuracy over every flow emitted (tested). Only the live panel refreshes each tick; settings
  survive visits to other stations; nothing is fitted.
- **One alert rule everywhere.** An alert needs an *attack* verdict whose attack probability reaches the Bench threshold
  (binary: P(Attack); multi-class: 1 − P(normal)), shared by Probe, Assay, Sweep and the PDF.
- **05 Assay.** A CSV is read in blocks with the same header, encoding and bad-value handling as training (≈2.8× the file
  size in memory: Wednesday's 215 MB scored in 4.9 s). Missing feature columns are listed and nothing is scored; uploaded
  lines are repeated exactly in the download with the verdict, per-class probabilities, attack probability and alert flag
  appended; accuracy, balanced accuracy and a confusion matrix appear when the file has labels (unknown labels are
  counted and left out). A Wednesday-fitted run scores the Thursday web-attack file at balanced accuracy 0.50 — an honest
  sign that one day's model does not transfer to unseen attack types.
- **07 Record.** A PDF built offline with fpdf2 from the same chart builders as the screen, always in the light palette with
  the bundled fonts: cover with contents, sample sheet, fit settings and channel notes, readings with a consensus row,
  per-class table, confusion matrices, ROC and PR curves, importance, timing, Assay and Sweep summaries when present,
  notes and limitations, and the dataset citation. Normal/attack/alert marks are drawn as shapes because the fonts lack
  those glyphs. A 5-channel binary record is 9 pages, ~0.35 MB, built in ~7 s. CSV exports (leaderboard, per-class,
  held-out predictions without feature values, cross-validation, run history, Assay results, Sweep log) download one by
  one or as a ZIP with a README.
- **Results belong to the run that made them**: an Assay batch or a built PDF is offered only to that exact run object, so
  a fit and its loaded copy never mix their results.

## Phase 9 — README, definition-of-done audit (2026-10-01)
- **README.md** written from scratch (stations, features, how the readings stay honest, setup, dataset and citation,
  walkthrough, channels and metrics, testing, project structure, limitations, licence and credits), with screenshot
  placeholders described in `docs/screenshots/README.md`.
- **Definition of done verified with evidence:** a fresh venv installs from `requirements.txt` (`pip check` clean) and the
  app answers its health check; synthetic mode works end to end with no data folder; the no-retrain test changes 45
  widgets across every station; binary and multi-class both run on Wednesday and on Thursday web attacks; Monday in
  binary mode shows the single-class warning at 02 Fit and starts no job; the 200,000-row benchmark fits every tree
  channel in under 10 s with the SVM cap stated; a saved set reloads in a separate process and predicts identically;
  Sweep replays only held-out rows; Assay, the PDF and every CSV export work; nothing has been pushed anywhere.
- **Distinct flows, not traffic volume.** Removing exact repeats before the split (as required) means readings count
  each distinct flow once. Some flows are repeated thousands of times in the files (one DoS Hulk flow 9,329 times in
  Wednesday), so a whole-file Assay can differ from the held-out readings (CH1: 0.9989 balanced accuracy held out,
  0.9390 accuracy over the whole Wednesday file; CH2 XGBoost 0.9996). The app now says how many recorded flows the
  held-out rows stand for, and 05 Assay recognises rows the run was trained on and reports the other rows separately.
- **One rule each** for printing scores (four decimals, never 1.0000 below 1), for the alert (attack verdict and attack
  probability at or above the threshold, compared in float32 at every station), and for what counts as normal traffic.
- **Top-K ranking** is fitted through the same counted entry point as the channels, so the no-retrain test also covers it.
- **Test-suite time.** The suite held 589 tests at this point; the default run (553) took about 2.5 minutes on the test
  laptop on mains power, a little over the 2-minute aim set in the build spec, kept rather than moving the main UI journeys out of the default
  run; whole-file and benchmark tests run with `-m realdata` / `-m slow`.
- **CH3 and saved sets** was left open for the owner at this point; decided below.

## Owner decision — saving CH3 by opt-in (2026-10-01)
- **Decision (owner, option B):** CH3, the kernel SVM, stays **out** of saved channel sets by default, but the Logbook
  offers an explicit opt-in, **"Also save CH3 (RBF SVM)"**, unticked every time.
- **Why:** reloading all five channels after a restart matters to the owner, and the brief's "don't copy the data into the
  project" is still honoured by default. A kernel SVM *is* its support vectors (training rows after gap filling, signed
  logarithm and standardisation, which the saved scaler turns back into the original values), so saving it necessarily
  writes those rows. The opt-in makes that an informed, per-save choice instead of a hidden one.
- **How it shows:** the checkbox caption states how many training rows CH3 would write and where (`saved_models\<run id>\`,
  git-ignored, this machine only); the save notice repeats it; the manifest records `training_rows_inside`; the saved-sets
  table has a "training rows inside" column; a loaded set that holds CH3 says so. Sets saved without CH3 (including
  those saved before this change) load and verify exactly as before.
- **Built and checked (phase 10):** `persist.save_run(..., include_svm=True)` writes `svm.joblib`, checksums it and gives
  it probe readings; on load CH3 is verified like every channel and returns as an ordinary channel whose held-out
  predictions match the original exactly (checked in fresh processes, binary and multi-class). The owner's set saved
  earlier (without CH3) still loads and verifies unchanged.
- **No silent leftovers.** Saving and deleting work through hidden temporary folders; if Windows blocks a step, the
  delete puts the set back where possible (removing `svm.joblib` first), and any leftover folder — especially one still
  holding CH3's rows — is named in the Logbook with a button to remove it (only after it has been untouched for 30 s,
  so another session's save in progress is never offered).
- **Loading a set that holds CH3** re-scores its held-out rows with per-block progress, and the opt-in caption says so.
- **Test suite** now 603 tests (567 by default): about 2.5 minutes on mains power, about 4.5 on battery.

## Follow-ups — recorded-traffic readings, station addresses, polish, screenshots (2026-10-04)
- **Recorded-traffic readings (owner request).** Readings stay counted over *distinct* flows by default, because repeats
  must be merged before the split. 03 Measure, the PDF and the CSV exports add a clearly labelled estimate over the
  *recorded traffic*: each held-out row is weighted by `copies / f`, the rows of the cleaned files it stands for (exact
  repeats and rows made identical over the chosen columns) divided by its class's sampling share at 01 Sample. It is
  computed once per run, never refits, and is never the default.
- **Honest uncertainty.** The standard error covers only the spread among held-out rows, and says so. Because a few flows
  are recorded thousands of times (one DoS Hulk flow 9,329 times in Wednesday), the estimate can swing between draws;
  01 Sample now records how often every distinct flow occurs, 02 Fit lists the heavily repeated ones (≥ 100 copies and
  ≥ 0.5 % of their class) and which were held out, and 03 Measure / the PDF show a per-channel *range* covering every
  verdict those unseen heavy flows could have, plus a caution when they could move a reading by 0.01 or more. Checked on
  Wednesday over five draws: the whole file's true reading fell inside the range every time.
- **Every station has its own address** (`/sample`, `/fit`, … `/bench`). A hidden page at `/` hands over to 01 Sample on
  a session's first visit; a later visit to `/` (the Back button) draws 01 Sample in place, so Back can still leave the app.
- **Narrow screens.** The class chart prints its counts in a column of their own; the stepper scrolls the current station
  into view with a small static script (the app's only `unsafe_allow_javascript`, holding no user data).
- **Readable at 1440 px.** 03 Measure's leaderboard holds readings only (balanced accuracy first, compact headings); fit
  time, flows per second (whole numbers) and single-flow latency moved to its Timing table. 02 Fit picks channels with
  wrapping buttons and lists channel notes under its table; the Logbook tables were narrowed; 03 Measure earns its ✓.
- **Test suite.** The test-only model profile uses 10 XGBoost rounds and single-threaded forests (the full profile is
  unchanged); with lighter fixtures the default run is about 11 % faster: 605 tests in ~130 s on mains power. Restart
  checks (a saved set reloaded and the history read in a fresh process) stay in the default run.
- **README screenshots** were captured from a separate app instance with temporary settings, a neutral data-folder name
  and no dataset rows (04 Probe shows a typical flow; 05 Assay is cropped above the scored rows).
