# Screenshots

The main [README](../../README.md) shows one image per station, and a second one of 02 Fit while it fits. This note
records what each image shows and how the set was taken, so a later set comes out consistent whoever captures it.

## The images

| File | Station | Alt text used in the README |
|---|---|---|
| `01-sample.png` | 01 Sample | 01 Sample: the sample sheet |
| `02-fit.png` | 02 Fit | 02 Fit: options and the readings of a finished fit |
| `02-fit-progress.png` | 02 Fit | 02 Fit: the progress panel while the channels are fitted |
| `03-measure.png` | 03 Measure | 03 Measure: leaderboard and confusion matrices |
| `04-probe.png` | 04 Probe | 04 Probe: one flow, every verdict and an explanation |
| `05-assay.png` | 05 Assay | 05 Assay: a scored file |
| `06-sweep.png` | 06 Sweep | 06 Sweep: the live panel |
| `07-record.png` | 07 Record | 07 Record: the PDF record and CSV exports |
| `logbook.png` | Logbook | Logbook: saved channel sets and run history |

Use exactly these names (lower case, `.png`) so the README links resolve.

## One session for the whole set

Every image comes from one session, taken in station order, so all of them show the same run:

1. **Bench:** data folder set to the CIC-IDS2017 files; every other value at its default (row budget 200,000, drop
   bad values, SVM cap 20,000, test share 0.25, seed 42, alert threshold 0.90).
2. **01 Sample:** the Wednesday file only (DoS variants and Heartbleed: five attack types, enough for the Sweep's
   attack mix; the station offers every file in the folder at first, so remove the other seven); row budget 200,000;
   *Draw sample*.
3. **02 Fit:** Binary, Curated, all five channels (all five are chosen by default), balanced class weights on; *Fit*.
   A binary fit keeps the confusion matrices small enough to read in a screenshot.
4. Then 03 to 07 and the Logbook as described below.

The current set shows run `20261004-084132-03b4`: 200,000 sampled rows, 145,262 training and 48,421 held-out rows. A
new capture gives a new run id; replace it here.

## How the set was taken

- A separate app server on another port, started by a small launcher that first points
  `nids.settings.SETTINGS_FILE`, `MODELS_DIR` and `HISTORY_DIR` at a scratch folder (as relative paths, with that
  folder as the working directory). The owner's settings, saved sets and run history stay untouched, and the paths
  the Logbook prints read `saved_models\...` and `run_history\runs.sqlite3`.
- Microsoft Edge, headless, driven through the Chrome DevTools Protocol from a short Python script: stations opened
  with the stepper links (one browser session, so the run stays in memory), widgets set by clicking them, and each
  image taken with `Page.captureScreenshot` after the viewport was made as tall as the page. Before each capture the
  focused widget was blurred and the pointer moved to the corner, so no text caret, focus ring or hover state shows.
- 06 Sweep keeps redrawing while it streams, and a tall capture takes about a second, longer than one tick: a plain
  capture mixes two ticks (the cards one tick ahead of the chart and the alerts note). That image was taken the moment
  every panel showed the same tick, with the page's scripts held still for the capture (`Debugger.pause`, then
  `Debugger.resume`).
- **Light** theme (chosen under Theme in the ⋮ menu), 1440 px wide, device scale factor 2, so each image is
  2,880 px wide and the numbers stay sharp when GitHub scales it down. The stepper is at the top of every image, with
  a ✓ on every station done so far.
- Each PNG was then reduced to a 256-colour palette with Pillow (no dithering), which keeps text and charts visually
  unchanged and every file at about 260 KB or less.

By hand the same set can be taken with the Windows Snipping Tool (Win+Shift+S) or Edge's own capture
(Ctrl+Shift+S) at about 1440 px wide and 100 % zoom; crop rather than scale down.

## Kept out of the images

- **Dataset rows.** The project stores no dataset rows by default (only CH3's support vectors, when you choose to
  save CH3 with a channel set), and its pictures hold none either. 04 Probe uses *A typical flow* (per-feature
  medians of one class, not a recorded flow) with the "All … feature values" panel closed; 05 Assay is cropped above
  the "First … scored rows" table; 06 Sweep's feed and alert tables hold verdicts, probabilities and labels only.
- **Personal paths.** Of these stations only 01 Sample prints the data folder (in the caption above *Draw sample*;
  the Bench prints it too but has no image). For this set the
  folder's real path was replaced in the page with the neutral `C:\cicids2017` just before the capture; pointing the
  Bench at a folder with a neutral name gives the same result. The Logbook's paths are the relative ones above.
- Notifications (toasts) and the one-time "Saved run …" message at the Logbook (the station was opened again after
  saving), other browser tabs and anything else from the desktop.

## What each image shows

**`01-sample.png`: 01 Sample.** The station from the stepper down: the form with the Wednesday file chosen and
*Draw sample*, then the sample sheet: the reading cards (rows read, bad-value rows, duplicates within and across
files, rows kept, rows sampled, classes) and the *Classes before and after sampling* table beside its chart (each
class's sampled count in a column at the chart's right), where Heartbleed's 11 rows and the 4,000-row floor show. The
image ends above the *Sources* section.

**`02-fit.png`: 02 Fit.** The Fit form with the options above (all five channels chosen, their buttons wrapping onto a
second line), and below it the *Readings* table of the finished fit: five channel rows with status, rows used, fit
seconds, flows per second, accuracy, balanced accuracy and macro F1. Under the table, one line per channel note (CH2's
early stopping, CH3 "Trained on 20,000 of 145,262 training rows"), the held-out class mix, and the notes on repeated
rows and conflicting labels. `02-fit-progress.png` shows the same station during the fit: the progress panel with
each channel's status, share done, seconds, detail and Cancel.

**`03-measure.png`: 03 Measure.** The stepper with 01 to 03 ticked; the *Count each reading over* control at the top,
left at *Distinct flows*; the *Readings overview* leaderboard (sorted by balanced accuracy, which is its first score
column, with the gap to the best channel; fit time, flows per second and single-flow latency sit in the *Timing* table
further down the station, outside the image), the dot plot of every metric and the held-out class mix, and the first
row of *Confusion matrices* (CH1 to CH3) in *Row %* view.

**`04-probe.png`: 04 Probe.** Flow source *A typical flow* of the ◆ Attack class; the *Verdict* table for all channels
with the Consensus line ("5 of 5 channels agree"); and the *Explanation* chart for CH2 XGBoost with *Exact
contributions (XGBoost, log-odds)*, its feature labels (with the typical flow's values) shown in full.

**`05-assay.png`: 05 Assay.** The Friday afternoon DDoS file (a file and an attack type the run has not seen) scored
with the Consensus: the "Scored … flows" note, the upload, channel and threshold controls, the reading cards (rows
scored, rows with bad values, attacks found, alerts, seconds, accuracy and balanced accuracy, the rows the run trained
on and the readings over the other rows), the
verdict counts and the confusion matrix, and the notes below it. Not the scored-rows table.

**`06-sweep.png`: 06 Sweep.** A running replay of held-out rows with CH1 Random forest at 25 flows per 1 s tick,
attack share and mix *Natural*, at tick 80: the settings, the status line, the reading cards (live accuracy and
balanced accuracy, alerts, pace, attack share seen), the detections timeline, the feed and the ▲ High-confidence
alerts list, every panel at the same tick (see the capture notes above). The stream was paused before moving on, so
07 Record's PDF includes its summary.

**`07-record.png`: 07 Record.** After *Build PDF record* (Assay and Sweep included): the *PDF record* section with its
reading cards (pages, size, build time) and *Download PDF record*, and the *CSV exports* list below it with
*Download all as ZIP*.

**`logbook.png`: Logbook.** After *Save the current fit* (CH3 left unticked): the *On the bench* panel naming the saved
folder with the note that CH3 stays out, the *Saved channel sets* table (every column in view, its Check column
reading "manifest intact") with the line naming the saved-sets folder, the *Saved set* picker with Load and Delete,
and the *Run history* table (the run marked as saved) with its two buttons.

## Replacing them

Take a new set the same way, overwrite the PNG files here and commit them; the README links stay as they are.
