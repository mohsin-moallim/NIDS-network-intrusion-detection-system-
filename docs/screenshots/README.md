# Screenshots

The main [README](../../README.md) shows one image per station. The image files are not in the repository yet; this
note fixes what each one should show and how to take it, so the set comes out consistent whoever captures it.

## The eight images

| File | Station | Alt text used in the README |
|---|---|---|
| `01-sample.png` | 01 Sample | 01 Sample: the sample sheet |
| `02-fit.png` | 02 Fit | 02 Fit: options, progress and first readings |
| `03-measure.png` | 03 Measure | 03 Measure: leaderboard and confusion matrices |
| `04-probe.png` | 04 Probe | 04 Probe: one flow, every verdict and an explanation |
| `05-assay.png` | 05 Assay | 05 Assay: a scored file |
| `06-sweep.png` | 06 Sweep | 06 Sweep: the live panel |
| `07-record.png` | 07 Record | 07 Record: the PDF record and CSV exports |
| `logbook.png` | Logbook | Logbook: saved channel sets and run history |

Use exactly these names (lower case, `.png`) so the README links resolve.

## One session for the whole set

Take every image from one session, in station order, so all of them show the same run:

1. **Bench:** data folder set to the CIC-IDS2017 files; every other value at its default (row budget 200,000, drop
   bad values, SVM cap 20,000, test share 0.25, seed 42, alert threshold 0.90).
2. **01 Sample:** files Tuesday, Wednesday and Friday afternoon (PortScan); row budget 200,000; *Draw sample*.
3. **02 Fit:** Binary, Curated, all five channels, balanced class weights on; *Fit*.
4. Then visit 03 to 07 and the Logbook as described below.

Three files give the class chart and the Sweep's attack mix several attack types to show, while a binary fit keeps
the confusion matrices small enough to read in a screenshot.

## How to capture

- Start the app as usual: `.\.venv\Scripts\python.exe -m streamlit run app.py`.
- Use the **light** theme (⋮ menu at the top right, under Theme), the one the PDF record uses.
- Browser window about 1440 px wide, page zoom 100 %. The stepper should be visible at the top of every image.
- Capture with the Windows Snipping Tool (Win+Shift+S, rectangle) or the browser's own capture (Edge: Ctrl+Shift+S),
  trimming the browser frame. Save as PNG straight into this folder.
- Aim for under about 500 KB per image; crop rather than scale down, so the numbers stay sharp.

## Keep out of the images

- **Dataset rows.** The project stores no dataset rows by default (only CH3's support vectors, when you choose to
  save CH3 with a channel set), and its pictures should hold none either. At 04 Probe use
  *A typical flow* (per-feature medians of one class, not a recorded flow) and keep the "All … feature values" panel
  closed; at 05 Assay capture the readings and scroll the "First … scored rows" table out of view.
- **Personal paths.** The Bench, 01 Sample and the Logbook print folder paths. Use a neutral data folder such as
  `D:\datasets\CIC-IDS2017` while capturing, or crop the paths out.
- Notifications, other browser tabs and anything else from the desktop.

## What each image should show

**`01-sample.png`: 01 Sample.** The sample sheet just after a draw, scrolled so the bottom of the form is still in
view: the reading cards (rows read, bad-value rows, duplicates within and across files, rows kept, rows sampled,
classes), and the *Classes before and after sampling* table beside its chart, where the rare classes' floors show.

**`02-fit.png`: 02 Fit.** The Fit form with the options above, and below it the *Readings* table of the finished fit:
five channel rows with rows used, fit seconds, flows per second, accuracy, balanced accuracy and macro F1, and the
CH3 note "trained on 20,000 of … rows". If you can, take a second capture during the fit to show the progress panel;
keep the finished one as `02-fit.png`.

**`03-measure.png`: 03 Measure.** The *Readings overview* leaderboard (sorted by balanced accuracy, with the gap to the
best channel) and, below it, the first row of *Confusion matrices* in *Row %* view. The ○ Normal and ◆ Attack marks
should be readable.

**`04-probe.png`: 04 Probe.** Flow source *A typical flow* for one attack class; the *Verdict* table for all
channels with the Consensus line ("… of 5 channels agree"); and the *Explanation* chart for CH2 XGBoost with
*Exact contributions (XGBoost, log-odds)*, the bars in the blue-to-vermilion scale.

**`05-assay.png`: 05 Assay.** After scoring a labelled file the run has not seen (for example the Friday afternoon
DDoS file) with the Consensus: the upload, channel and threshold controls, the reading cards (rows scored, attacks
found, alerts, accuracy and balanced accuracy) and the confusion matrix. Not the scored-rows table.

**`06-sweep.png`: 06 Sweep.** A running replay of held-out rows with CH1 Random forest at a pace of 25 flows per
1 s tick, attack share *Natural*: the status line, *Live readings* (live accuracy and balanced accuracy), the
detections timeline and the ▲ High-confidence alerts list with a few entries.

**`07-record.png`: 07 Record.** After *Build PDF record*: the *PDF record* section with its reading cards (pages
among them) and *Download PDF record*, and the *CSV exports* list below it with *Download all as ZIP*. A second,
optional image may show the first page of the PDF itself, opened in a PDF viewer.

**`logbook.png`: Logbook.** After *Save the current fit*: the *On the bench* panel naming the saved folder (with the
note that CH3 stays out), the *Saved channel sets* table with its Check column reading "manifest intact", and the top
of the *Run history* table.

## Adding them

Put the PNG files here and commit them with the README unchanged; GitHub and most Markdown viewers then show them in
place. Until then the README shows each image's alt text instead.
