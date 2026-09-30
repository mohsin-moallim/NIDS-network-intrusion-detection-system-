# Graticule

*A measuring bench for training, testing and comparing network-intrusion classifiers.*

> **Status:** under construction. The shell, theme and settings are in place; the stations are being built phase by
> phase. This README is completed in the final phase.

Graticule learns to tell normal network flows from attacks using the labelled CIC-IDS2017 flow records (or a built-in
synthetic generator when no dataset is available). It fits several classifier families, measures them side by side,
explains single verdicts, replays held-out traffic as a live stream, scores CSV files and exports a PDF record.

## Quick start (Windows, PowerShell)

Requires Python 3.12 or newer (tested on 3.13).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Calling the venv's `python.exe` directly means you never need to activate the environment, which also avoids
PowerShell's script-execution policy.

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Data

See [data/README.md](data/README.md) for where to get CIC-IDS2017 and how to point Graticule at it
(the **Bench** page or the `NIDS_DATA_DIR` environment variable). The dataset is never copied into this project.

## Dataset citation

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and
Intrusion Traffic Characterization", 4th International Conference on Information Systems Security and Privacy
(ICISSP), 2018.

## Licence

MIT, © 2026 mohsin-moallim. See [LICENSE](LICENSE). Bundled fonts are under the SIL Open Font License
(see `static/fonts/OFL-*.txt`).
