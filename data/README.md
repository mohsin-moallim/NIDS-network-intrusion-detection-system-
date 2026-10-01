# Data

Graticule does not ship any network-flow data, and nothing in this folder is committed except this note.
Git ignores every other file here and every `*.csv` file anywhere in the project.

## Getting CIC-IDS2017

1. Request and download the dataset from the Canadian Institute for Cybersecurity:
   <https://www.unb.ca/cic/datasets/ids-2017.html>
2. From the download, take the **MachineLearningCSV** set: eight CSV files, one per capture session
   (Monday to Friday), about 0.85 GB together. Each file has 79 columns, 78 numeric flow features plus `Label`
   (one feature, `Fwd Header Length`, appears twice, so Graticule works with 77 distinct features).
3. Put the eight files in any folder you like, preferably outside this project. Graticule expects these names:

   | File | Session | Contents |
   |---|---|---|
   | `Monday-WorkingHours.pcap_ISCX.csv` | Monday | normal traffic only |
   | `Tuesday-WorkingHours.pcap_ISCX.csv` | Tuesday | FTP and SSH password guessing |
   | `Wednesday-workingHours.pcap_ISCX.csv` | Wednesday | DoS variants and Heartbleed |
   | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Thursday morning | web attacks |
   | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Thursday afternoon | infiltration |
   | `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Friday morning | botnet |
   | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday afternoon | port scan |
   | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Friday afternoon | DDoS |

   Other CSV files in the same folder are listed too and can be sampled, provided they use the same columns.

## Pointing Graticule at the files

Use either of these (the in-app setting wins when both are present):

- **In the app:** open **Bench** and set *Data folder*. The value is saved to `local_settings.json`
  in the project root, which git ignores.
- **Environment variable:** set `NIDS_DATA_DIR` before starting the app, for example in PowerShell:

  ```powershell
  $env:NIDS_DATA_DIR = "D:\datasets\CIC-IDS2017\MachineLearningCSV"
  .\.venv\Scripts\python.exe -m streamlit run app.py
  ```

  `$env:` lasts for that PowerShell window only. The Bench shows which folder is in use, where the choice came
  from, and which of the eight files were found or are missing.

Graticule only reads `*.csv` files that sit directly in that folder (sub-folders are not searched). It never copies,
moves or rewrites them. With no folder configured, the app runs on its built-in synthetic flow generator, which
produces the same 77 feature columns.

## What the reader repairs

The published files have a few known quirks. Graticule handles each of them while reading and reports what it did on
the 01 Sample sheet:

- column names with leading spaces (`" Destination Port"`) are trimmed;
- the repeated `Fwd Header Length` column is dropped after checking that both copies hold the same values;
- `Infinity` and empty cells (in `Flow Bytes/s` and `Flow Packets/s`) are dropped, imputed or recomputed, as chosen;
- the Thursday Web Attack labels, whose dash arrived as a replacement character (`Web Attack � Brute Force`), become
  `Web Attack - Brute Force`;
- exact duplicate rows are removed within each file and then across files, with every count shown;
- negative values such as the `-1` "not seen" markers in `Init_Win_bytes_forward` are kept and handled by a signed
  logarithm in the channels that need scaling.

Nothing derived from the files is written back to disk: saved channel sets hold models, summary statistics and
synthetic probe flows, never dataset rows.

## Tests

The real-data tests use the same folder (Bench setting or `NIDS_DATA_DIR`), or one given on the command line, and
skip when none is available:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m realdata "--data-dir=D:\datasets\CIC-IDS2017\MachineLearningCSV"
```

Keep `--data-dir=` and the folder in one token as shown. Given as two separate arguments, pytest takes the folder
for a test path, looks for its configuration in the wrong place and reports `--data-dir` as unrecognised.

## Citation

If you use CIC-IDS2017, cite the dataset authors:

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion
> Detection Dataset and Intrusion Traffic Characterization", 4th International Conference on
> Information Systems Security and Privacy (ICISSP), 2018.
