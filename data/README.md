# Data

Graticule does not ship any network-flow data, and nothing in this folder is committed except this note.

## Getting CIC-IDS2017

1. Request and download the dataset from the Canadian Institute for Cybersecurity:
   <https://www.unb.ca/cic/datasets/ids-2017.html>
2. From the download, take the **MachineLearningCSV** set: eight CSV files, one per capture session
   (Monday to Friday). Each file has 79 columns, 78 numeric flow features plus `Label`.
3. Put the eight files in any folder you like. They do not need to be inside this project.

## Pointing Graticule at the files

Use either of these (the in-app setting wins when both are present):

- **In the app:** open **Bench** and set *Data folder*. The value is saved to `local_settings.json`
  in the project root, which git ignores.
- **Environment variable:** set `NIDS_DATA_DIR` before starting the app, for example in PowerShell:

  ```powershell
  $env:NIDS_DATA_DIR = "D:\datasets\CIC-IDS2017\MachineLearningCSV"
  ```

Graticule only reads `*.csv` files that sit directly in that folder. It never copies or rewrites them.
With no folder configured, the app runs on its built-in synthetic flow generator.

## Citation

If you use CIC-IDS2017, cite the dataset authors:

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion
> Detection Dataset and Intrusion Traffic Characterization", 4th International Conference on
> Information Systems Security and Privacy (ICISSP), 2018.
