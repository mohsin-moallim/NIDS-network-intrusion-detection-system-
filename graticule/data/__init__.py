"""Data handling for Graticule: reading flow CSVs, cleaning them, sampling and assembling a prepared dataset.

Modules:

* ``reader`` turns one CSV file (or an uploaded file) into a float32 feature frame plus clean labels.
* ``clean`` handles infinite and missing values, duplicate rows, conflicting labels and degenerate columns.
* ``sampling`` builds targets for a detection mode, draws a rare-aware sample and splits it into train and test.
* ``prepare`` runs the whole 01 Sample procedure and returns a :class:`~graticule.data.prepare.PreparedDataset`.
* ``synthetic`` generates flows that look like the real files when no data folder is available.
"""
