"""Models for NIDS: the five channels, how they are trained, and how their readings are combined.

Modules:

* ``transforms`` holds the small transformers every channel pipeline starts with (input sanitising, signed log).
* ``zoo`` describes the channels (random forest, XGBoost, RBF SVM, MLP, logistic regression), builds an unfitted
  pipeline for each and computes the capped, balanced sample weights they all train with.
* ``train`` turns a prepared sample into training and test matrices and fits the requested channels.
* ``jobs`` runs a fit in a background thread with progress, elapsed time and cancellation.
* ``verdict`` combines the channels' class probabilities into one consensus reading.
"""
