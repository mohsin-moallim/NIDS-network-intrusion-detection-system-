"""NIDS (Network Intrusion Detection System): a measuring bench for training, testing and comparing
network-intrusion classifiers.

The ``nids`` package holds every piece of data handling and machine-learning logic. It never imports
the web framework, so each part can be exercised from plain Python or from the test suite.
"""

__version__ = "0.1.0"
#: Short display name, and the long form it stands for.
APP_NAME = "NIDS"
APP_FULL_NAME = "Network Intrusion Detection System"
#: The name as titles and first mentions give it.
APP_TITLE = f"{APP_NAME} — {APP_FULL_NAME}"
TAGLINE = "A measuring bench for training, testing and comparing network-intrusion classifiers."
