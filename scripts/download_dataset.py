"""Downloads a dataset into the git-ignored data directory and extracts it.

Usage:
    uv run python scripts/download_dataset.py ohr_bench [--overwrite]
"""

import logging
import pathlib
import sys

from docrag.datasets import pull_cli

_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(pull_cli.main(data_root=_DATA_ROOT))
