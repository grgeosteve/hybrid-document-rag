r"""Parses a dataset's PDFs into a parquet file under the data directory.

Usage:
    uv run python scripts/parse_dataset.py \
        ohr_bench --backend pypdfium2 [--overwrite]
"""

import logging
import pathlib
import sys

from docrag.datasets import parse_cli

_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(parse_cli.main(data_root=_DATA_ROOT))
