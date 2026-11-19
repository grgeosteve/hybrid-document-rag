"""Command line that pulls a named dataset into a data root."""

import argparse
from collections.abc import Mapping
from collections.abc import Sequence
import logging
import pathlib
import types
import zipfile

import requests

from docrag.datasets import dataset
from docrag.datasets import ohr_bench
from docrag.datasets import pull
from docrag.file_ops import archive
from docrag.file_ops import download

_DATASETS = types.MappingProxyType({ohr_bench.DATASET.name: ohr_bench.DATASET})

_logger = logging.getLogger(__name__)


def main(
    argv: Sequence[str] | None = None,
    *,
    data_root: pathlib.Path,
    datasets: Mapping[str, dataset.Dataset] = _DATASETS,
    downloader: download.Downloader = download.download_data,
) -> int:
    """Pulls the dataset named on the command line.

    An error that a pull is expected to meet, such as a failed download or
    a checksum mismatch, is logged as one line and gives exit status 1.

    Args:
        argv: The command-line arguments without the program name, or None
            to read them from sys.argv.
        data_root: The directory that holds every dataset.
        datasets: The datasets that can be pulled, by name.
        downloader: The function that downloads a URL to a path.

    Returns:
        The exit status: 0 if the dataset was pulled, 1 if it was not.

    Raises:
        SystemExit: The arguments are not valid, with exit status 2, or help
            was requested, with exit status 0.
    """
    parser = argparse.ArgumentParser(
        description="Downloads a dataset and extracts its archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset", choices=sorted(datasets), help="The dataset to pull."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files and an extraction from another revision.",
    )
    args = parser.parse_args(argv)
    try:
        pull.pull_dataset(
            data_root,
            datasets[args.dataset],
            downloader,
            overwrite=args.overwrite,
        )
    except (
        OSError,
        requests.RequestException,
        download.ChecksumMismatchError,
        download.UnsafeDestinationError,
        archive.UnsafeArchiveError,
        zipfile.BadZipFile,
    ) as error:
        _logger.error("%s", error)
        return 1
    return 0
