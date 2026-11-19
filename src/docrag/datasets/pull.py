"""Pulls a dataset: downloads and verifies its files, extracts its archive."""

import logging
import pathlib

from docrag.datasets import dataset
from docrag.file_ops import archive
from docrag.file_ops import download
from docrag.file_ops import paths

_logger = logging.getLogger(__name__)


def pull_dataset(
    data_root: pathlib.Path,
    dataset_to_pull: dataset.Dataset,
    downloader: download.Downloader = download.download_data,
    *,
    overwrite: bool = False,
) -> None:
    """Downloads a dataset's files and extracts its archive.

    The dataset lives in the directory named after it under data_root.
    Each source is made present there and verified, then the archive is
    extracted into the dataset's extraction directory. The dataset's
    directory is checked before anything is downloaded.

    Args:
        data_root: The directory that holds every dataset. Created if
            missing.
        dataset_to_pull: The dataset to pull.
        downloader: The function that downloads a URL to a path.
        overwrite: Whether files and an extraction from another revision
            may be replaced.

    Raises:
        download.UnsafeDestinationError: A file's path resolves outside the
            dataset's directory.
        FileExistsError: The dataset's directory or a path under it is not
            of the expected type, or holds another revision and overwrite
            is false.
        download.ChecksumMismatchError: A downloaded file does not match
            its checksum.
        requests.RequestException: The default downloader could not
            download a file.
        archive.UnsafeArchiveError: A member of the archive is a symbolic
            link or resolves to a path outside the extraction directory.
        zipfile.BadZipFile: The archive is not a ZIP archive.
        OSError: The dataset's directory or a path under it is a symbolic
            link, or a file operation fails.
    """
    data_dir = data_root / dataset_to_pull.name
    paths.reject_symlink(data_dir)
    paths.reject_non_directory(data_dir)

    for source in dataset_to_pull.sources:
        if download.ensure_file(
            source, data_dir, downloader, overwrite=overwrite
        ):
            _logger.info("Downloaded %s", source.filename)
        else:
            _logger.info("Already present, skipped %s", source.filename)

    extract_dir = data_dir / dataset_to_pull.extract_dir_name
    if archive.extract_archive(
        data_dir / dataset_to_pull.archive_source.filename,
        extract_dir,
        dataset_to_pull.archive_source.checksum,
        overwrite=overwrite,
    ):
        _logger.info("Extracted %s", extract_dir)
    else:
        _logger.info("Already extracted, skipped %s", extract_dir)
