"""Pulls a dataset: downloads and verifies its files, extracts its archive."""

from collections.abc import Sequence
import logging
import pathlib

from docrag.file_ops import archive
from docrag.file_ops import download
from docrag.file_ops import paths

_logger = logging.getLogger(__name__)


def pull_dataset(
    data_dir: pathlib.Path,
    sources: Sequence[download.Source],
    archive_source: download.Source,
    extract_dir_name: str,
    downloader: download.Downloader = download.download_data,
    *,
    overwrite: bool = False,
) -> None:
    """Downloads a dataset's files and extracts its archive.

    Each source is made present in data_dir and verified, then
    archive_source is extracted into the extract_dir_name directory under
    data_dir. The arguments and data_dir are checked before anything is
    downloaded.

    Args:
        data_dir: The directory that holds the dataset. Created if missing.
        sources: The files to download. They include archive_source.
        archive_source: The ZIP archive to extract.
        extract_dir_name: The name of the directory the archive is
            extracted into, under data_dir.
        downloader: The function that downloads a URL to a path.
        overwrite: Whether files and an extraction from another revision
            may be replaced.

    Raises:
        ValueError: archive_source is not in sources, or extract_dir_name
            is not a plain file name or is the file name of a source.
        download.UnsafeDestinationError: A file's path resolves outside
            data_dir.
        FileExistsError: data_dir or a path under it is not of the
            expected type, or holds another revision and overwrite is
            false.
        download.ChecksumMismatchError: A downloaded file does not match
            its checksum.
        requests.RequestException: The default downloader could not
            download a file.
        archive.UnsafeArchiveError: A member of the archive is a symbolic
            link or resolves to a path outside the extraction directory.
        zipfile.BadZipFile: The archive is not a ZIP archive.
        OSError: data_dir or a path under it is a symbolic link, or a file
            operation fails.
    """
    if archive_source not in sources:
        raise ValueError(
            f"{archive_source.filename!r}: archive source is not in sources"
        )
    paths.reject_non_plain_name(extract_dir_name)
    if any(source.filename == extract_dir_name for source in sources):
        raise ValueError(f"{extract_dir_name!r}: also names a source file")
    paths.reject_symlink(data_dir)
    paths.reject_non_directory(data_dir)

    for source in sources:
        if download.ensure_file(
            source, data_dir, downloader, overwrite=overwrite
        ):
            _logger.info("Downloaded %s", source.filename)
        else:
            _logger.info("Already present, skipped %s", source.filename)

    extract_dir = data_dir / extract_dir_name
    if archive.extract_archive(
        data_dir / archive_source.filename,
        extract_dir,
        archive_source.checksum,
        overwrite=overwrite,
    ):
        _logger.info("Extracted %s", extract_dir)
    else:
        _logger.info("Already extracted, skipped %s", extract_dir)
