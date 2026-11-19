"""Downloads files and verifies them against pinned checksums."""

from collections.abc import Callable
import dataclasses
import logging
import pathlib
import secrets

import requests
import tqdm

from docrag.file_ops import checksum
from docrag.file_ops import paths

_logger = logging.getLogger(__name__)

_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB
_PARTIAL_INFIX = ".download-"
_PARTIAL_NAME_ATTEMPTS = 100
_TIMEOUT_SECONDS = 30

Downloader = Callable[[str, pathlib.Path], None]


class ChecksumMismatchError(Exception):
    """The checksum computed from a downloaded file does not match its source.

    Attributes:
        path: The final path of the file that was checked.
        expected: The checksum the source publishes.
        actual: The digest computed from the file with the expected
            algorithm, as lowercase hex.
    """

    def __init__(
        self, path: pathlib.Path, expected: checksum.Checksum, actual: str
    ) -> None:
        """Initializes the error with the file and both checksums.

        Args:
            path: The final path of the file that was checked.
            expected: The checksum the source publishes.
            actual: The digest computed from the file with the expected
                algorithm, as lowercase hex.
        """
        self.path = path
        self.expected = expected
        self.actual = actual

        super().__init__(
            f"{self.path}: {expected.algorithm.value} checksum mismatch.\n"
            f"Expected: {self.expected.hexdigest}\n"
            f"Actual:   {self.actual}"
        )


class UnsafeDestinationError(Exception):
    """A file's destination resolves to a path outside its data directory.

    Attributes:
        destination_path: The path the file would be written to.
        data_dir: The directory the file must stay within.
    """

    def __init__(
        self, destination_path: pathlib.Path, data_dir: pathlib.Path
    ) -> None:
        """Initializes the error with the destination and the directory.

        Args:
            destination_path: The path the file would be written to.
            data_dir: The directory the file must stay within.
        """
        self.destination_path = destination_path
        self.data_dir = data_dir

        super().__init__(f"{destination_path}: resolves outside {data_dir}")


@dataclasses.dataclass(frozen=True)
class Source:
    """A remote file pinned to a revision and a checksum.

    Attributes:
        url: Address the file is downloaded from, including the pinned
            revision.
        filename: Name the file is given inside the data directory.
        checksum: The checksum the source publishes for the file.
    """

    url: str
    filename: str
    checksum: checksum.Checksum

    def __post_init__(self) -> None:
        """Checks that filename is a plain file name.

        Raises:
            ValueError: filename is not a plain file name, for example "",
                ".." or "a/b".
        """
        if (
            self.filename in ("", ".", "..")
            or pathlib.PurePath(self.filename).name != self.filename
        ):
            raise ValueError(f"{self.filename!r}: not a plain file name")


def parse_content_length(value: str | None) -> int | None:
    """Returns the body size declared by a Content-Length value.

    The value must match the grammar in RFC 9110, section 8.6: one or more
    ASCII digits. Any other value, including a list of repeated values, is
    treated as absent.

    Args:
        value: The value of the Content-Length header, or None if the
            response has no such header.

    Returns:
        The declared size in bytes, or None if the value is missing or
        does not match the grammar.
    """
    if value is None or not (value.isascii() and value.isdigit()):
        return None
    return int(value)


def download_data(
    url: str,
    destination_path: pathlib.Path,
    http_get: Callable[..., requests.Response] = requests.get,
) -> None:
    """Streams the body of an HTTP response to a file.

    The response is written in chunks, so the file never has to fit in
    memory. A file already at the destination is overwritten.

    Args:
        url: The URL to download.
        destination_path: The file to write. Its directory must exist.
        http_get: The function that sends the GET request. It is called
            with the URL, stream=True, a timeout and request headers,
            and returns the response.

    Raises:
        requests.HTTPError: The server answered with an error status.
        requests.RequestException: The request failed before or during the
            transfer, for example on a connection error or a timeout.
        OSError: The file cannot be opened or written.
    """
    _logger.info("Downloading %s...", url)
    with http_get(
        url,
        stream=True,
        timeout=_TIMEOUT_SECONDS,
        headers={"Accept-Encoding": "identity"},
    ) as response:
        response.raise_for_status()

        # Extract total file size from headers for progress tracking
        total_size = parse_content_length(
            response.headers.get("content-length")
        )

        with tqdm.tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=destination_path.name,
        ) as pbar:
            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=_CHUNK_SIZE_BYTES
                ):
                    if chunk:
                        pbar.update(len(chunk))
                        f.write(chunk)


def _remove_partial_files(destination_path: pathlib.Path) -> None:
    """Removes the partial files that earlier calls left for a destination.

    Args:
        destination_path: The final path of the file. Its directory does not
            have to exist.

    Raises:
        FileExistsError: A path named as a partial file is not a regular
            file.
        OSError: A path named as a partial file is a symbolic link, or a
            partial file cannot be removed.
    """
    if not destination_path.parent.is_dir():
        return
    prefix = destination_path.name + _PARTIAL_INFIX
    for path in sorted(destination_path.parent.iterdir()):
        if path.name.startswith(prefix):
            paths.reject_symlink(path)
            paths.reject_non_file(path)
            path.unlink()


def _create_partial_file(destination_path: pathlib.Path) -> pathlib.Path:
    """Creates a new, empty partial file next to a destination.

    The file gets a random name and is created only if no path has that
    name. A name that is taken is replaced by a new one.

    Args:
        destination_path: The final path of the file. Its directory must
            exist.

    Returns:
        The partial file.

    Raises:
        FileExistsError: Every name tried was taken.
        OSError: The file cannot be created.
    """
    for _ in range(_PARTIAL_NAME_ATTEMPTS):
        partial_path = destination_path.with_name(
            destination_path.name + _PARTIAL_INFIX + secrets.token_hex(8)
        )
        try:
            partial_path.touch(exist_ok=False)
        except FileExistsError:
            continue
        return partial_path
    raise FileExistsError(
        f"{destination_path}: no free partial file name after "
        f"{_PARTIAL_NAME_ATTEMPTS} attempts"
    )


def ensure_file(
    source: Source,
    data_dir: pathlib.Path,
    download: Downloader = download_data,
    *,
    overwrite: bool = False,
) -> bool:
    """Makes sure a source file is present in the data directory.

    A file already present is kept if it matches source.checksum, and is
    replaced only if overwrite is true. The file is downloaded into a
    partial file in data_dir, verified, and renamed into place, so an
    existing file is replaced only by a verified one. A failed call can
    leave a partial file behind, and the next call removes it.

    Args:
        source: The file to make available.
        data_dir: The directory that holds the dataset. Created if missing.
        download: The function that downloads a URL to a path.
        overwrite: Whether a file that does not match source.checksum may
            be replaced.

    Returns:
        True if the file was downloaded, False if a file matching
        source.checksum was already present.

    Raises:
        UnsafeDestinationError: The file's path resolves outside data_dir.
        FileExistsError: The file's path or a path named as a partial file
            exists and is not a regular file, the file does not match
            source.checksum and overwrite is false, or no free partial file
            name was found.
        ChecksumMismatchError: The downloaded file does not match
            source.checksum.
        requests.RequestException: The default downloader could not
            download the file.
        OSError: The file's path or a path named as a partial file is a
            symbolic link, or a file operation fails.
    """
    destination_path = data_dir / source.filename
    if not paths.is_safe_path(data_dir, destination_path):
        raise UnsafeDestinationError(destination_path, data_dir)
    paths.reject_symlink(destination_path)
    paths.reject_non_file(destination_path)
    _remove_partial_files(destination_path)
    if destination_path.is_file():
        existing_checksum = checksum.compute(
            destination_path, source.checksum.algorithm
        )
        if existing_checksum == source.checksum.hexdigest:
            return False
        if not overwrite:
            raise FileExistsError(
                f"{destination_path}: does not match its checksum and "
                "overwrite is false"
            )

    data_dir.mkdir(parents=True, exist_ok=True)
    partial_path = _create_partial_file(destination_path)
    try:
        download(source.url, partial_path)
        computed_checksum = checksum.compute(
            partial_path, source.checksum.algorithm
        )
        if computed_checksum != source.checksum.hexdigest:
            raise ChecksumMismatchError(
                path=destination_path,
                expected=source.checksum,
                actual=computed_checksum,
            )
        partial_path.rename(destination_path)
    except (requests.RequestException, OSError, ChecksumMismatchError):
        partial_path.unlink()
        raise
    return True
