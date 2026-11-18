"""Tests for docrag.file_ops.checksum."""

import errno
import hashlib
import io
import os
import pathlib

import pytest

from docrag.file_ops import checksum

_SHA256_OF_ABC = (
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
)
_SHA256_OF_EMPTY = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
# From `git hash-object`.
_GIT_BLOB_SHA1_OF_ABC = "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"
_GIT_BLOB_SHA1_OF_EMPTY = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


class _FailingReader:
    """Wraps an open file and fails with an I/O error on the second read."""

    def __init__(self, file: io.BufferedReader) -> None:
        """Initializes the reader.

        Args:
            file: The open file to read from.
        """
        self._file = file
        self._reads = 0

    def __enter__(self) -> "_FailingReader":
        """Returns the reader itself."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Closes the wrapped file."""
        self._file.close()

    def fileno(self) -> int:
        """Returns the descriptor of the wrapped file."""
        return self._file.fileno()

    def readable(self) -> bool:
        """Reports that the reader can be read."""
        return True

    def readinto(self, buffer: bytearray) -> int:
        """Reads into a buffer the first time, and fails after that.

        Args:
            buffer: The buffer to fill.

        Returns:
            The number of bytes read.

        Raises:
            OSError: Any read after the first.
        """
        if self._reads:
            raise OSError(errno.EIO, "Input/output error")
        self._reads += 1
        return self._file.readinto(buffer)


def _make_reads_fail_midway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes `open` in the checksum module return a failing reader.

    Args:
        monkeypatch: The fixture that undoes the patch after the test.
    """

    def failing_open(path: pathlib.Path, mode: str) -> _FailingReader:
        del mode  # The checksum module always opens files in binary mode.
        return _FailingReader(open(path, "rb"))

    monkeypatch.setattr(checksum, "open", failing_open, raising=False)


@pytest.mark.parametrize(
    "hexdigest",
    [
        pytest.param(_SHA256_OF_ABC.upper(), id="uppercase"),
        pytest.param(f" {_SHA256_OF_ABC}\n", id="surrounding-whitespace"),
        pytest.param(
            f" {_SHA256_OF_ABC.upper()} ", id="uppercase-and-whitespace"
        ),
    ],
)
def test_checksum_unnormalised_digest(hexdigest: str) -> None:
    value = checksum.Checksum(
        algorithm=checksum.Algorithm.SHA256, hexdigest=hexdigest
    )

    assert value.hexdigest == _SHA256_OF_ABC


def test_sha256_checksum_abc(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")

    assert checksum.sha256_checksum(path) == _SHA256_OF_ABC


def test_sha256_checksum_empty_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert checksum.sha256_checksum(path) == _SHA256_OF_EMPTY


def test_sha256_checksum_file_larger_than_buffer(
    tmp_path: pathlib.Path,
) -> None:
    # 1 MiB and 256 bytes.
    data = bytes(range(256)) * 4097
    path = tmp_path / "large.bin"
    path.write_bytes(data)

    assert checksum.sha256_checksum(path) == hashlib.sha256(data).hexdigest()


def test_sha256_checksum_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        checksum.sha256_checksum(tmp_path / "missing.bin")


def test_sha256_checksum_directory(tmp_path: pathlib.Path) -> None:
    with pytest.raises(IsADirectoryError):
        checksum.sha256_checksum(tmp_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_sha256_checksum_unreadable_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "unreadable.bin"
    path.write_bytes(b"abc")
    path.chmod(0)

    with pytest.raises(PermissionError):
        checksum.sha256_checksum(path)


def test_sha256_checksum_read_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    _make_reads_fail_midway(monkeypatch)

    with pytest.raises(OSError) as excinfo:
        checksum.sha256_checksum(path)

    assert excinfo.value.errno == errno.EIO


def test_git_blob_sha1_abc(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")

    assert checksum.git_blob_sha1(path) == _GIT_BLOB_SHA1_OF_ABC


def test_git_blob_sha1_empty_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert checksum.git_blob_sha1(path) == _GIT_BLOB_SHA1_OF_EMPTY


def test_git_blob_sha1_file_larger_than_buffer(
    tmp_path: pathlib.Path,
) -> None:
    # 1 MiB and 256 bytes.
    data = bytes(range(256)) * 4097
    path = tmp_path / "large.bin"
    path.write_bytes(data)
    expected = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

    assert checksum.git_blob_sha1(path) == expected


def test_git_blob_sha1_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        checksum.git_blob_sha1(tmp_path / "missing.bin")


def test_git_blob_sha1_directory(tmp_path: pathlib.Path) -> None:
    with pytest.raises(IsADirectoryError):
        checksum.git_blob_sha1(tmp_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_git_blob_sha1_unreadable_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "unreadable.bin"
    path.write_bytes(b"abc")
    path.chmod(0)

    with pytest.raises(PermissionError):
        checksum.git_blob_sha1(path)


def test_git_blob_sha1_read_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    _make_reads_fail_midway(monkeypatch)

    with pytest.raises(OSError) as excinfo:
        checksum.git_blob_sha1(path)

    assert excinfo.value.errno == errno.EIO


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        (checksum.Algorithm.SHA256, _SHA256_OF_ABC),
        (checksum.Algorithm.GIT_BLOB_SHA1, _GIT_BLOB_SHA1_OF_ABC),
    ],
)
def test_compute_abc(
    tmp_path: pathlib.Path, algorithm: checksum.Algorithm, expected: str
) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")

    assert checksum.compute(path, algorithm) == expected


@pytest.mark.parametrize("algorithm", list(checksum.Algorithm))
def test_compute_missing_file(
    tmp_path: pathlib.Path, algorithm: checksum.Algorithm
) -> None:
    with pytest.raises(FileNotFoundError):
        checksum.compute(tmp_path / "missing.bin", algorithm)


@pytest.mark.parametrize("algorithm", list(checksum.Algorithm))
def test_compute_directory(
    tmp_path: pathlib.Path, algorithm: checksum.Algorithm
) -> None:
    with pytest.raises(IsADirectoryError):
        checksum.compute(tmp_path, algorithm)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
@pytest.mark.parametrize("algorithm", list(checksum.Algorithm))
def test_compute_unreadable_file(
    tmp_path: pathlib.Path, algorithm: checksum.Algorithm
) -> None:
    path = tmp_path / "unreadable.bin"
    path.write_bytes(b"abc")
    path.chmod(0)

    with pytest.raises(PermissionError):
        checksum.compute(path, algorithm)


@pytest.mark.parametrize("algorithm", list(checksum.Algorithm))
def test_compute_read_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    algorithm: checksum.Algorithm,
) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    _make_reads_fail_midway(monkeypatch)

    with pytest.raises(OSError) as excinfo:
        checksum.compute(path, algorithm)

    assert excinfo.value.errno == errno.EIO
