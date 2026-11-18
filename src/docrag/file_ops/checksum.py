"""Checksums of files: the supported algorithms and how to compute them."""

import dataclasses
import enum
import hashlib
import os
import pathlib
import typing


class Algorithm(enum.Enum):
    """A hash algorithm a source publishes for its files.

    Attributes:
        SHA256: SHA-256 over the bytes of the file.
        GIT_BLOB_SHA1: SHA-1 over a git blob header and the bytes of the
            file, the object ID that `git hash-object` reports.
    """

    SHA256 = "sha256"
    GIT_BLOB_SHA1 = "git-blob-sha1"


@dataclasses.dataclass(frozen=True)
class Checksum:
    """The expected checksum of a file, as its source publishes it.

    The digest is stored without surrounding whitespace and in lowercase,
    so a digest copied in another case still matches.

    Attributes:
        algorithm: The algorithm the checksum was computed with.
        hexdigest: The digest as lowercase hex.
    """

    algorithm: Algorithm
    hexdigest: str

    def __post_init__(self) -> None:
        """Strips whitespace from the digest and converts it to lowercase."""
        object.__setattr__(self, "hexdigest", self.hexdigest.strip().lower())


def sha256_checksum(path: pathlib.Path) -> str:
    """Returns the SHA-256 of a file as lowercase hex.

    The file is read in chunks, so it never has to fit in memory.

    Args:
        path: The file to hash.

    Returns:
        The digest as a string of 64 lowercase hexadecimal characters.

    Raises:
        FileNotFoundError: No file exists at path.
        IsADirectoryError: path is a directory.
        PermissionError: The file cannot be opened for reading.
        OSError: The file could not be read to the end.
    """
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def git_blob_sha1(path: pathlib.Path) -> str:
    """Returns the git blob SHA-1 of a file as lowercase hex.

    The hash covers a header, `blob`, a space, the size of the file in
    bytes and a NUL byte, followed by the bytes of the file. The result is
    the object ID git gives the file. The file is read in chunks, so it
    never has to fit in memory.

    Args:
        path: The file to hash.

    Returns:
        The digest as a string of 40 lowercase hexadecimal characters.

    Raises:
        FileNotFoundError: No file exists at path.
        IsADirectoryError: path is a directory.
        PermissionError: The file cannot be opened for reading.
        OSError: The file could not be read to the end.
    """
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        digest = hashlib.sha1(b"blob %d\0" % size, usedforsecurity=False)
        return hashlib.file_digest(f, lambda: digest).hexdigest()


def compute(path: pathlib.Path, algorithm: Algorithm) -> str:
    """Returns the checksum of a file under the given algorithm.

    Args:
        path: The file to hash.
        algorithm: The algorithm to compute.

    Returns:
        The digest as lowercase hex.

    Raises:
        FileNotFoundError: No file exists at path.
        IsADirectoryError: path is a directory.
        PermissionError: The file cannot be opened for reading.
        OSError: The file could not be read to the end.
    """
    match algorithm:
        case Algorithm.SHA256:
            return sha256_checksum(path)
        case Algorithm.GIT_BLOB_SHA1:
            return git_blob_sha1(path)
        case _:  # pragma: no cover
            typing.assert_never(algorithm)
