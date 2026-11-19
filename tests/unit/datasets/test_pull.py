"""Tests for docrag.datasets.pull."""

from collections.abc import Mapping
import hashlib
import io
import logging
import pathlib
import zipfile

import pytest
import requests

from docrag.datasets import dataset
from docrag.datasets import pull
from docrag.file_ops import archive
from docrag.file_ops import checksum
from docrag.file_ops import download


def _zip_bytes(members: Mapping[str, bytes]) -> bytes:
    """Returns the bytes of a ZIP archive.

    Args:
        members: Member names mapped to their contents. Names are stored
            as given, so a name may contain `..`.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for name, content in members.items():
            zip_file.writestr(name, content)
    return buffer.getvalue()


def _source(filename: str, content: bytes) -> download.Source:
    """Returns a source whose SHA-256 checksum matches the content.

    Args:
        filename: The name the file is given inside the data directory.
        content: The bytes the source serves.
    """
    return download.Source(
        url=f"https://example.com/{filename}",
        filename=filename,
        checksum=checksum.Checksum(
            algorithm=checksum.Algorithm.SHA256,
            hexdigest=hashlib.sha256(content).hexdigest(),
        ),
    )


class _FakeDownloader:
    """Writes the bytes served for each URL and records the URLs.

    Attributes:
        urls: The URL of each call, in order.
    """

    def __init__(self, contents: Mapping[str, bytes]) -> None:
        """Initializes the downloader.

        Args:
            contents: Each URL mapped to the bytes written for it.
        """
        self._contents = contents
        self.urls: list[str] = []

    def __call__(self, url: str, destination_path: pathlib.Path) -> None:
        """Writes the bytes served for the URL to a file.

        Args:
            url: The URL that would be downloaded.
            destination_path: The file to write.
        """
        self.urls.append(url)
        destination_path.write_bytes(self._contents[url])


def _failing_download(url: str, destination_path: pathlib.Path) -> None:
    """Fails as a download that cannot connect does.

    Args:
        url: The URL that would be downloaded.
        destination_path: The file that would be written.

    Raises:
        requests.ConnectionError: Always.
    """
    del destination_path  # Unused.
    raise requests.ConnectionError(url)


_NAME = "sample"
_EXTRACT_DIR = "pdfs"
_TEXT = b"contents of the text file"
_ARCHIVE = _zip_bytes({"a.pdf": b"alpha", "nested/b.pdf": b"beta"})
_TEXT_SOURCE = _source("text.json", _TEXT)
_ARCHIVE_SOURCE = _source("pdfs.zip", _ARCHIVE)
_CONTENTS = {_TEXT_SOURCE.url: _TEXT, _ARCHIVE_SOURCE.url: _ARCHIVE}


def _dataset(
    sources: tuple[download.Source, ...] = (_TEXT_SOURCE, _ARCHIVE_SOURCE),
    archive_source: download.Source = _ARCHIVE_SOURCE,
) -> dataset.Dataset:
    """Returns the test dataset.

    Args:
        sources: The files of the dataset.
        archive_source: The archive among sources that the dataset
            extracts.
    """
    return dataset.Dataset(
        name=_NAME,
        sources=sources,
        archive_source=archive_source,
        extract_dir_name=_EXTRACT_DIR,
    )


def test_pull_dataset_new_data_root(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=pull.__name__)
    data_root = tmp_path / "data"
    data_dir = data_root / _NAME
    extract_dir = data_dir / _EXTRACT_DIR
    downloader = _FakeDownloader(_CONTENTS)

    pull.pull_dataset(data_root, _dataset(), downloader)

    assert downloader.urls == [_TEXT_SOURCE.url, _ARCHIVE_SOURCE.url]
    assert list(data_root.iterdir()) == [data_dir]
    assert (data_dir / "text.json").read_bytes() == _TEXT
    assert (data_dir / "pdfs.zip").read_bytes() == _ARCHIVE
    assert (extract_dir / "a.pdf").read_bytes() == b"alpha"
    assert (extract_dir / "nested" / "b.pdf").read_bytes() == b"beta"
    assert set(data_dir.iterdir()) == {
        data_dir / "text.json",
        data_dir / "pdfs.zip",
        extract_dir,
    }
    assert caplog.messages == [
        "Downloaded text.json",
        "Downloaded pdfs.zip",
        f"Extracted {extract_dir}",
    ]


def test_pull_dataset_second_run(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    data_root = tmp_path / "data"
    extract_dir = data_root / _NAME / _EXTRACT_DIR
    pull.pull_dataset(data_root, _dataset(), _FakeDownloader(_CONTENTS))
    added = extract_dir / "added.pdf"
    added.write_bytes(b"added")
    caplog.set_level(logging.INFO, logger=pull.__name__)
    downloader = _FakeDownloader(_CONTENTS)

    pull.pull_dataset(data_root, _dataset(), downloader)

    assert not downloader.urls
    assert added.read_bytes() == b"added"
    assert caplog.messages == [
        "Already present, skipped text.json",
        "Already present, skipped pdfs.zip",
        f"Already extracted, skipped {extract_dir}",
    ]


def test_pull_dataset_changed_archive_no_overwrite(
    tmp_path: pathlib.Path,
) -> None:
    data_root = tmp_path / "data"
    archive_path = data_root / _NAME / "pdfs.zip"
    extract_dir = data_root / _NAME / _EXTRACT_DIR
    pull.pull_dataset(data_root, _dataset(), _FakeDownloader(_CONTENTS))
    new_archive = _zip_bytes({"c.pdf": b"gamma"})
    new_archive_source = _source("pdfs.zip", new_archive)
    downloader = _FakeDownloader(
        {_TEXT_SOURCE.url: _TEXT, new_archive_source.url: new_archive}
    )

    with pytest.raises(FileExistsError) as excinfo:
        pull.pull_dataset(
            data_root,
            _dataset((_TEXT_SOURCE, new_archive_source), new_archive_source),
            downloader,
        )

    assert str(excinfo.value) == (
        f"{archive_path}: does not match its checksum and overwrite is false"
    )
    assert not downloader.urls
    assert archive_path.read_bytes() == _ARCHIVE
    assert (extract_dir / "a.pdf").read_bytes() == b"alpha"
    assert (extract_dir / "nested" / "b.pdf").read_bytes() == b"beta"
    assert not (extract_dir / "c.pdf").exists()


def test_pull_dataset_changed_archive_overwrite(
    tmp_path: pathlib.Path,
) -> None:
    data_root = tmp_path / "data"
    data_dir = data_root / _NAME
    extract_dir = data_dir / _EXTRACT_DIR
    pull.pull_dataset(data_root, _dataset(), _FakeDownloader(_CONTENTS))
    new_archive = _zip_bytes({"c.pdf": b"gamma"})
    new_archive_source = _source("pdfs.zip", new_archive)
    downloader = _FakeDownloader(
        {_TEXT_SOURCE.url: _TEXT, new_archive_source.url: new_archive}
    )

    pull.pull_dataset(
        data_root,
        _dataset((_TEXT_SOURCE, new_archive_source), new_archive_source),
        downloader,
        overwrite=True,
    )

    assert downloader.urls == [new_archive_source.url]
    assert (data_dir / "pdfs.zip").read_bytes() == new_archive
    assert (extract_dir / "c.pdf").read_bytes() == b"gamma"
    assert not (extract_dir / "a.pdf").exists()
    assert not (extract_dir / "nested").exists()


def test_pull_dataset_data_dir_is_symlink(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    data_dir = data_root / _NAME
    data_dir.symlink_to(real_dir, target_is_directory=True)
    downloader = _FakeDownloader(_CONTENTS)

    with pytest.raises(OSError) as excinfo:
        pull.pull_dataset(data_root, _dataset(), downloader)

    assert str(excinfo.value) == f"{data_dir}: symbolic links are not allowed"
    assert not downloader.urls
    assert data_dir.is_symlink()
    assert not list(real_dir.iterdir())


def test_pull_dataset_data_dir_is_file(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    data_dir = data_root / _NAME
    data_dir.write_bytes(b"not a directory")
    downloader = _FakeDownloader(_CONTENTS)

    with pytest.raises(FileExistsError) as excinfo:
        pull.pull_dataset(data_root, _dataset(), downloader)

    assert str(excinfo.value) == f"{data_dir}: exists and is not a directory"
    assert not downloader.urls
    assert data_dir.read_bytes() == b"not a directory"


def test_pull_dataset_checksum_mismatch(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    data_dir = data_root / _NAME
    downloader = _FakeDownloader(
        {_TEXT_SOURCE.url: _TEXT, _ARCHIVE_SOURCE.url: b"corrupted"}
    )

    with pytest.raises(download.ChecksumMismatchError) as excinfo:
        pull.pull_dataset(data_root, _dataset(), downloader)

    assert excinfo.value.path == data_dir / "pdfs.zip"
    assert not (data_dir / "pdfs.zip").exists()
    assert not (data_dir / _EXTRACT_DIR).exists()


def test_pull_dataset_download_error(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    data_dir = data_root / _NAME

    with pytest.raises(requests.ConnectionError):
        pull.pull_dataset(data_root, _dataset(), _failing_download)

    assert not (data_dir / "text.json").exists()
    assert not (data_dir / _EXTRACT_DIR).exists()


def test_pull_dataset_unsafe_archive(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    data_dir = data_root / _NAME
    unsafe_archive = _zip_bytes({"../evil.pdf": b"evil"})
    unsafe_source = _source("pdfs.zip", unsafe_archive)
    downloader = _FakeDownloader({unsafe_source.url: unsafe_archive})

    with pytest.raises(archive.UnsafeArchiveError):
        pull.pull_dataset(
            data_root, _dataset((unsafe_source,), unsafe_source), downloader
        )

    assert not (data_dir / _EXTRACT_DIR).exists()
    assert not (data_dir / "evil.pdf").exists()


def test_pull_dataset_not_a_zip(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    data_dir = data_root / _NAME
    not_a_zip = b"not a ZIP archive"
    not_a_zip_source = _source("pdfs.zip", not_a_zip)
    downloader = _FakeDownloader({not_a_zip_source.url: not_a_zip})

    with pytest.raises(zipfile.BadZipFile):
        pull.pull_dataset(
            data_root,
            _dataset((not_a_zip_source,), not_a_zip_source),
            downloader,
        )

    assert not (data_dir / _EXTRACT_DIR).exists()
