"""Tests for docrag.datasets.pull_cli."""

from collections.abc import Mapping
import hashlib
import io
import logging
import pathlib
import zipfile

import pytest
import requests

from docrag.datasets import dataset
from docrag.datasets import pull_cli
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


def _archive_dataset(content: bytes) -> dataset.Dataset:
    """Returns a dataset whose only file is an archive with the content.

    Args:
        content: The bytes of the archive. Its checksum matches them.
    """
    source = download.Source(
        url="https://example.com/pdfs.zip",
        filename="pdfs.zip",
        checksum=checksum.Checksum(
            algorithm=checksum.Algorithm.SHA256,
            hexdigest=hashlib.sha256(content).hexdigest(),
        ),
    )
    return dataset.Dataset(
        name="sample",
        sources=(source,),
        archive_source=source,
        extract_dir_name="pdfs",
    )


class _FakeDownloader:
    """Writes fixed bytes to the destination and records each URL.

    Attributes:
        urls: The URL of each call, in order.
    """

    def __init__(self, content: bytes) -> None:
        """Initialises the downloader.

        Args:
            content: The bytes written on each call.
        """
        self._content = content
        self.urls: list[str] = []

    def __call__(self, url: str, destination_path: pathlib.Path) -> None:
        """Writes the content to a file.

        Args:
            url: The URL that would be downloaded.
            destination_path: The file to write.
        """
        self.urls.append(url)
        destination_path.write_bytes(self._content)


def _failing_download(url: str, destination_path: pathlib.Path) -> None:
    """Fails as a download that cannot connect does.

    Args:
        url: The URL that would be downloaded.
        destination_path: The file that would be written.

    Raises:
        requests.ConnectionError: Always.
    """
    del destination_path  # Unused.
    raise requests.ConnectionError(f"{url}: connection refused")


_ARCHIVE = _zip_bytes({"a.pdf": b"alpha"})
_DATASETS = {"sample": _archive_dataset(_ARCHIVE)}


def _error_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Returns the messages logged at error level.

    Args:
        caplog: The fixture that captured the log records.
    """
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
    ]


def test_main_success(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    downloader = _FakeDownloader(_ARCHIVE)

    status = pull_cli.main(
        ["sample"],
        data_root=tmp_path,
        datasets=_DATASETS,
        downloader=downloader,
    )

    assert status == 0
    assert downloader.urls == ["https://example.com/pdfs.zip"]
    assert (tmp_path / "sample" / "pdfs.zip").read_bytes() == _ARCHIVE
    assert (tmp_path / "sample" / "pdfs" / "a.pdf").read_bytes() == b"alpha"
    assert not _error_messages(caplog)


def test_main_changed_archive_no_overwrite(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    pull_cli.main(
        ["sample"],
        data_root=tmp_path,
        datasets=_DATASETS,
        downloader=_FakeDownloader(_ARCHIVE),
    )
    new_archive = _zip_bytes({"b.pdf": b"beta"})
    archive_path = tmp_path / "sample" / "pdfs.zip"

    status = pull_cli.main(
        ["sample"],
        data_root=tmp_path,
        datasets={"sample": _archive_dataset(new_archive)},
        downloader=_FakeDownloader(new_archive),
    )

    assert status == 1
    assert _error_messages(caplog) == [
        f"{archive_path}: does not match its checksum and overwrite is false"
    ]
    assert archive_path.read_bytes() == _ARCHIVE


def test_main_changed_archive_overwrite(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    pull_cli.main(
        ["sample"],
        data_root=tmp_path,
        datasets=_DATASETS,
        downloader=_FakeDownloader(_ARCHIVE),
    )
    new_archive = _zip_bytes({"b.pdf": b"beta"})

    status = pull_cli.main(
        ["sample", "--overwrite"],
        data_root=tmp_path,
        datasets={"sample": _archive_dataset(new_archive)},
        downloader=_FakeDownloader(new_archive),
    )

    assert status == 0
    assert (tmp_path / "sample" / "pdfs" / "b.pdf").read_bytes() == b"beta"
    assert not (tmp_path / "sample" / "pdfs" / "a.pdf").exists()
    assert not _error_messages(caplog)


def test_main_download_error(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    status = pull_cli.main(
        ["sample"],
        data_root=tmp_path,
        datasets=_DATASETS,
        downloader=_failing_download,
    )

    assert status == 1
    assert _error_messages(caplog) == [
        "https://example.com/pdfs.zip: connection refused"
    ]


def test_main_checksum_mismatch(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    archive_path = tmp_path / "sample" / "pdfs.zip"

    status = pull_cli.main(
        ["sample"],
        data_root=tmp_path,
        datasets=_DATASETS,
        downloader=_FakeDownloader(b"corrupted"),
    )

    assert status == 1
    messages = _error_messages(caplog)
    assert len(messages) == 1
    assert messages[0].startswith(f"{archive_path}: sha256 checksum mismatch.")


def test_main_unsafe_destination(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    data_dir = tmp_path / "sample"
    data_dir.mkdir()
    archive_path = data_dir / "pdfs.zip"
    outside_file = tmp_path / "outside.zip"
    outside_file.write_bytes(b"outside")
    archive_path.symlink_to(outside_file)

    status = pull_cli.main(
        ["sample"],
        data_root=tmp_path,
        datasets=_DATASETS,
        downloader=_FakeDownloader(_ARCHIVE),
    )

    assert status == 1
    assert _error_messages(caplog) == [
        f"{archive_path}: resolves outside {data_dir}"
    ]
    assert outside_file.read_bytes() == b"outside"


def test_main_unsafe_archive(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    member_name = "../evil.pdf"
    unsafe_archive = _zip_bytes({member_name: b"evil"})
    archive_path = tmp_path / "sample" / "pdfs.zip"

    status = pull_cli.main(
        ["sample"],
        data_root=tmp_path,
        datasets={"sample": _archive_dataset(unsafe_archive)},
        downloader=_FakeDownloader(unsafe_archive),
    )

    assert status == 1
    assert _error_messages(caplog) == [
        f"{archive_path}: member {member_name!r} resolves outside the "
        "extraction directory"
    ]


def test_main_not_a_zip(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    not_a_zip = b"not a ZIP archive"

    status = pull_cli.main(
        ["sample"],
        data_root=tmp_path,
        datasets={"sample": _archive_dataset(not_a_zip)},
        downloader=_FakeDownloader(not_a_zip),
    )

    assert status == 1
    assert _error_messages(caplog) == ["File is not a zip file"]


def test_main_unknown_dataset(tmp_path: pathlib.Path) -> None:
    downloader = _FakeDownloader(_ARCHIVE)

    with pytest.raises(SystemExit) as excinfo:
        pull_cli.main(
            ["unknown"],
            data_root=tmp_path,
            datasets=_DATASETS,
            downloader=downloader,
        )

    assert excinfo.value.code == 2
    assert not downloader.urls
    assert not list(tmp_path.iterdir())


def test_main_default_datasets(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        pull_cli.main(["--help"], data_root=tmp_path)

    assert excinfo.value.code == 0
    assert "{ohr_bench}" in capsys.readouterr().out
