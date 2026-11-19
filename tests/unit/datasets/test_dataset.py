"""Tests for docrag.datasets.dataset."""

import pytest

from docrag.datasets import dataset
from docrag.file_ops import checksum
from docrag.file_ops import download

_INVALID_NAMES = [
    pytest.param("", id="empty"),
    pytest.param(".", id="dot"),
    pytest.param("..", id="dot-dot"),
    pytest.param("nested/name", id="nested"),
    pytest.param("../name", id="parent"),
    pytest.param("/name", id="absolute"),
]


def _source(filename: str) -> download.Source:
    """Returns a source with the given file name.

    Args:
        filename: The name the file is given inside the data directory.
    """
    return download.Source(
        url=f"https://example.com/{filename}",
        filename=filename,
        checksum=checksum.Checksum(
            algorithm=checksum.Algorithm.SHA256, hexdigest="a" * 64
        ),
    )


_TEXT_SOURCE = _source("text.json")
_ARCHIVE_SOURCE = _source("pdfs.zip")


def test_dataset_consistent_fields() -> None:
    value = dataset.Dataset(
        name="sample",
        sources=(_TEXT_SOURCE, _ARCHIVE_SOURCE),
        archive_source=_ARCHIVE_SOURCE,
        extract_dir_name="pdfs",
    )

    assert value.name == "sample"
    assert value.sources == (_TEXT_SOURCE, _ARCHIVE_SOURCE)
    assert value.archive_source == _ARCHIVE_SOURCE
    assert value.extract_dir_name == "pdfs"


@pytest.mark.parametrize("name", _INVALID_NAMES)
def test_dataset_invalid_name(name: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        dataset.Dataset(
            name=name,
            sources=(_ARCHIVE_SOURCE,),
            archive_source=_ARCHIVE_SOURCE,
            extract_dir_name="pdfs",
        )

    assert str(excinfo.value) == f"{name!r}: not a plain file name"


@pytest.mark.parametrize("extract_dir_name", _INVALID_NAMES)
def test_dataset_invalid_extract_dir_name(extract_dir_name: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        dataset.Dataset(
            name="sample",
            sources=(_ARCHIVE_SOURCE,),
            archive_source=_ARCHIVE_SOURCE,
            extract_dir_name=extract_dir_name,
        )

    assert str(excinfo.value) == (
        f"{extract_dir_name!r}: not a plain file name"
    )


def test_dataset_duplicate_filename() -> None:
    other_archive_source = download.Source(
        url="https://example.org/pdfs.zip",
        filename="pdfs.zip",
        checksum=_ARCHIVE_SOURCE.checksum,
    )

    with pytest.raises(ValueError) as excinfo:
        dataset.Dataset(
            name="sample",
            sources=(_ARCHIVE_SOURCE, other_archive_source),
            archive_source=_ARCHIVE_SOURCE,
            extract_dir_name="pdfs",
        )

    assert str(excinfo.value) == "'pdfs.zip': names more than one source"


def test_dataset_archive_not_in_sources() -> None:
    with pytest.raises(ValueError) as excinfo:
        dataset.Dataset(
            name="sample",
            sources=(_TEXT_SOURCE,),
            archive_source=_ARCHIVE_SOURCE,
            extract_dir_name="pdfs",
        )

    assert str(excinfo.value) == (
        "'pdfs.zip': archive source is not in sources"
    )


def test_dataset_extract_dir_name_is_source_filename() -> None:
    with pytest.raises(ValueError) as excinfo:
        dataset.Dataset(
            name="sample",
            sources=(_TEXT_SOURCE, _ARCHIVE_SOURCE),
            archive_source=_ARCHIVE_SOURCE,
            extract_dir_name="text.json",
        )

    assert str(excinfo.value) == "'text.json': also names a source file"
