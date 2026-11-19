"""Tests for docrag.datasets.ohr_bench."""

import re
from urllib import parse

import pytest

from docrag.datasets import ohr_bench
from docrag.file_ops import checksum
from docrag.file_ops import download

_HEX_DIGEST_LENGTHS = {
    checksum.Algorithm.SHA256: 64,
    checksum.Algorithm.GIT_BLOB_SHA1: 40,
}
_URL_CHARACTERS = re.compile(r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+")


def _filename_id(source: download.Source) -> str:
    """Returns the test ID of a source, which is its file name.

    Args:
        source: The source under test.
    """
    return source.filename


def test_dataset_files() -> None:
    assert ohr_bench.DATASET.name == "ohr_bench"
    assert [source.filename for source in ohr_bench.DATASET.sources] == [
        ohr_bench.PDFS_ARCHIVE,
        ohr_bench.GROUND_TRUTH_FILE,
        ohr_bench.QA_FILE,
    ]
    assert ohr_bench.DATASET.archive_source.filename == ohr_bench.PDFS_ARCHIVE
    assert ohr_bench.DATASET.extract_dir_name == ohr_bench.PDFS_DIR


@pytest.mark.parametrize("source", ohr_bench.DATASET.sources, ids=_filename_id)
def test_sources_url_ends_with_filename(source: download.Source) -> None:
    assert source.url.split("/")[-1] == source.filename


@pytest.mark.parametrize("source", ohr_bench.DATASET.sources, ids=_filename_id)
def test_sources_digest_matches_algorithm(source: download.Source) -> None:
    digest_length = len(source.checksum.hexdigest)
    checksum_alg_length = _HEX_DIGEST_LENGTHS[source.checksum.algorithm]
    assert digest_length == checksum_alg_length


@pytest.mark.parametrize("source", ohr_bench.DATASET.sources, ids=_filename_id)
def test_sources_digest_is_hex(source: download.Source) -> None:
    assert set(source.checksum.hexdigest) <= set("0123456789abcdef")


@pytest.mark.parametrize("source", ohr_bench.DATASET.sources, ids=_filename_id)
def test_sources_url_is_https_with_allowed_characters(
    source: download.Source,
) -> None:
    parts = parse.urlsplit(source.url)
    assert _URL_CHARACTERS.fullmatch(source.url)
    assert parts.scheme == "https"
    assert parts.hostname
    assert parts.port is None
