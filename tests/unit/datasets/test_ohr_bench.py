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


def test_sources_filenames_unique() -> None:
    filenames = [source.filename for source in ohr_bench.SOURCES]
    assert len(set(filenames)) == len(filenames)


@pytest.mark.parametrize("source", ohr_bench.SOURCES, ids=_filename_id)
def test_sources_url_ends_with_filename(source: download.Source) -> None:
    assert source.url.split("/")[-1] == source.filename


@pytest.mark.parametrize("source", ohr_bench.SOURCES, ids=_filename_id)
def test_sources_digest_matches_algorithm(source: download.Source) -> None:
    digest_length = len(source.checksum.hexdigest)
    checksum_alg_length = _HEX_DIGEST_LENGTHS[source.checksum.algorithm]
    assert digest_length == checksum_alg_length


@pytest.mark.parametrize("source", ohr_bench.SOURCES, ids=_filename_id)
def test_sources_digest_is_hex(source: download.Source) -> None:
    assert set(source.checksum.hexdigest) <= set("0123456789abcdef")


@pytest.mark.parametrize("source", ohr_bench.SOURCES, ids=_filename_id)
def test_sources_url_is_https_with_allowed_characters(
    source: download.Source,
) -> None:
    parts = parse.urlsplit(source.url)
    assert _URL_CHARACTERS.fullmatch(source.url)
    assert parts.scheme == "https"
    assert parts.hostname
    assert parts.port is None


def test_pdfs_source_in_sources() -> None:
    assert ohr_bench.PDFS_SOURCE in ohr_bench.SOURCES
