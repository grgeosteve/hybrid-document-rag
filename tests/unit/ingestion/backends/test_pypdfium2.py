"""Tests for docrag.ingestion.backends.pypdfium2."""

from collections.abc import Callable
from collections.abc import Sequence
import pathlib

import pytest

from docrag.ingestion import protocol
from docrag.ingestion.backends import pypdfium2

_WritePdf = Callable[[pathlib.Path, Sequence[str]], None]
_BreakPage = Callable[[pathlib.Path, int], None]
_PAGE_LOAD_ERROR = "page could not be loaded"


def test_parse_text_layer(tmp_path: pathlib.Path, write_pdf: _WritePdf) -> None:
    path = tmp_path / "a.pdf"
    write_pdf(path, ["first page", "second page"])

    assert pypdfium2.Pypdfium2Parser().parse(path) == [
        protocol.ParsedPage("first page"),
        protocol.ParsedPage("second page"),
    ]


def test_parse_page_without_text_layer(
    tmp_path: pathlib.Path, write_pdf: _WritePdf
) -> None:
    path = tmp_path / "a.pdf"
    write_pdf(path, ["first page", ""])

    assert pypdfium2.Pypdfium2Parser().parse(path) == [
        protocol.ParsedPage("first page"),
        protocol.ParsedPage(""),
    ]


def test_parse_page_fails_to_load(
    tmp_path: pathlib.Path, write_pdf: _WritePdf, break_page: _BreakPage
) -> None:
    path = tmp_path / "a.pdf"
    write_pdf(path, ["first page", "second page", "third page"])
    break_page(path, 1)

    assert pypdfium2.Pypdfium2Parser().parse(path) == [
        protocol.ParsedPage("first page"),
        protocol.ParsedPage("", _PAGE_LOAD_ERROR),
        protocol.ParsedPage("third page"),
    ]


def test_parse_missing_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "a.pdf"

    with pytest.raises(FileNotFoundError) as excinfo:
        pypdfium2.Pypdfium2Parser().parse(path)

    assert str(excinfo.value) == f"{path}: does not exist"


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty-file"),
        pytest.param(b"not a pdf", id="not-a-pdf"),
    ],
)
def test_parse_unreadable_pdf(tmp_path: pathlib.Path, content: bytes) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(content)

    with pytest.raises(protocol.UnreadablePdfError) as excinfo:
        pypdfium2.Pypdfium2Parser().parse(path)

    assert excinfo.value.path == path
    assert excinfo.value.__cause__ is not None
