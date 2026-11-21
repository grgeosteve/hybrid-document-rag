"""Fixtures shared by the unit tests."""

from collections.abc import Callable
from collections.abc import Sequence
import pathlib
from typing import cast

import fpdf
import pypdf
from pypdf import generic
import pytest


def _write_pdf(path: pathlib.Path, page_texts: Sequence[str]) -> None:
    """Writes a PDF with one page per text.

    Args:
        path: The PDF file to write.
        page_texts: The text of each page. An empty text gives a page with
            no text layer.
    """
    pdf = fpdf.FPDF()
    pdf.set_font("Helvetica")
    for text in page_texts:
        pdf.add_page()
        if text:
            pdf.cell(text=text)
    pdf.output(path)


def _break_page(path: pathlib.Path, index: int) -> None:
    """Makes one page of a PDF impossible to load.

    The page's entry in the page tree is replaced by a number.

    Args:
        path: The PDF file to change.
        index: The index of the page, from 0.
    """
    writer = pypdf.PdfWriter(clone_from=path)
    page_tree = cast(generic.DictionaryObject, writer.root_object["/Pages"])
    kids = cast(generic.ArrayObject, page_tree["/Kids"])
    kids[index] = generic.NumberObject(0)
    writer.write(path)


@pytest.fixture(name="write_pdf")
def fixture_write_pdf() -> Callable[[pathlib.Path, Sequence[str]], None]:
    """Returns a function that writes a PDF with one page per text."""
    return _write_pdf


@pytest.fixture(name="break_page")
def fixture_break_page() -> Callable[[pathlib.Path, int], None]:
    """Returns a function that makes one page of a PDF impossible to load."""
    return _break_page
