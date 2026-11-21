"""Tests for docrag.ingestion.factory."""

import pytest

from docrag.ingestion import factory
from docrag.ingestion.backends import pypdfium2


def test_parser_backend_name() -> None:
    assert factory.ParserBackend("pypdfium2") is factory.ParserBackend.PYPDFIUM2


def test_parser_backend_unknown_name() -> None:
    with pytest.raises(ValueError):
        factory.ParserBackend("pypdfium")


def test_create_pdf_parser_pypdfium2() -> None:
    parser = factory.create_pdf_parser(factory.ParserBackend.PYPDFIUM2)

    assert isinstance(parser, pypdfium2.Pypdfium2Parser)
