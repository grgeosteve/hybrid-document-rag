"""Tests for docrag.ingestion.protocol."""

import dataclasses
import pathlib

import pytest

from docrag.ingestion import protocol


@pytest.mark.parametrize(
    "text, error",
    [
        pytest.param("one", None, id="text"),
        pytest.param("", None, id="blank-page"),
        pytest.param("", "reason", id="damaged-page"),
    ],
)
def test_parsed_page_valid(text: str, error: str | None) -> None:
    page = protocol.ParsedPage(text, error)

    assert page.text == text
    assert page.error == error


def test_parsed_page_default_error() -> None:
    assert protocol.ParsedPage("one").error is None


@pytest.mark.parametrize(
    "text, error, message",
    [
        pytest.param("", "", "'': error reason is empty", id="empty-error"),
        pytest.param(
            "one",
            "reason",
            "'reason': page with an error has text",
            id="error-with-text",
        ),
    ],
)
def test_parsed_page_invalid(text: str, error: str, message: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        protocol.ParsedPage(text, error)

    assert str(excinfo.value) == message


def test_parsed_page_frozen() -> None:
    page = protocol.ParsedPage("one")

    with pytest.raises(dataclasses.FrozenInstanceError):
        page.text = "two"  # type: ignore[misc]


def test_unreadable_pdf_error_message() -> None:
    path = pathlib.Path("law/a.pdf")

    error = protocol.UnreadablePdfError(path)

    assert error.path == path
    assert str(error) == "law/a.pdf: not a readable PDF"
