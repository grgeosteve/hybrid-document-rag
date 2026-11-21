"""The PDF parser interface, the page it returns, and its error."""

import dataclasses
import pathlib
import typing


@dataclasses.dataclass(frozen=True)
class ParsedPage:
    """The text of one page of a PDF.

    Attributes:
        text: The text of the page. Empty if the page has no text or could
            not be parsed.
        error: Why the page could not be parsed, or None if it was parsed.
    """

    text: str
    error: str | None = None

    def __post_init__(self) -> None:
        """Validates the fields.

        Raises:
            ValueError: error is an empty string, or error is set and text
                is not empty.
        """
        if self.error is None:
            return
        if not self.error:
            raise ValueError(f"{self.error!r}: error reason is empty")
        if self.text:
            raise ValueError(f"{self.error!r}: page with an error has text")


class UnreadablePdfError(Exception):
    """A file could not be read as a PDF.

    Attributes:
        path: The file that could not be read.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """Initialises the error with the file.

        Args:
            path: The file that could not be read.
        """
        super().__init__(f"{path}: not a readable PDF")
        self.path = path


class PdfParser(typing.Protocol):
    """Extracts the text of each page of a PDF."""

    def parse(self, pdf_path: pathlib.Path) -> list[ParsedPage]:
        """Parses each page of a PDF.

        Args:
            pdf_path: The PDF file.

        Returns:
            One parsed page per page of the PDF, in page order. A page that
            cannot be parsed has empty text and the reason as its error.

        Raises:
            FileNotFoundError: pdf_path does not exist.
            UnreadablePdfError: The file cannot be opened as a PDF.
        """
