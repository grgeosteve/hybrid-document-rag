"""PDF parser backends and the parser for each."""

import enum
import typing

from docrag.ingestion import protocol
from docrag.ingestion.backends import pypdfium2


class ParserBackend(enum.Enum):
    """A backend a PDF parser can use.

    Attributes:
        PYPDFIUM2: Reads the text layer of each page with pypdfium2.
    """

    PYPDFIUM2 = "pypdfium2"


def create_pdf_parser(backend: ParserBackend) -> protocol.PdfParser:
    """Returns a PDF parser that uses the given backend.

    Args:
        backend: The backend to parse with.

    Returns:
        A parser that uses backend.
    """
    match backend:
        case ParserBackend.PYPDFIUM2:
            return pypdfium2.Pypdfium2Parser()
        case _:  # pragma: no cover
            typing.assert_never(backend)
