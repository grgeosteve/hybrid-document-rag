"""A PDF parser that reads the text layer of each page with pypdfium2."""

import pathlib

import pypdfium2 as pdfium

from docrag.ingestion import protocol

_PAGE_LOAD_ERROR = "page could not be loaded"


class Pypdfium2Parser(protocol.PdfParser):
    """Extracts the text layer of each page of a PDF."""

    def parse(self, pdf_path: pathlib.Path) -> list[protocol.ParsedPage]:
        """Reads the text layer of each page of a PDF.

        Args:
            pdf_path: The PDF file.

        Returns:
            One parsed page per page of the PDF, in page order. A page that
            loads has its text layer as text, empty if it has none. A page
            that fails to load has empty text and the error "page could not
            be loaded".

        Raises:
            FileNotFoundError: pdf_path does not exist.
            UnreadablePdfError: The file cannot be opened as a PDF.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path}: does not exist")

        try:
            pdf = pdfium.PdfDocument(pdf_path)
        except pdfium.PdfiumError as e:
            raise protocol.UnreadablePdfError(pdf_path) from e

        pages: list[protocol.ParsedPage] = []
        try:
            for i in range(len(pdf)):
                try:
                    page = pdf[i]
                    textpage = page.get_textpage()
                    text = textpage.get_text_bounded()
                except pdfium.PdfiumError:
                    pages.append(protocol.ParsedPage("", _PAGE_LOAD_ERROR))
                else:
                    pages.append(protocol.ParsedPage(text))
        finally:
            pdf.close()

        return pages
