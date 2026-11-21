"""Per-page text layers of PDF files, and their parquet files."""

from collections.abc import Callable
from collections.abc import Sequence
import os
import pathlib
import secrets

import pandas as pd
import pypdfium2 as pdfium
import tqdm

from docrag.file_ops import paths

_TEXT_COLUMNS = ("doc_name", "page_idx", "layer_text", "error")
_PAGE_LOAD_ERROR = "page could not be loaded"


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


def read_page_texts(pdf_path: pathlib.Path) -> list[tuple[str, str | None]]:
    """Reads the text layer of each page of a PDF.

    Args:
        pdf_path: The PDF file.

    Returns:
        One (text, error) pair per page, in page order. For a page that
        loads, text is its text layer, empty if it has none, and error is
        None. For a page that fails to load, text is empty and error is
        "page could not be loaded".

    Raises:
        FileNotFoundError: pdf_path does not exist.
        UnreadablePdfError: The file cannot be opened as a PDF.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path}: does not exist")

    page_texts: list[tuple[str, str | None]] = []
    try:
        pdf = pdfium.PdfDocument(pdf_path)

    except pdfium.PdfiumError as e:
        raise UnreadablePdfError(pdf_path) from e

    try:
        for i in range(len(pdf)):
            try:
                page = pdf[i]
                textpage = page.get_textpage()
                text = textpage.get_text_bounded()
            except pdfium.PdfiumError:
                page_texts.append(("", _PAGE_LOAD_ERROR))
            else:
                page_texts.append((text, None))
    finally:
        pdf.close()

    return page_texts


def extract_page_texts(
    pdf_dir: pathlib.Path,
    doc_names: Sequence[str],
    *,
    read_pages: Callable[
        [pathlib.Path], list[tuple[str, str | None]]
    ] = read_page_texts,
) -> pd.DataFrame:
    """Reads the text layer of every page of the given documents.

    Shows a progress bar on standard error.

    Args:
        pdf_dir: The directory that holds the PDFs.
        doc_names: The documents, each read from pdf_dir / f"{name}.pdf".
        read_pages: The function that returns the (text, error) pair of
            each page.

    Returns:
        One row per page, with the columns doc_name, page_idx (from 0),
        layer_text, and error, in the order of doc_names and pages. error
        is missing for a page that loaded, and the reason for one that did
        not. No documents give an empty frame with these columns.

    Raises:
        FileNotFoundError: A PDF does not exist.
        UnreadablePdfError: A file cannot be opened as a PDF.
    """
    doc_dataframes: list[pd.DataFrame] = []
    for doc_name in tqdm.tqdm(doc_names, unit="document"):
        doc_path = pdf_dir / f"{doc_name}.pdf"

        page_results = read_pages(doc_path)
        df = pd.DataFrame(
            {
                "doc_name": doc_name,
                "page_idx": range(len(page_results)),
                "layer_text": pd.Series(
                    [text for text, _ in page_results], dtype="str"
                ),
                "error": pd.Series(
                    [error for _, error in page_results], dtype="str"
                ),
            }
        )
        doc_dataframes.append(df)

    if not doc_dataframes:
        return pd.DataFrame(columns=list(_TEXT_COLUMNS))

    documents_df = pd.concat(doc_dataframes, ignore_index=True)

    return documents_df


def save_page_texts(
    page_texts: pd.DataFrame, path: pathlib.Path, *, overwrite: bool = False
) -> None:
    """Writes page texts to a parquet file.

    An interrupted call never leaves a file at path.

    Args:
        page_texts: One row per page, with the columns doc_name, page_idx,
            layer_text, and error.
        path: The parquet file to write.
        overwrite: Whether to replace an existing file.

    Raises:
        OSError: path is a symbolic link.
        FileExistsError: path exists and overwrite is false, or path exists
            and is not a regular file.
        ValueError: page_texts lacks a required column, or holds values
            that cannot be written.
        FileNotFoundError: The directory of path does not exist.
    """
    paths.reject_symlink(path)
    paths.reject_non_file(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path}: already exists")

    missing = [
        column for column in _TEXT_COLUMNS if column not in page_texts.columns
    ]
    if missing:
        raise ValueError(f"page_texts is missing columns {missing}")

    partial = path.with_name(f"{path.name}.partial-{secrets.token_hex(8)}")
    try:
        with partial.open("xb") as file:
            # Values that the file format cannot store raise ValueError,
            # TypeError, or NotImplementedError.
            try:
                page_texts.to_parquet(file)
            except (ValueError, TypeError, NotImplementedError) as e:
                raise ValueError(f"{path}: page texts cannot be written") from e
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def load_page_texts(path: pathlib.Path) -> pd.DataFrame:
    """Reads page texts from a parquet file.

    Args:
        path: The parquet file of page texts.

    Returns:
        One row per page, with the columns doc_name, page_idx, layer_text,
        and error.

    Raises:
        FileExistsError: path exists and is not a regular file.
        FileNotFoundError: path does not exist.
        ValueError: The file is not a parquet file, or lacks a required
            column.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path}: does not exist")
    paths.reject_non_file(path)
    try:
        page_texts = pd.read_parquet(path)
    except ValueError as e:
        raise ValueError(f"{path}: not a parquet file") from e

    missing = [
        column for column in _TEXT_COLUMNS if column not in page_texts.columns
    ]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    return page_texts
