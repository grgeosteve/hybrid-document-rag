"""Parsed pages of benchmark PDFs, and their parquet files."""

from collections.abc import Sequence
import os
import pathlib
import secrets

import pandas as pd
import tqdm

from docrag.file_ops import paths
from docrag.ingestion import protocol

_COLUMNS = ("doc_name", "page_idx", "text", "error")


def parse_documents(
    pdf_dir: pathlib.Path,
    doc_names: Sequence[str],
    *,
    parser: protocol.PdfParser,
) -> pd.DataFrame:
    """Parses every page of the given documents.

    Shows a progress bar on standard error.

    Args:
        pdf_dir: The directory that holds the PDFs.
        doc_names: The documents, each read from pdf_dir / f"{name}.pdf".
        parser: The parser to read each PDF with.

    Returns:
        One row per page, with the columns doc_name, page_idx (from 0),
        text, and error, in the order of doc_names and pages. error is
        missing for a page that was parsed, and the reason for one that was
        not. No documents give an empty frame with these columns.

    Raises:
        FileNotFoundError: A PDF does not exist.
        UnreadablePdfError: A file cannot be opened as a PDF.
    """
    doc_dataframes: list[pd.DataFrame] = []
    for doc_name in tqdm.tqdm(doc_names, unit="document"):
        pages = parser.parse(pdf_dir / f"{doc_name}.pdf")
        df = pd.DataFrame(
            {
                "doc_name": doc_name,
                "page_idx": range(len(pages)),
                "text": pd.Series([page.text for page in pages], dtype="str"),
                "error": pd.Series([page.error for page in pages], dtype="str"),
            }
        )
        doc_dataframes.append(df)

    if not doc_dataframes:
        return pd.DataFrame(columns=list(_COLUMNS))

    return pd.concat(doc_dataframes, ignore_index=True)


def save_parsed_pages(
    parsed_pages: pd.DataFrame, path: pathlib.Path, *, overwrite: bool = False
) -> None:
    """Writes parsed pages to a parquet file.

    An interrupted call never leaves a file at path.

    Args:
        parsed_pages: One row per page, with the columns doc_name,
            page_idx, text, and error.
        path: The parquet file to write.
        overwrite: Whether to replace an existing file.

    Raises:
        OSError: path is a symbolic link.
        FileExistsError: path exists and overwrite is false, or path exists
            and is not a regular file.
        ValueError: parsed_pages lacks a required column, or holds values
            that cannot be written.
        FileNotFoundError: The directory of path does not exist.
    """
    paths.reject_symlink(path)
    paths.reject_non_file(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path}: already exists")

    missing = [
        column for column in _COLUMNS if column not in parsed_pages.columns
    ]
    if missing:
        raise ValueError(f"parsed_pages is missing columns {missing}")

    partial = path.with_name(f"{path.name}.partial-{secrets.token_hex(8)}")
    try:
        with partial.open("xb") as file:
            # Values that the file format cannot store raise ValueError,
            # TypeError, or NotImplementedError.
            try:
                parsed_pages.to_parquet(file)
            except (ValueError, TypeError, NotImplementedError) as e:
                raise ValueError(
                    f"{path}: parsed pages cannot be written"
                ) from e
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def load_parsed_pages(path: pathlib.Path) -> pd.DataFrame:
    """Reads parsed pages from a parquet file.

    Args:
        path: The parquet file of parsed pages.

    Returns:
        One row per page, with the columns doc_name, page_idx, text, and
        error.

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
        parsed_pages = pd.read_parquet(path)
    except ValueError as e:
        raise ValueError(f"{path}: not a parquet file") from e

    missing = [
        column for column in _COLUMNS if column not in parsed_pages.columns
    ]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    return parsed_pages
