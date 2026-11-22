"""Batch parsing of benchmark PDFs, its parquet files, and reference pairs."""

from collections.abc import Sequence
import os
import pathlib
import secrets

import pandas as pd
import tqdm

from docrag.file_ops import paths
from docrag.ingestion import protocol

_COLUMNS = ("doc_name", "page_idx", "text", "error")
_KEY_COLUMNS = ("doc_name", "page_idx")
_REFERENCE_COLUMNS = ("doc_name", "page_idx", "text")
_RESERVED_COLUMNS = ("reference", "candidate", "error")
_MISMATCH_NAMES_SHOWN = 5


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


def pair_with_reference(
    parsed: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """Pairs each parsed page with the reference text of the same page.

    parsed and reference are not changed.

    Args:
        parsed: One row per parsed page, with the columns doc_name,
            page_idx, text, and error. Other columns are ignored.
        reference: One row per reference page, with the columns doc_name,
            page_idx, and text, and any other columns.

    Returns:
        One row per parsed page, in the order of parsed, with an index
        from 0. The columns are those of reference in their order, with
        text renamed to reference, followed by candidate, the parsed text,
        and error. Documents in reference that are not in parsed are left
        out. No parsed pages give an empty frame with these columns.

    Raises:
        ValueError: A required column is missing, reference has a column
            named reference, candidate, or error, either frame has the
            same page twice, or the pages of a parsed document differ from
            its pages in reference.
    """
    missing = [column for column in _COLUMNS if column not in parsed]
    if missing:
        raise ValueError(f"parsed is missing columns {missing}")
    missing = [
        column for column in _REFERENCE_COLUMNS if column not in reference
    ]
    if missing:
        raise ValueError(f"reference is missing columns {missing}")
    reserved = [column for column in _RESERVED_COLUMNS if column in reference]
    if reserved:
        raise ValueError(f"reference has reserved columns {reserved}")

    key_columns = list(_KEY_COLUMNS)
    if parsed.duplicated(key_columns).any():
        raise ValueError("parsed has duplicate pages")
    if reference.duplicated(key_columns).any():
        raise ValueError("reference has duplicate pages")

    candidates = parsed[[*_KEY_COLUMNS, "text", "error"]].rename(
        columns={"text": "candidate"}
    )
    references = reference[
        reference["doc_name"].isin(candidates["doc_name"])
    ].rename(columns={"text": "reference"})

    # Fail on any document whose pages differ between parsed and reference.
    differing = pd.MultiIndex.from_frame(
        candidates[key_columns]
    ).symmetric_difference(pd.MultiIndex.from_frame(references[key_columns]))
    names = sorted(differing.get_level_values("doc_name").unique())
    if names:
        unit = "document" if len(names) == 1 else "documents"
        raise ValueError(
            f"page mismatch in {len(names)} {unit}, including "
            f"{names[:_MISMATCH_NAMES_SHOWN]}"
        )

    pairs = candidates.reset_index(drop=True).merge(
        references.reset_index(drop=True), on=key_columns, how="inner"
    )
    return pairs[[*references.columns, "candidate", "error"]]
