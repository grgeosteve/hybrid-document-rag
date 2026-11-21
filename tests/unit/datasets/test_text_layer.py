"""Tests for docrag.datasets.text_layer."""

from collections.abc import Mapping
from collections.abc import Sequence
import pathlib
from typing import Any
from typing import cast

import fpdf
import pandas as pd
import pypdf
from pypdf import generic
import pytest

from docrag.datasets import text_layer

_COLUMNS = ["doc_name", "page_idx", "layer_text", "error"]
_PAGE_LOAD_ERROR = "page could not be loaded"


class _FakeReadPages:
    """Returns fixed page results per PDF path and records the paths read."""

    def __init__(
        self,
        results: Mapping[pathlib.Path, list[tuple[str, str | None]]],
        errors: Mapping[pathlib.Path, Exception] | None = None,
    ) -> None:
        self.paths: list[pathlib.Path] = []
        self._results = results
        self._errors = errors or {}

    def __call__(self, pdf_path: pathlib.Path) -> list[tuple[str, str | None]]:
        self.paths.append(pdf_path)
        if pdf_path in self._errors:
            raise self._errors[pdf_path]
        return list(self._results[pdf_path])


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


def _page_texts() -> pd.DataFrame:
    """Returns page texts with one page that loaded and one that did not."""
    return pd.DataFrame(
        {
            "doc_name": ["law/a", "law/a"],
            "page_idx": [0, 1],
            "layer_text": ["one", ""],
            "error": pd.Series([None, _PAGE_LOAD_ERROR], dtype="str"),
        }
    )


def _directory_contents(directory: pathlib.Path) -> dict[str, bytes]:
    """Returns the name and content of each file in a directory."""
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_unreadable_pdf_error_message() -> None:
    path = pathlib.Path("law/a.pdf")

    error = text_layer.UnreadablePdfError(path)

    assert error.path == path
    assert str(error) == "law/a.pdf: not a readable PDF"


def test_read_page_texts_text_layer(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "a.pdf"
    _write_pdf(path, ["first page", "second page"])

    assert text_layer.read_page_texts(path) == [
        ("first page", None),
        ("second page", None),
    ]


def test_read_page_texts_page_without_text_layer(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "a.pdf"
    _write_pdf(path, ["first page", ""])

    assert text_layer.read_page_texts(path) == [
        ("first page", None),
        ("", None),
    ]


def test_read_page_texts_page_fails_to_load(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "a.pdf"
    _write_pdf(path, ["first page", "second page", "third page"])
    _break_page(path, 1)

    assert text_layer.read_page_texts(path) == [
        ("first page", None),
        ("", _PAGE_LOAD_ERROR),
        ("third page", None),
    ]


def test_read_page_texts_missing_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "a.pdf"

    with pytest.raises(FileNotFoundError) as excinfo:
        text_layer.read_page_texts(path)

    assert str(excinfo.value) == f"{path}: does not exist"


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty-file"),
        pytest.param(b"not a pdf", id="not-a-pdf"),
    ],
)
def test_read_page_texts_unreadable_pdf(
    tmp_path: pathlib.Path, content: bytes
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(content)

    with pytest.raises(text_layer.UnreadablePdfError) as excinfo:
        text_layer.read_page_texts(path)

    assert excinfo.value.path == path
    assert excinfo.value.__cause__ is not None


def test_extract_page_texts_rows(tmp_path: pathlib.Path) -> None:
    read_pages = _FakeReadPages(
        {
            tmp_path / "law" / "a.pdf": [("one", None), ("", "reason")],
            tmp_path / "news" / "b.pdf": [("two", None)],
        }
    )

    page_texts = text_layer.extract_page_texts(
        tmp_path, ["law/a", "news/b"], read_pages=read_pages
    )

    expected = pd.DataFrame(
        {
            "doc_name": ["law/a", "law/a", "news/b"],
            "page_idx": [0, 1, 0],
            "layer_text": ["one", "", "two"],
            "error": pd.Series([None, "reason", None], dtype="str"),
        }
    )
    pd.testing.assert_frame_equal(page_texts, expected)


def test_extract_page_texts_pdf_paths(tmp_path: pathlib.Path) -> None:
    read_pages = _FakeReadPages(
        {
            tmp_path / "law" / "a.pdf": [("one", None)],
            tmp_path / "news" / "b.pdf": [("two", None)],
        }
    )

    text_layer.extract_page_texts(
        tmp_path, ["law/a", "news/b"], read_pages=read_pages
    )

    assert read_pages.paths == [
        tmp_path / "law" / "a.pdf",
        tmp_path / "news" / "b.pdf",
    ]


def test_extract_page_texts_document_without_pages(
    tmp_path: pathlib.Path,
) -> None:
    read_pages = _FakeReadPages(
        {
            tmp_path / "law" / "a.pdf": [],
            tmp_path / "news" / "b.pdf": [("two", None)],
        }
    )

    page_texts = text_layer.extract_page_texts(
        tmp_path, ["law/a", "news/b"], read_pages=read_pages
    )

    expected = pd.DataFrame(
        {
            "doc_name": ["news/b"],
            "page_idx": [0],
            "layer_text": ["two"],
            "error": pd.Series([None], dtype="str"),
        }
    )
    pd.testing.assert_frame_equal(page_texts, expected)


def test_extract_page_texts_no_documents(tmp_path: pathlib.Path) -> None:
    read_pages = _FakeReadPages({})

    page_texts = text_layer.extract_page_texts(
        tmp_path, [], read_pages=read_pages
    )

    assert page_texts.empty
    assert list(page_texts.columns) == _COLUMNS
    assert not read_pages.paths


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            FileNotFoundError("news/b.pdf: does not exist"), id="missing-file"
        ),
        pytest.param(
            text_layer.UnreadablePdfError(pathlib.Path("news/b.pdf")),
            id="unreadable-pdf",
        ),
    ],
)
def test_extract_page_texts_read_error(
    tmp_path: pathlib.Path, error: Exception
) -> None:
    read_pages = _FakeReadPages(
        {tmp_path / "law" / "a.pdf": [("one", None)]},
        errors={tmp_path / "news" / "b.pdf": error},
    )

    with pytest.raises(type(error)) as excinfo:
        text_layer.extract_page_texts(
            tmp_path, ["law/a", "news/b"], read_pages=read_pages
        )

    assert excinfo.value is error


def test_extract_page_texts_default_reader(tmp_path: pathlib.Path) -> None:
    (tmp_path / "law").mkdir()
    _write_pdf(tmp_path / "law" / "a.pdf", ["one", ""])

    page_texts = text_layer.extract_page_texts(tmp_path, ["law/a"])

    expected = pd.DataFrame(
        {
            "doc_name": ["law/a", "law/a"],
            "page_idx": [0, 1],
            "layer_text": ["one", ""],
            "error": pd.Series([None, None], dtype="str"),
        }
    )
    pd.testing.assert_frame_equal(page_texts, expected)


def test_extract_page_texts_progress_bar(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    read_pages = _FakeReadPages({tmp_path / "law" / "a.pdf": [("one", None)]})

    text_layer.extract_page_texts(tmp_path, ["law/a"], read_pages=read_pages)

    captured = capsys.readouterr()
    assert captured.err
    assert not captured.out


def test_save_page_texts_round_trip(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "texts.parquet"

    text_layer.save_page_texts(_page_texts(), path)

    pd.testing.assert_frame_equal(
        text_layer.load_page_texts(path), _page_texts()
    )
    assert list(_directory_contents(tmp_path)) == ["texts.parquet"]


def test_save_page_texts_existing_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "texts.parquet"
    path.write_bytes(b"earlier")

    with pytest.raises(FileExistsError) as excinfo:
        text_layer.save_page_texts(_page_texts(), path)

    assert str(excinfo.value) == f"{path}: already exists"
    assert _directory_contents(tmp_path) == {"texts.parquet": b"earlier"}


def test_save_page_texts_overwrite(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "texts.parquet"
    path.write_bytes(b"earlier")

    text_layer.save_page_texts(_page_texts(), path, overwrite=True)

    pd.testing.assert_frame_equal(
        text_layer.load_page_texts(path), _page_texts()
    )
    assert list(_directory_contents(tmp_path)) == ["texts.parquet"]


def test_save_page_texts_symlink(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "target.parquet"
    target.write_bytes(b"target")
    path = tmp_path / "texts.parquet"
    path.symlink_to(target)

    with pytest.raises(OSError) as excinfo:
        text_layer.save_page_texts(_page_texts(), path, overwrite=True)

    assert str(excinfo.value) == f"{path}: symbolic links are not allowed"
    assert path.is_symlink()
    assert target.read_bytes() == b"target"


def test_save_page_texts_directory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "texts.parquet"
    path.mkdir()

    with pytest.raises(FileExistsError) as excinfo:
        text_layer.save_page_texts(_page_texts(), path, overwrite=True)

    assert str(excinfo.value) == f"{path}: exists and is not a regular file"
    assert path.is_dir()


@pytest.mark.parametrize("column", _COLUMNS)
def test_save_page_texts_missing_column(
    tmp_path: pathlib.Path, column: str
) -> None:
    path = tmp_path / "texts.parquet"

    with pytest.raises(ValueError) as excinfo:
        text_layer.save_page_texts(_page_texts().drop(columns=[column]), path)

    assert str(excinfo.value) == f"page_texts is missing columns ['{column}']"
    assert not _directory_contents(tmp_path)


def test_save_page_texts_missing_directory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "missing" / "texts.parquet"

    with pytest.raises(FileNotFoundError):
        text_layer.save_page_texts(_page_texts(), path)

    assert not _directory_contents(tmp_path)


@pytest.mark.parametrize(
    "earlier_files",
    [
        pytest.param({}, id="no-earlier-file"),
        pytest.param({"texts.parquet": b"earlier"}, id="earlier-file"),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([object(), object()], id="objects"),
        pytest.param([{1: "a"}, {2: "b"}], id="dicts-with-integer-keys"),
        pytest.param([1 + 2j, 3 + 4j], id="complex-numbers"),
    ],
)
def test_save_page_texts_unwritable_values(
    tmp_path: pathlib.Path,
    values: list[Any],
    earlier_files: dict[str, bytes],
) -> None:
    for name, content in earlier_files.items():
        (tmp_path / name).write_bytes(content)
    path = tmp_path / "texts.parquet"

    with pytest.raises(ValueError) as excinfo:
        text_layer.save_page_texts(
            _page_texts().assign(extra=values), path, overwrite=True
        )

    assert str(excinfo.value) == f"{path}: page texts cannot be written"
    assert excinfo.value.__cause__ is not None
    assert _directory_contents(tmp_path) == earlier_files


def test_load_page_texts_missing_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "texts.parquet"

    with pytest.raises(FileNotFoundError) as excinfo:
        text_layer.load_page_texts(path)

    assert str(excinfo.value) == f"{path}: does not exist"


def test_load_page_texts_directory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "texts.parquet"
    path.mkdir()

    with pytest.raises(FileExistsError) as excinfo:
        text_layer.load_page_texts(path)

    assert str(excinfo.value) == f"{path}: exists and is not a regular file"


def test_load_page_texts_not_parquet(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "texts.parquet"
    path.write_bytes(b"not parquet")

    with pytest.raises(ValueError) as excinfo:
        text_layer.load_page_texts(path)

    assert str(excinfo.value) == f"{path}: not a parquet file"
    assert excinfo.value.__cause__ is not None


@pytest.mark.parametrize("column", _COLUMNS)
def test_load_page_texts_missing_column(
    tmp_path: pathlib.Path, column: str
) -> None:
    path = tmp_path / "texts.parquet"
    _page_texts().drop(columns=[column]).to_parquet(path)

    with pytest.raises(ValueError) as excinfo:
        text_layer.load_page_texts(path)

    assert str(excinfo.value) == f"{path}: missing columns ['{column}']"
