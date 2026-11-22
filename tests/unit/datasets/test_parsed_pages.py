"""Tests for docrag.datasets.parsed_pages."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import pathlib
from typing import Any

import pandas as pd
import pytest

from docrag.datasets import parsed_pages
from docrag.ingestion import factory
from docrag.ingestion import protocol

_WritePdf = Callable[[pathlib.Path, Sequence[str]], None]
_COLUMNS = ["doc_name", "page_idx", "text", "error"]
_PAGE_LOAD_ERROR = "page could not be loaded"


class _FakeParser:
    """Returns fixed pages per PDF path and records the paths parsed."""

    def __init__(
        self,
        results: Mapping[pathlib.Path, list[protocol.ParsedPage]],
        errors: Mapping[pathlib.Path, Exception] | None = None,
    ) -> None:
        self.paths: list[pathlib.Path] = []
        self._results = results
        self._errors = errors or {}

    def parse(self, pdf_path: pathlib.Path) -> list[protocol.ParsedPage]:
        self.paths.append(pdf_path)
        if pdf_path in self._errors:
            raise self._errors[pdf_path]
        return list(self._results[pdf_path])


def _sample_pages() -> pd.DataFrame:
    """Returns parsed pages with one page that loaded and one that did not."""
    return pd.DataFrame(
        {
            "doc_name": ["law/a", "law/a"],
            "page_idx": [0, 1],
            "text": ["one", ""],
            "error": pd.Series([None, _PAGE_LOAD_ERROR], dtype="str"),
        }
    )


def _parsed(rows: Sequence[tuple[str, int, str, str | None]]) -> pd.DataFrame:
    """Returns parsed pages from (doc_name, page_idx, text, error) rows."""
    return pd.DataFrame(list(rows), columns=_COLUMNS)


def _reference(rows: Sequence[tuple[str, int, str]]) -> pd.DataFrame:
    """Returns reference pages from (doc_name, page_idx, text) rows."""
    return pd.DataFrame(list(rows), columns=["doc_name", "page_idx", "text"])


def _directory_contents(directory: pathlib.Path) -> dict[str, bytes]:
    """Returns the name and content of each file in a directory."""
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_parse_documents_rows(tmp_path: pathlib.Path) -> None:
    parser = _FakeParser(
        {
            tmp_path / "law" / "a.pdf": [
                protocol.ParsedPage("one"),
                protocol.ParsedPage("", "reason"),
            ],
            tmp_path / "news" / "b.pdf": [protocol.ParsedPage("two")],
        }
    )

    pages = parsed_pages.parse_documents(
        tmp_path, ["law/a", "news/b"], parser=parser
    )

    expected = pd.DataFrame(
        {
            "doc_name": ["law/a", "law/a", "news/b"],
            "page_idx": [0, 1, 0],
            "text": ["one", "", "two"],
            "error": pd.Series([None, "reason", None], dtype="str"),
        }
    )
    pd.testing.assert_frame_equal(pages, expected)


def test_parse_documents_pdf_paths(tmp_path: pathlib.Path) -> None:
    parser = _FakeParser(
        {
            tmp_path / "law" / "a.pdf": [protocol.ParsedPage("one")],
            tmp_path / "news" / "b.pdf": [protocol.ParsedPage("two")],
        }
    )

    parsed_pages.parse_documents(tmp_path, ["law/a", "news/b"], parser=parser)

    assert parser.paths == [
        tmp_path / "law" / "a.pdf",
        tmp_path / "news" / "b.pdf",
    ]


def test_parse_documents_document_without_pages(
    tmp_path: pathlib.Path,
) -> None:
    parser = _FakeParser(
        {
            tmp_path / "law" / "a.pdf": [],
            tmp_path / "news" / "b.pdf": [protocol.ParsedPage("two")],
        }
    )

    pages = parsed_pages.parse_documents(
        tmp_path, ["law/a", "news/b"], parser=parser
    )

    expected = pd.DataFrame(
        {
            "doc_name": ["news/b"],
            "page_idx": [0],
            "text": ["two"],
            "error": pd.Series([None], dtype="str"),
        }
    )
    pd.testing.assert_frame_equal(pages, expected)


def test_parse_documents_no_documents(tmp_path: pathlib.Path) -> None:
    parser = _FakeParser({})

    pages = parsed_pages.parse_documents(tmp_path, [], parser=parser)

    assert pages.empty
    assert list(pages.columns) == _COLUMNS
    assert not parser.paths


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            FileNotFoundError("news/b.pdf: does not exist"), id="missing-file"
        ),
        pytest.param(
            protocol.UnreadablePdfError(pathlib.Path("news/b.pdf")),
            id="unreadable-pdf",
        ),
    ],
)
def test_parse_documents_parse_error(
    tmp_path: pathlib.Path, error: Exception
) -> None:
    parser = _FakeParser(
        {tmp_path / "law" / "a.pdf": [protocol.ParsedPage("one")]},
        errors={tmp_path / "news" / "b.pdf": error},
    )

    with pytest.raises(type(error)) as excinfo:
        parsed_pages.parse_documents(
            tmp_path, ["law/a", "news/b"], parser=parser
        )

    assert excinfo.value is error


def test_parse_documents_pypdfium2_backend(
    tmp_path: pathlib.Path, write_pdf: _WritePdf
) -> None:
    (tmp_path / "law").mkdir()
    write_pdf(tmp_path / "law" / "a.pdf", ["one", ""])
    parser = factory.create_pdf_parser(factory.ParserBackend.PYPDFIUM2)

    pages = parsed_pages.parse_documents(tmp_path, ["law/a"], parser=parser)

    expected = pd.DataFrame(
        {
            "doc_name": ["law/a", "law/a"],
            "page_idx": [0, 1],
            "text": ["one", ""],
            "error": pd.Series([None, None], dtype="str"),
        }
    )
    pd.testing.assert_frame_equal(pages, expected)


def test_parse_documents_progress_bar(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = _FakeParser(
        {tmp_path / "law" / "a.pdf": [protocol.ParsedPage("one")]}
    )

    parsed_pages.parse_documents(tmp_path, ["law/a"], parser=parser)

    captured = capsys.readouterr()
    assert captured.err
    assert not captured.out


def test_save_parsed_pages_round_trip(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pages.parquet"

    parsed_pages.save_parsed_pages(_sample_pages(), path)

    pd.testing.assert_frame_equal(
        parsed_pages.load_parsed_pages(path), _sample_pages()
    )
    assert list(_directory_contents(tmp_path)) == ["pages.parquet"]


def test_save_parsed_pages_existing_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pages.parquet"
    path.write_bytes(b"earlier")

    with pytest.raises(FileExistsError) as excinfo:
        parsed_pages.save_parsed_pages(_sample_pages(), path)

    assert str(excinfo.value) == f"{path}: already exists"
    assert _directory_contents(tmp_path) == {"pages.parquet": b"earlier"}


def test_save_parsed_pages_overwrite(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pages.parquet"
    path.write_bytes(b"earlier")

    parsed_pages.save_parsed_pages(_sample_pages(), path, overwrite=True)

    pd.testing.assert_frame_equal(
        parsed_pages.load_parsed_pages(path), _sample_pages()
    )
    assert list(_directory_contents(tmp_path)) == ["pages.parquet"]


def test_save_parsed_pages_symlink(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "target.parquet"
    target.write_bytes(b"target")
    path = tmp_path / "pages.parquet"
    path.symlink_to(target)

    with pytest.raises(OSError) as excinfo:
        parsed_pages.save_parsed_pages(_sample_pages(), path, overwrite=True)

    assert str(excinfo.value) == f"{path}: symbolic links are not allowed"
    assert path.is_symlink()
    assert target.read_bytes() == b"target"


def test_save_parsed_pages_directory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pages.parquet"
    path.mkdir()

    with pytest.raises(FileExistsError) as excinfo:
        parsed_pages.save_parsed_pages(_sample_pages(), path, overwrite=True)

    assert str(excinfo.value) == f"{path}: exists and is not a regular file"
    assert path.is_dir()


@pytest.mark.parametrize("column", _COLUMNS)
def test_save_parsed_pages_missing_column(
    tmp_path: pathlib.Path, column: str
) -> None:
    path = tmp_path / "pages.parquet"

    with pytest.raises(ValueError) as excinfo:
        parsed_pages.save_parsed_pages(
            _sample_pages().drop(columns=[column]), path
        )

    assert str(excinfo.value) == f"parsed_pages is missing columns ['{column}']"
    assert not _directory_contents(tmp_path)


def test_save_parsed_pages_missing_directory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "missing" / "pages.parquet"

    with pytest.raises(FileNotFoundError):
        parsed_pages.save_parsed_pages(_sample_pages(), path)

    assert not _directory_contents(tmp_path)


@pytest.mark.parametrize(
    "earlier_files",
    [
        pytest.param({}, id="no-earlier-file"),
        pytest.param({"pages.parquet": b"earlier"}, id="earlier-file"),
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
def test_save_parsed_pages_unwritable_values(
    tmp_path: pathlib.Path,
    values: list[Any],
    earlier_files: dict[str, bytes],
) -> None:
    for name, content in earlier_files.items():
        (tmp_path / name).write_bytes(content)
    path = tmp_path / "pages.parquet"

    with pytest.raises(ValueError) as excinfo:
        parsed_pages.save_parsed_pages(
            _sample_pages().assign(extra=values), path, overwrite=True
        )

    assert str(excinfo.value) == f"{path}: parsed pages cannot be written"
    assert excinfo.value.__cause__ is not None
    assert _directory_contents(tmp_path) == earlier_files


def test_load_parsed_pages_missing_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pages.parquet"

    with pytest.raises(FileNotFoundError) as excinfo:
        parsed_pages.load_parsed_pages(path)

    assert str(excinfo.value) == f"{path}: does not exist"


def test_load_parsed_pages_directory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pages.parquet"
    path.mkdir()

    with pytest.raises(FileExistsError) as excinfo:
        parsed_pages.load_parsed_pages(path)

    assert str(excinfo.value) == f"{path}: exists and is not a regular file"


def test_load_parsed_pages_not_parquet(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pages.parquet"
    path.write_bytes(b"not parquet")

    with pytest.raises(ValueError) as excinfo:
        parsed_pages.load_parsed_pages(path)

    assert str(excinfo.value) == f"{path}: not a parquet file"
    assert excinfo.value.__cause__ is not None


@pytest.mark.parametrize("column", _COLUMNS)
def test_load_parsed_pages_missing_column(
    tmp_path: pathlib.Path, column: str
) -> None:
    path = tmp_path / "pages.parquet"
    _sample_pages().drop(columns=[column]).to_parquet(path)

    with pytest.raises(ValueError) as excinfo:
        parsed_pages.load_parsed_pages(path)

    assert str(excinfo.value) == f"{path}: missing columns ['{column}']"


@pytest.mark.parametrize(
    "index_name",
    [None, "doc_name", "page_idx"],
    ids=["unnamed", "document-name", "page-index"],
)
def test_pair_with_reference_rows(index_name: str | None) -> None:
    parsed = _parsed(
        [
            ("a", 0, "one", None),
            ("a", 1, "", _PAGE_LOAD_ERROR),
            ("b", 0, "two", None),
        ]
    )
    reference = _reference(
        [("b", 0, "TWO"), ("a", 1, "one more"), ("a", 0, "ONE")]
    )

    pairs = parsed_pages.pair_with_reference(
        parsed.rename_axis(index_name), reference.rename_axis(index_name)
    )

    expected = pd.DataFrame(
        [
            ("a", 0, "ONE", "one", None),
            ("a", 1, "one more", "", _PAGE_LOAD_ERROR),
            ("b", 0, "TWO", "two", None),
        ],
        columns=["doc_name", "page_idx", "reference", "candidate", "error"],
    )
    pd.testing.assert_frame_equal(pairs, expected)


@pytest.mark.parametrize(
    "metadata, columns",
    [
        pytest.param(
            {},
            ["doc_name", "page_idx", "reference", "candidate", "error"],
            id="no-metadata",
        ),
        pytest.param(
            {"source": "x", "group": "y"},
            [
                "doc_name",
                "page_idx",
                "reference",
                "source",
                "group",
                "candidate",
                "error",
            ],
            id="two-metadata-columns",
        ),
    ],
)
def test_pair_with_reference_metadata_columns(
    metadata: dict[str, str], columns: list[str]
) -> None:
    parsed = _parsed([("a", 0, "one", None)])
    reference = _reference([("a", 0, "ONE")]).assign(**metadata)

    pairs = parsed_pages.pair_with_reference(parsed, reference)

    assert list(pairs.columns) == columns
    for name, value in metadata.items():
        assert pairs[name].to_list() == [value]


def test_pair_with_reference_unparsed_documents() -> None:
    parsed = _parsed([("a", 0, "one", None)])
    reference = _reference([("a", 0, "ONE"), ("b", 0, "TWO")])

    pairs = parsed_pages.pair_with_reference(parsed, reference)

    assert pairs["doc_name"].to_list() == ["a"]


def test_pair_with_reference_no_parsed_pages() -> None:
    parsed = _parsed([])
    reference = _reference([("a", 0, "ONE")])

    pairs = parsed_pages.pair_with_reference(parsed, reference)

    assert pairs.empty
    assert list(pairs.columns) == [
        "doc_name",
        "page_idx",
        "reference",
        "candidate",
        "error",
    ]


@pytest.mark.parametrize(
    "parsed_rows, reference_rows",
    [
        pytest.param(
            [("a", 0, "one", None)],
            [("a", 0, "ONE"), ("a", 1, "TWO")],
            id="reference-extra-page",
        ),
        pytest.param(
            [("a", 0, "one", None), ("a", 1, "two", None)],
            [("a", 0, "ONE")],
            id="parsed-extra-page",
        ),
        pytest.param(
            [("b", 0, "one", None), ("a", 0, "two", None)],
            [("b", 0, "ONE"), ("c", 0, "TWO")],
            id="document-not-in-reference",
        ),
    ],
)
def test_pair_with_reference_page_mismatch(
    parsed_rows: list[tuple[str, int, str, str | None]],
    reference_rows: list[tuple[str, int, str]],
) -> None:
    with pytest.raises(ValueError) as excinfo:
        parsed_pages.pair_with_reference(
            _parsed(parsed_rows), _reference(reference_rows)
        )

    assert str(excinfo.value) == "page mismatch in 1 document, including ['a']"


def test_pair_with_reference_mismatch_names() -> None:
    parsed = _parsed([(name, 0, "one", None) for name in "fedcba"])
    reference = _reference([("z", 0, "ONE")])

    with pytest.raises(ValueError) as excinfo:
        parsed_pages.pair_with_reference(parsed, reference)

    assert str(excinfo.value) == (
        "page mismatch in 6 documents, including ['a', 'b', 'c', 'd', 'e']"
    )


@pytest.mark.parametrize(
    "frame, column",
    [
        *(
            pytest.param("parsed", column, id=f"parsed-{column}")
            for column in _COLUMNS
        ),
        *(
            pytest.param("reference", column, id=f"reference-{column}")
            for column in ["doc_name", "page_idx", "text"]
        ),
    ],
)
def test_pair_with_reference_missing_column(frame: str, column: str) -> None:
    frames = {
        "parsed": _parsed([("a", 0, "one", None)]),
        "reference": _reference([("a", 0, "ONE")]),
    }
    frames[frame] = frames[frame].drop(columns=[column])

    with pytest.raises(ValueError) as excinfo:
        parsed_pages.pair_with_reference(frames["parsed"], frames["reference"])

    assert str(excinfo.value) == f"{frame} is missing columns ['{column}']"


@pytest.mark.parametrize("column", ["reference", "candidate", "error"])
def test_pair_with_reference_reserved_column(column: str) -> None:
    parsed = _parsed([("a", 0, "one", None)])
    reference = _reference([("a", 0, "ONE")]).assign(**{column: "x"})

    with pytest.raises(ValueError) as excinfo:
        parsed_pages.pair_with_reference(parsed, reference)

    assert str(excinfo.value) == f"reference has reserved columns ['{column}']"


@pytest.mark.parametrize("frame", ["parsed", "reference"])
def test_pair_with_reference_duplicate_pages(frame: str) -> None:
    frames = {
        "parsed": _parsed([("a", 0, "one", None)]),
        "reference": _reference([("a", 0, "ONE")]),
    }
    frames[frame] = pd.concat([frames[frame], frames[frame]])

    with pytest.raises(ValueError) as excinfo:
        parsed_pages.pair_with_reference(frames["parsed"], frames["reference"])

    assert str(excinfo.value) == f"{frame} has duplicate pages"


def test_pair_with_reference_inputs_unchanged() -> None:
    parsed = _parsed([("a", 0, "one", None)])
    reference = _reference([("a", 0, "ONE"), ("b", 0, "TWO")])

    parsed_pages.pair_with_reference(parsed, reference)

    pd.testing.assert_frame_equal(parsed, _parsed([("a", 0, "one", None)]))
    pd.testing.assert_frame_equal(
        reference, _reference([("a", 0, "ONE"), ("b", 0, "TWO")])
    )
