"""Tests for docrag.datasets.parse_cli."""

import logging
import pathlib

import pytest

from docrag.datasets import dataset
from docrag.datasets import parse_cli
from docrag.datasets import parsed_pages
from docrag.file_ops import checksum
from docrag.file_ops import download
from docrag.ingestion import factory
from docrag.ingestion import protocol

_SOURCE = download.Source(
    url="https://example.com/pdfs.zip",
    filename="pdfs.zip",
    checksum=checksum.Checksum(
        algorithm=checksum.Algorithm.SHA256,
        hexdigest="0" * 64,
    ),
)
_DATASET = dataset.Dataset(
    name="sample",
    sources=(_SOURCE,),
    archive_source=_SOURCE,
    extract_dir_name="pdfs",
)
_DATASETS = {_DATASET.name: _DATASET}


class _FakeParser:
    """Returns fixed pages per PDF stem and records parsed paths.

    Attributes:
        paths: Each PDF path passed to parse, in call order.
    """

    def __init__(
        self,
        pages_by_stem: dict[str, list[protocol.ParsedPage]] | None = None,
        *,
        fail_on: str | None = None,
    ) -> None:
        """Initialises the fake parser.

        Args:
            pages_by_stem: Mapping from PDF stem to returned parsed pages.
            fail_on: PDF stem that triggers UnreadablePdfError.
        """
        self._pages_by_stem = pages_by_stem or {}
        self._fail_on = fail_on
        self.paths: list[pathlib.Path] = []

    def parse(self, pdf_path: pathlib.Path) -> list[protocol.ParsedPage]:
        """Returns configured pages or raises UnreadablePdfError.

        Args:
            pdf_path: The PDF file path.

        Returns:
            The configured parsed pages for the PDF stem.

        Raises:
            protocol.UnreadablePdfError: The stem matches fail_on.
        """
        self.paths.append(pdf_path)
        if self._fail_on is not None and pdf_path.stem == self._fail_on:
            raise protocol.UnreadablePdfError(pdf_path)
        return self._pages_by_stem.get(
            pdf_path.stem, [protocol.ParsedPage(text=pdf_path.stem)]
        )


class _FakeParserFactory:
    """Returns a fixed parser and records requested backends.

    Attributes:
        backends: Each backend requested from the factory, in call order.
        parser: The parser returned by the factory.
    """

    def __init__(self, parser: _FakeParser | None = None) -> None:
        """Initialises the fake factory.

        Args:
            parser: The parser to return, or None for a default fake parser.
        """
        self.parser = parser or _FakeParser()
        self.backends: list[factory.ParserBackend] = []

    def __call__(self, backend: factory.ParserBackend) -> protocol.PdfParser:
        """Records the backend and returns the configured parser.

        Args:
            backend: The requested parser backend.

        Returns:
            The configured fake parser.
        """
        self.backends.append(backend)
        return self.parser


def _create_sample_pdfs(data_root: pathlib.Path) -> pathlib.Path:
    """Creates sample PDF files in two nested domain folders.

    Args:
        data_root: The root directory holding datasets.

    Returns:
        The dataset's PDF directory.
    """
    pdf_dir = data_root / _DATASET.name / _DATASET.extract_dir_name
    (pdf_dir / "law").mkdir(parents=True)
    (pdf_dir / "finance").mkdir(parents=True)
    (pdf_dir / "law" / "b.pdf").write_bytes(b"%PDF")
    (pdf_dir / "finance" / "a.pdf").write_bytes(b"%PDF")
    (pdf_dir / "finance" / "notes.txt").write_text("ignored", encoding="utf-8")
    return pdf_dir


def _error_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Returns the messages logged at error level.

    Args:
        caplog: The fixture that captured the log records.
    """
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
    ]


def test_parse_dataset_parses_and_saves(tmp_path: pathlib.Path) -> None:
    _create_sample_pdfs(tmp_path)
    parser_factory = _FakeParserFactory(
        _FakeParser(
            {
                "a": [
                    protocol.ParsedPage(text="alpha 0"),
                    protocol.ParsedPage(text="alpha 1"),
                ],
                "b": [
                    protocol.ParsedPage(text="", error="page could not load")
                ],
            }
        )
    )

    output_path = parse_cli.parse_dataset(
        tmp_path,
        _DATASET,
        factory.ParserBackend.PYPDFIUM2,
        create_parser=parser_factory,
    )

    expected_path = (
        tmp_path
        / _DATASET.name
        / parsed_pages.PARSED_PAGES_DIR
        / "pypdfium2.parquet"
    )
    assert output_path == expected_path
    assert parser_factory.backends == [factory.ParserBackend.PYPDFIUM2]
    loaded = parsed_pages.load_parsed_pages(output_path)
    assert loaded["doc_name"].tolist() == [
        "finance/a",
        "finance/a",
        "law/b",
    ]
    assert loaded["page_idx"].tolist() == [0, 1, 0]
    assert loaded["text"].tolist() == ["alpha 0", "alpha 1", ""]
    assert loaded["error"].isna().tolist() == [True, True, False]
    assert loaded["error"].iloc[2] == "page could not load"


def test_parse_dataset_existing_output_no_overwrite(
    tmp_path: pathlib.Path,
) -> None:
    _create_sample_pdfs(tmp_path)
    first_factory = _FakeParserFactory(
        _FakeParser({"a": [protocol.ParsedPage(text="old")]})
    )
    output_path = parse_cli.parse_dataset(
        tmp_path,
        _DATASET,
        factory.ParserBackend.PYPDFIUM2,
        create_parser=first_factory,
    )
    second_factory = _FakeParserFactory(
        _FakeParser({"a": [protocol.ParsedPage(text="new")]})
    )

    with pytest.raises(FileExistsError) as excinfo:
        parse_cli.parse_dataset(
            tmp_path,
            _DATASET,
            factory.ParserBackend.PYPDFIUM2,
            create_parser=second_factory,
        )

    assert str(excinfo.value) == f"{output_path}: already exists"
    assert not second_factory.backends
    assert not second_factory.parser.paths
    loaded = parsed_pages.load_parsed_pages(output_path)
    assert loaded["text"].tolist() == ["old", "b"]


def test_parse_dataset_existing_output_overwrite(
    tmp_path: pathlib.Path,
) -> None:
    _create_sample_pdfs(tmp_path)
    first_factory = _FakeParserFactory(
        _FakeParser({"a": [protocol.ParsedPage(text="old")]})
    )
    output_path = parse_cli.parse_dataset(
        tmp_path,
        _DATASET,
        factory.ParserBackend.PYPDFIUM2,
        create_parser=first_factory,
    )
    second_factory = _FakeParserFactory(
        _FakeParser({"a": [protocol.ParsedPage(text="new")]})
    )

    result = parse_cli.parse_dataset(
        tmp_path,
        _DATASET,
        factory.ParserBackend.PYPDFIUM2,
        create_parser=second_factory,
        overwrite=True,
    )

    assert result == output_path
    loaded = parsed_pages.load_parsed_pages(output_path)
    assert loaded["text"].tolist() == ["new", "b"]


def test_parse_dataset_missing_pdf_dir(tmp_path: pathlib.Path) -> None:
    pdf_dir = tmp_path / _DATASET.name / _DATASET.extract_dir_name
    parser_factory = _FakeParserFactory()

    with pytest.raises(FileNotFoundError) as excinfo:
        parse_cli.parse_dataset(
            tmp_path,
            _DATASET,
            factory.ParserBackend.PYPDFIUM2,
            create_parser=parser_factory,
        )

    assert str(excinfo.value) == f"{pdf_dir}: does not exist"
    assert not (
        tmp_path / _DATASET.name / parsed_pages.PARSED_PAGES_DIR
    ).exists()
    assert not parser_factory.backends


def test_parse_dataset_empty_pdf_dir(tmp_path: pathlib.Path) -> None:
    pdf_dir = tmp_path / _DATASET.name / _DATASET.extract_dir_name
    pdf_dir.mkdir(parents=True)
    parser_factory = _FakeParserFactory()

    with pytest.raises(ValueError) as excinfo:
        parse_cli.parse_dataset(
            tmp_path,
            _DATASET,
            factory.ParserBackend.PYPDFIUM2,
            create_parser=parser_factory,
        )

    assert str(excinfo.value) == f"{pdf_dir}: contains no PDF files"
    assert not (
        tmp_path / _DATASET.name / parsed_pages.PARSED_PAGES_DIR
    ).exists()
    assert not parser_factory.backends


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("dataset-dir", id="dataset-dir"),
        pytest.param("pdf-dir", id="pdf-dir"),
        pytest.param("output-dir", id="output-dir"),
        pytest.param("output-path", id="output-path"),
        pytest.param("pdf-file", id="pdf-file"),
    ],
)
def test_parse_dataset_symlink_path(
    tmp_path: pathlib.Path, target: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "target.bin"
    outside_file.write_bytes(b"data")

    dataset_dir = tmp_path / _DATASET.name
    pdf_dir = dataset_dir / _DATASET.extract_dir_name
    output_dir = dataset_dir / parsed_pages.PARSED_PAGES_DIR
    output_path = output_dir / "pypdfium2.parquet"

    if target == "dataset-dir":
        dataset_dir.symlink_to(outside)
    elif target == "pdf-dir":
        dataset_dir.mkdir()
        pdf_dir.symlink_to(outside)
    elif target == "output-dir":
        _create_sample_pdfs(tmp_path)
        output_dir.symlink_to(outside)
    elif target == "output-path":
        _create_sample_pdfs(tmp_path)
        output_dir.mkdir()
        output_path.symlink_to(outside_file)
    else:
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "linked.pdf").symlink_to(outside_file)

    parser_factory = _FakeParserFactory()
    with pytest.raises(OSError, match="symbolic links are not allowed"):
        parse_cli.parse_dataset(
            tmp_path,
            _DATASET,
            factory.ParserBackend.PYPDFIUM2,
            create_parser=parser_factory,
        )

    assert not parser_factory.backends


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("dataset-dir-file", id="dataset-dir-file"),
        pytest.param("pdf-dir-file", id="pdf-dir-file"),
        pytest.param("output-dir-file", id="output-dir-file"),
        pytest.param("output-path-dir", id="output-path-dir"),
        pytest.param("pdf-path-dir", id="pdf-path-dir"),
    ],
)
def test_parse_dataset_wrong_file_type(
    tmp_path: pathlib.Path, target: str
) -> None:
    dataset_dir = tmp_path / _DATASET.name
    pdf_dir = dataset_dir / _DATASET.extract_dir_name
    output_dir = dataset_dir / parsed_pages.PARSED_PAGES_DIR
    output_path = output_dir / "pypdfium2.parquet"

    if target == "dataset-dir-file":
        dataset_dir.write_text("file", encoding="utf-8")
    elif target == "pdf-dir-file":
        dataset_dir.mkdir()
        pdf_dir.write_text("file", encoding="utf-8")
    elif target == "output-dir-file":
        _create_sample_pdfs(tmp_path)
        output_dir.write_text("file", encoding="utf-8")
    elif target == "output-path-dir":
        _create_sample_pdfs(tmp_path)
        output_path.mkdir(parents=True)
    else:
        (pdf_dir / "folder.pdf").mkdir(parents=True)

    parser_factory = _FakeParserFactory()
    with pytest.raises(FileExistsError):
        parse_cli.parse_dataset(
            tmp_path,
            _DATASET,
            factory.ParserBackend.PYPDFIUM2,
            create_parser=parser_factory,
        )

    assert not parser_factory.backends


def test_parse_dataset_unreadable_pdf(tmp_path: pathlib.Path) -> None:
    pdf_dir = _create_sample_pdfs(tmp_path)
    parser_factory = _FakeParserFactory(_FakeParser(fail_on="b"))

    with pytest.raises(protocol.UnreadablePdfError) as excinfo:
        parse_cli.parse_dataset(
            tmp_path,
            _DATASET,
            factory.ParserBackend.PYPDFIUM2,
            create_parser=parser_factory,
        )

    assert excinfo.value.path == pdf_dir / "law" / "b.pdf"
    assert not (
        tmp_path / _DATASET.name / parsed_pages.PARSED_PAGES_DIR
    ).exists()


def test_main_success(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    _create_sample_pdfs(tmp_path)
    parser = _FakeParser()

    status = parse_cli.main(
        ["sample", "--backend", "pypdfium2"],
        data_root=tmp_path,
        datasets=_DATASETS,
        create_parser=lambda _: parser,
    )

    output_path = (
        tmp_path
        / _DATASET.name
        / parsed_pages.PARSED_PAGES_DIR
        / "pypdfium2.parquet"
    )
    assert status == 0
    assert output_path.is_file()
    assert not _error_messages(caplog)


def test_main_existing_output_no_overwrite(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    _create_sample_pdfs(tmp_path)
    parse_cli.main(
        ["sample", "--backend", "pypdfium2"],
        data_root=tmp_path,
        datasets=_DATASETS,
        create_parser=lambda _: _FakeParser(
            {"a": [protocol.ParsedPage(text="old")]}
        ),
    )
    output_path = (
        tmp_path
        / _DATASET.name
        / parsed_pages.PARSED_PAGES_DIR
        / "pypdfium2.parquet"
    )

    status = parse_cli.main(
        ["sample", "--backend", "pypdfium2"],
        data_root=tmp_path,
        datasets=_DATASETS,
        create_parser=lambda _: _FakeParser(
            {"a": [protocol.ParsedPage(text="new")]}
        ),
    )

    assert status == 1
    assert _error_messages(caplog) == [f"{output_path}: already exists"]
    loaded = parsed_pages.load_parsed_pages(output_path)
    assert loaded["text"].tolist() == ["old", "b"]


def test_main_existing_output_overwrite(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    _create_sample_pdfs(tmp_path)
    parse_cli.main(
        ["sample", "--backend", "pypdfium2"],
        data_root=tmp_path,
        datasets=_DATASETS,
        create_parser=lambda _: _FakeParser(
            {"a": [protocol.ParsedPage(text="old")]}
        ),
    )
    output_path = (
        tmp_path
        / _DATASET.name
        / parsed_pages.PARSED_PAGES_DIR
        / "pypdfium2.parquet"
    )

    status = parse_cli.main(
        ["sample", "--backend", "pypdfium2", "--overwrite"],
        data_root=tmp_path,
        datasets=_DATASETS,
        create_parser=lambda _: _FakeParser(
            {"a": [protocol.ParsedPage(text="new")]}
        ),
    )

    assert status == 0
    assert not _error_messages(caplog)
    loaded = parsed_pages.load_parsed_pages(output_path)
    assert loaded["text"].tolist() == ["new", "b"]


def test_main_missing_pdf_dir(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_dir = tmp_path / _DATASET.name / _DATASET.extract_dir_name

    status = parse_cli.main(
        ["sample", "--backend", "pypdfium2"],
        data_root=tmp_path,
        datasets=_DATASETS,
        create_parser=lambda _: _FakeParser(),
    )

    assert status == 1
    assert _error_messages(caplog) == [f"{pdf_dir}: does not exist"]


def test_main_empty_pdf_dir(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_dir = tmp_path / _DATASET.name / _DATASET.extract_dir_name
    pdf_dir.mkdir(parents=True)

    status = parse_cli.main(
        ["sample", "--backend", "pypdfium2"],
        data_root=tmp_path,
        datasets=_DATASETS,
        create_parser=lambda _: _FakeParser(),
    )

    assert status == 1
    assert _error_messages(caplog) == [f"{pdf_dir}: contains no PDF files"]


def test_main_unreadable_pdf(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_dir = _create_sample_pdfs(tmp_path)
    broken_pdf = pdf_dir / "law" / "b.pdf"

    status = parse_cli.main(
        ["sample", "--backend", "pypdfium2"],
        data_root=tmp_path,
        datasets=_DATASETS,
        create_parser=lambda _: _FakeParser(fail_on="b"),
    )

    assert status == 1
    assert _error_messages(caplog) == [f"{broken_pdf}: not a readable PDF"]


def test_main_unknown_dataset(tmp_path: pathlib.Path) -> None:
    parser_factory = _FakeParserFactory()

    with pytest.raises(SystemExit) as excinfo:
        parse_cli.main(
            ["unknown", "--backend", "pypdfium2"],
            data_root=tmp_path,
            datasets=_DATASETS,
            create_parser=parser_factory,
        )

    assert excinfo.value.code == 2
    assert not parser_factory.backends


def test_main_unknown_backend(tmp_path: pathlib.Path) -> None:
    parser_factory = _FakeParserFactory()

    with pytest.raises(SystemExit) as excinfo:
        parse_cli.main(
            ["sample", "--backend", "unknown"],
            data_root=tmp_path,
            datasets=_DATASETS,
            create_parser=parser_factory,
        )

    assert excinfo.value.code == 2
    assert not parser_factory.backends


def test_main_default_datasets_and_backends(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_cli.main(["--help"], data_root=tmp_path)

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "{ohr_bench}" in help_text
    assert "{pypdfium2}" in help_text
