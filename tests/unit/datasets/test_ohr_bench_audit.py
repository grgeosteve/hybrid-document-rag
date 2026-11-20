"""Tests for docrag.datasets.ohr_bench_audit."""

from collections.abc import Sequence
import json
import pathlib
from typing import Any

import pandas as pd
import pytest

from docrag.datasets import ohr_bench_audit

_PAGE_COLUMNS = ["domain", "doc_name", "page_idx", "gt_text"]
_QA_COLUMNS = [
    "doc_name",
    "ID",
    "questions",
    "answers",
    "doc_type",
    "answer_form",
    "evidence_source",
    "evidence_context",
    "evidence_page_no",
]
_AUDIT_QA_COLUMNS = [
    "domain" if column == "doc_type" else column for column in _QA_COLUMNS
]
_LANGUAGE_COLUMNS = [
    "texts",
    "confidence_median",
    "confidence_p10",
    "confidence_min",
]
_LENGTH_COLUMNS = ["count", "median", "p90", "max"]

_PAGE_ROWS = [
    ("law", "law/a", 0, "court one"),
    ("law", "law/a", 1, "   "),
    ("law", "law/b", 0, "tribunal"),
    ("news", "news/c", 0, "news one two"),
    ("news", "news/c", 1, "news three"),
]
_DETECTIONS = {
    "court one": ("ENGLISH", 0.9),
    "tribunal": ("FRENCH", 0.8),
    "news one two": ("ENGLISH", 0.5),
    "news three": ("ENGLISH", 0.7),
    "law question": ("ENGLISH", 0.95),
    "question de droit": ("FRENCH", 0.6),
    "news question": ("ENGLISH", 0.4),
}


class _FakeDetect:
    """Returns fixed detections and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: Sequence[str]) -> list[tuple[str, float]]:
        self.calls += 1
        return [_DETECTIONS.get(text, ("UNKNOWN", 0.0)) for text in texts]


def _qa_record(**fields: Any) -> dict[str, Any]:
    """Returns a QA record with placeholder values for the unset fields.

    Args:
        **fields: The values of the fields to set.
    """
    record: dict[str, Any] = {
        "doc_name": "law/a",
        "ID": "1",
        "questions": "law question",
        "answers": "answer",
        "doc_type": "law",
        "answer_form": "String",
        "evidence_source": "text",
        "evidence_context": "context",
        "evidence_page_no": 0,
    }
    record.update(fields)
    return record


_QA_RECORDS = [
    _qa_record(
        ID="1",
        questions="law question",
        answers="one two",
        doc_type="law",
        answer_form="String",
        evidence_source="text",
    ),
    _qa_record(
        ID="2",
        questions="question de droit",
        answers="three",
        doc_type="law",
        answer_form="Numeric",
        evidence_source="table",
    ),
    _qa_record(
        ID="3",
        doc_name="news/c",
        questions="news question",
        answers="a b c d",
        doc_type="news",
        answer_form="String",
        evidence_source="text",
    ),
]


def _pages() -> pd.DataFrame:
    """Returns the sample pages."""
    return pd.DataFrame(_PAGE_ROWS, columns=_PAGE_COLUMNS)


def _qa() -> pd.DataFrame:
    """Returns the sample QA pairs."""
    return pd.DataFrame.from_records(_QA_RECORDS)


def _audit(detect: _FakeDetect | None = None) -> ohr_bench_audit.OHRBenchAudit:
    """Returns an audit over the sample pages and QA pairs.

    Args:
        detect: The fake detector, or None for a new one.
    """
    return ohr_bench_audit.OHRBenchAudit(
        _pages(), _qa(), detect=detect or _FakeDetect()
    )


def _write_files(
    directory: pathlib.Path, pages: pd.DataFrame, records: object
) -> tuple[pathlib.Path, pathlib.Path]:
    """Writes a parquet file and a QA file.

    Args:
        directory: The directory to write into.
        pages: The pages to write as parquet.
        records: The value to write as JSON.

    Returns:
        The parquet path and the QA path.
    """
    parquet_path = directory / "pages.parquet"
    qa_path = directory / "qa.json"
    pages.to_parquet(parquet_path)
    qa_path.write_text(json.dumps(records), encoding="utf-8")
    return parquet_path, qa_path


def _row(frame: pd.DataFrame, key: object) -> list[Any]:
    """Returns the values of one row of a frame.

    Args:
        frame: The frame to read.
        key: The row's index label.
    """
    values: list[Any] = frame.loc[[key]].to_numpy().tolist()[0]
    return values


def test_detect_languages() -> None:
    results = ohr_bench_audit.detect_languages(
        [
            "The court ruled on the appeal this morning.",
            "Le tribunal a rendu sa décision ce matin.",
        ]
    )

    assert [language for language, _ in results] == ["ENGLISH", "FRENCH"]
    assert all(0 < confidence <= 1 for _, confidence in results)


def test_detect_languages_blank_text() -> None:
    results = ohr_bench_audit.detect_languages(["", "  \n"])

    assert results == [("UNKNOWN", 0.0), ("UNKNOWN", 0.0)]


def test_detect_languages_no_language() -> None:
    [(_, confidence)] = ohr_bench_audit.detect_languages(["1234"])

    assert confidence == 0.0


def test_detect_languages_batch_size_one() -> None:
    texts = [
        "The court ruled on the appeal this morning.",
        "",
        "Le tribunal a rendu sa décision ce matin.",
    ]

    one_at_a_time = ohr_bench_audit.detect_languages(texts, batch_size=1)
    default = ohr_bench_audit.detect_languages(texts)

    assert [language for language, _ in one_at_a_time] == [
        language for language, _ in default
    ]
    assert [confidence for _, confidence in one_at_a_time] == pytest.approx(
        [confidence for _, confidence in default]
    )


def test_detect_languages_no_texts() -> None:
    assert not ohr_bench_audit.detect_languages([])


@pytest.mark.parametrize(
    "batch_size",
    [pytest.param(0, id="zero"), pytest.param(-1, id="negative")],
)
def test_detect_languages_invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError) as excinfo:
        ohr_bench_audit.detect_languages(["text"], batch_size=batch_size)

    assert str(excinfo.value) == (
        f"batch_size must be at least 1, got {batch_size}"
    )


@pytest.mark.parametrize("column", _PAGE_COLUMNS)
def test_init_missing_page_column(column: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        ohr_bench_audit.OHRBenchAudit(
            _pages().drop(columns=[column]), _qa(), detect=_FakeDetect()
        )

    assert str(excinfo.value) == (
        f"pages data is missing the following required columns: ['{column}']"
    )


@pytest.mark.parametrize("column", _QA_COLUMNS)
def test_init_missing_qa_column(column: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        ohr_bench_audit.OHRBenchAudit(
            _pages(), _qa().drop(columns=[column]), detect=_FakeDetect()
        )

    assert str(excinfo.value) == (
        f"qa data is missing the following required columns: ['{column}']"
    )


def test_init_keeps_required_columns() -> None:
    audit = ohr_bench_audit.OHRBenchAudit(
        _pages().assign(extra="x"),
        _qa().assign(extra="x"),
        detect=_FakeDetect(),
    )

    assert list(audit.pages.columns) == _PAGE_COLUMNS
    assert list(audit.qa.columns) == _AUDIT_QA_COLUMNS


def test_from_files_reads_both(tmp_path: pathlib.Path) -> None:
    parquet_path, qa_path = _write_files(
        tmp_path, _pages().assign(extra="x"), _QA_RECORDS
    )

    audit = ohr_bench_audit.OHRBenchAudit.from_files(
        parquet_path, qa_path, detect=_FakeDetect()
    )

    assert list(audit.pages.columns) == _PAGE_COLUMNS
    assert audit.pages["gt_text"].tolist() == [row[3] for row in _PAGE_ROWS]
    assert list(audit.qa.columns) == _AUDIT_QA_COLUMNS
    assert audit.qa["ID"].tolist() == ["1", "2", "3"]


def test_from_files_keeps_json_types(tmp_path: pathlib.Path) -> None:
    records = [
        _qa_record(answers="42", evidence_page_no=3),
        _qa_record(answers="7.5", evidence_page_no=[4, 5]),
    ]
    parquet_path, qa_path = _write_files(tmp_path, _pages(), records)

    audit = ohr_bench_audit.OHRBenchAudit.from_files(
        parquet_path, qa_path, detect=_FakeDetect()
    )

    assert audit.qa["answers"].tolist() == ["42", "7.5"]
    assert audit.qa["evidence_page_no"].tolist() == [3, [4, 5]]


def test_from_files_missing_page_column(tmp_path: pathlib.Path) -> None:
    parquet_path, qa_path = _write_files(
        tmp_path, _pages().drop(columns=["gt_text"]), _QA_RECORDS
    )

    with pytest.raises(ValueError) as excinfo:
        ohr_bench_audit.OHRBenchAudit.from_files(
            parquet_path, qa_path, detect=_FakeDetect()
        )

    assert str(excinfo.value) == (
        "pages data is missing the following required columns: ['gt_text']"
    )


def test_from_files_top_level_not_list(tmp_path: pathlib.Path) -> None:
    parquet_path, qa_path = _write_files(
        tmp_path, _pages(), {"records": _QA_RECORDS}
    )

    with pytest.raises(ValueError) as excinfo:
        ohr_bench_audit.OHRBenchAudit.from_files(
            parquet_path, qa_path, detect=_FakeDetect()
        )

    assert str(excinfo.value) == f"{qa_path}: top level is not a list"


@pytest.mark.parametrize(
    "missing",
    [pytest.param("parquet", id="parquet"), pytest.param("qa", id="qa")],
)
def test_from_files_missing_file(tmp_path: pathlib.Path, missing: str) -> None:
    parquet_path, qa_path = _write_files(tmp_path, _pages(), _QA_RECORDS)
    if missing == "parquet":
        parquet_path = tmp_path / "absent.parquet"
    else:
        qa_path = tmp_path / "absent.json"

    with pytest.raises(FileNotFoundError):
        ohr_bench_audit.OHRBenchAudit.from_files(
            parquet_path, qa_path, detect=_FakeDetect()
        )


def test_from_files_not_parquet(tmp_path: pathlib.Path) -> None:
    _, qa_path = _write_files(tmp_path, _pages(), _QA_RECORDS)
    text_path = tmp_path / "pages.txt"
    text_path.write_text("plain text", encoding="utf-8")

    with pytest.raises(ValueError):
        ohr_bench_audit.OHRBenchAudit.from_files(
            text_path, qa_path, detect=_FakeDetect()
        )


def test_from_files_not_json(tmp_path: pathlib.Path) -> None:
    parquet_path, qa_path = _write_files(tmp_path, _pages(), _QA_RECORDS)
    qa_path.write_text("not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        ohr_bench_audit.OHRBenchAudit.from_files(
            parquet_path, qa_path, detect=_FakeDetect()
        )


def test_missing_values_counts() -> None:
    pages = pd.DataFrame(
        [
            ("law", "law/a", 0, "text"),
            ("law", "law/a", 1, "  "),
            ("law", "law/a", 2, None),
        ],
        columns=_PAGE_COLUMNS,
    )
    qa = pd.DataFrame.from_records(
        [
            _qa_record(answers=None, evidence_context=["", "context"]),
            _qa_record(questions="\n"),
        ]
    )

    result = ohr_bench_audit.OHRBenchAudit(
        pages, qa, detect=_FakeDetect()
    ).missing_values()

    assert list(result.index.names) == ["source", "column"]
    assert list(result.columns) == ["null", "empty"]
    assert len(result) == len(_PAGE_COLUMNS) + len(_QA_COLUMNS)
    assert _row(result, ("pages", "gt_text")) == [1, 1]
    assert _row(result, ("pages", "page_idx")) == [0, 0]
    assert _row(result, ("qa", "answers")) == [1, 0]
    assert _row(result, ("qa", "questions")) == [0, 1]
    assert _row(result, ("qa", "evidence_context")) == [0, 0]


def test_empty_pages_per_domain() -> None:
    result = _audit().empty_pages()

    assert list(result.columns) == ["pages", "empty"]
    assert _row(result, "law") == [3, 1]
    assert _row(result, "news") == [2, 0]


def test_page_languages_grouped() -> None:
    result = _audit().page_languages()

    assert list(result.columns) == _LANGUAGE_COLUMNS
    assert list(result.index) == [
        ("law", "ENGLISH"),
        ("law", "FRENCH"),
        ("news", "ENGLISH"),
    ]
    assert _row(result, ("law", "ENGLISH")) == pytest.approx([1, 0.9, 0.9, 0.9])
    assert _row(result, ("news", "ENGLISH")) == pytest.approx(
        [2, 0.6, 0.52, 0.5]
    )


def test_page_languages_skips_empty_pages() -> None:
    result = _audit().page_languages()

    assert ("law", "UNKNOWN") not in result.index
    assert result["texts"].sum() == 4


def test_question_languages_grouped() -> None:
    result = _audit().question_languages()

    assert list(result.columns) == _LANGUAGE_COLUMNS
    assert list(result.index) == [
        ("law", "ENGLISH"),
        ("law", "FRENCH"),
        ("news", "ENGLISH"),
    ]
    assert _row(result, ("law", "FRENCH")) == pytest.approx([1, 0.6, 0.6, 0.6])
    assert _row(result, ("news", "ENGLISH")) == pytest.approx(
        [1, 0.4, 0.4, 0.4]
    )


def test_detections_computed_once() -> None:
    detect = _FakeDetect()
    audit = _audit(detect)

    for _ in range(2):
        audit.page_languages()
        audit.question_languages()
        audit.english_share(0.5)

    assert detect.calls == 2


def test_english_share_threshold() -> None:
    result = _audit().english_share(0.7)

    assert list(result.columns) == ["pages", "questions"]
    assert _row(result, "law") == pytest.approx([0.5, 0.5])
    assert _row(result, "news") == pytest.approx([0.5, 0.0])


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-0.1, id="negative"),
        pytest.param(1.1, id="above-one"),
    ],
)
def test_english_share_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        _audit().english_share(threshold)


@pytest.mark.parametrize(
    ("by_domain", "names", "expected"),
    [
        pytest.param(
            False,
            ["answer_form"],
            {"Numeric": [1, 1, 1, 1], "String": [2, 3, 3.8, 4]},
            id="all-domains",
        ),
        pytest.param(
            True,
            ["domain", "answer_form"],
            {
                ("law", "Numeric"): [1, 1, 1, 1],
                ("law", "String"): [1, 2, 2, 2],
                ("news", "String"): [1, 4, 4, 4],
            },
            id="by-domain",
        ),
    ],
)
def test_answer_lengths_by_form(
    by_domain: bool, names: list[str], expected: dict[Any, list[float]]
) -> None:
    result = _audit().answer_lengths_by_form(by_domain=by_domain)

    assert list(result.index.names) == names
    assert list(result.columns) == _LENGTH_COLUMNS
    assert list(result.index) == list(expected)
    for key, values in expected.items():
        assert _row(result, key) == pytest.approx(values)


@pytest.mark.parametrize(
    ("by_domain", "names", "expected"),
    [
        pytest.param(
            False,
            ["evidence_source"],
            {"table": [1, 1, 1, 1], "text": [2, 3, 3.8, 4]},
            id="all-domains",
        ),
        pytest.param(
            True,
            ["domain", "evidence_source"],
            {
                ("law", "table"): [1, 1, 1, 1],
                ("law", "text"): [1, 2, 2, 2],
                ("news", "text"): [1, 4, 4, 4],
            },
            id="by-domain",
        ),
    ],
)
def test_answer_lengths_by_evidence(
    by_domain: bool, names: list[str], expected: dict[Any, list[float]]
) -> None:
    result = _audit().answer_lengths_by_evidence(by_domain=by_domain)

    assert list(result.index.names) == names
    assert list(result.columns) == _LENGTH_COLUMNS
    assert list(result.index) == list(expected)
    for key, values in expected.items():
        assert _row(result, key) == pytest.approx(values)


def test_document_lengths_per_domain() -> None:
    result = _audit().document_lengths()

    assert list(result.columns) == [
        "documents",
        "pages_total",
        "pages_mean",
        "pages_median",
        "pages_p90",
        "pages_max",
        "characters_median",
        "characters_p90",
        "characters_max",
        "words_median",
        "words_p90",
        "words_max",
    ]
    assert list(result.index) == ["law", "news", "overall"]
    assert _row(result, "law") == pytest.approx(
        [2, 3, 1.5, 1.5, 1.9, 2, 10, 11.6, 12, 1.5, 1.9, 2]
    )
    assert _row(result, "news") == pytest.approx(
        [1, 2, 2, 2, 2, 2, 22, 22, 22, 5, 5, 5]
    )
    assert _row(result, "overall") == pytest.approx(
        [3, 5, 5 / 3, 2, 2, 2, 12, 20, 22, 2, 4.4, 5]
    )
